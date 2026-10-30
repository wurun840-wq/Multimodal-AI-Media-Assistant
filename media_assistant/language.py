"""Part 2 -- LLM analysis with validated structured output.

The model is asked for JSON, and the JSON is then checked in Python
before anything downstream is allowed to use it. Validation covers the
three things the brief requires -- required keys, field types, and that
list-valued fields really are lists -- plus the cardinality rules the
brief states in prose (3-6 key points, at least one topic).

When validation fails, the specific problems are fed back to the model
and it is asked to fix them, up to ``config.analysis_max_retries`` times.
If it still fails, :class:`AnalysisValidationError` is raised. A partial
or malformed analysis is never returned.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import Config
from .exceptions import AnalysisValidationError
from .llm import LocalLLM, extract_json
from .utils import LOGGER, truncate

# Mirrors the constant in the project brief's starter notebook.
REQUIRED_ANALYSIS_KEYS = {
    "title",
    "summary",
    "key_points",
    "topics",
    "action_items",
    "entities",
    "languages",
    "follow_up_questions",
}

# How much transcript to send. A 3-8 minute recording is far below this;
# the cap exists so a longer input degrades predictably instead of
# silently overflowing the context window.
MAX_TRANSCRIPT_CHARS = 24_000


@dataclass(frozen=True)
class FieldSpec:
    """Declarative rule for one field of the analysis object."""

    name: str
    types: tuple[type, ...]
    description: str
    item_types: tuple[type, ...] | None = None
    min_items: int | None = None
    max_items: int | None = None

    @property
    def is_list(self) -> bool:
        return self.types == (list,)

    def type_names(self) -> str:
        return " or ".join(t.__name__ for t in self.types)


# ``entities`` and ``action_items`` accept either a plain string or an
# object, because a 4B model is markedly more reliable when it is not
# forced into one nested shape. Both forms are normalised to objects by
# `normalise_analysis`, so downstream code sees a single structure.
ANALYSIS_SCHEMA: tuple[FieldSpec, ...] = (
    FieldSpec("title", (str,), "a short title for the recording"),
    FieldSpec("summary", (str,), "a concise summary, 3-5 sentences"),
    FieldSpec(
        "key_points", (list,), "3 to 6 key points",
        item_types=(str,), min_items=3, max_items=6,
    ),
    FieldSpec(
        "topics", (list,), "the topics discussed",
        item_types=(str,), min_items=1,
    ),
    FieldSpec(
        "action_items", (list,),
        'action items or decisions, each as {"description": string, "owner": '
        'string or null, "due": string or null}; an empty list if the '
        "recording has none",
        item_types=(str, dict),
    ),
    FieldSpec(
        "entities", (list,),
        'people and organisations mentioned, each as {"name": string, "type": '
        '"person" | "organisation" | "product"}',
        item_types=(str, dict),
    ),
    FieldSpec(
        "languages", (list,), "language names observed in the transcript",
        item_types=(str,), min_items=1,
    ),
    FieldSpec(
        "follow_up_questions", (list,), "3 to 5 useful follow-up questions",
        item_types=(str,), min_items=1,
    ),
)

SYSTEM_PROMPT = (
    "You are a careful meeting-analysis assistant. You read a transcript and "
    "return a single JSON object describing it. You never invent facts that "
    "are not in the transcript: if the transcript does not contain something, "
    "you use an empty list or say so plainly in the summary. You reply with "
    "JSON only, with no commentary before or after it.\n\n"
    "You always write every string value in {language}, whatever language the "
    "transcript is in. Names of people, products and organisations keep their "
    "original spelling."
)


def describe_schema() -> str:
    """Render the schema as prompt text, so it cannot drift from the validator."""
    lines = []
    for spec in ANALYSIS_SCHEMA:
        constraint = ""
        if spec.min_items is not None and spec.max_items is not None:
            constraint = f" ({spec.min_items}-{spec.max_items} items)"
        elif spec.min_items is not None:
            constraint = f" (at least {spec.min_items} item)"
        kind = "array" if spec.is_list else spec.type_names()
        lines.append(f'  "{spec.name}": {kind}{constraint} — {spec.description}')
    return "{\n" + ",\n".join(lines) + "\n}"


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------

def check_analysis(candidate: object) -> list[str]:
    """Return a list of human-readable problems; empty means valid.

    Kept separate from :func:`validate_analysis` because the retry loop
    needs the problems as text to hand back to the model, not as an
    exception.
    """
    problems: list[str] = []

    if not isinstance(candidate, dict):
        return [f"top-level value must be a JSON object, got {type(candidate).__name__}"]

    missing = sorted(REQUIRED_ANALYSIS_KEYS - candidate.keys())
    if missing:
        problems.append(f"missing required keys: {', '.join(missing)}")

    for spec in ANALYSIS_SCHEMA:
        if spec.name not in candidate:
            continue
        value = candidate[spec.name]

        if not isinstance(value, spec.types):
            problems.append(
                f"'{spec.name}' must be {spec.type_names()}, got {type(value).__name__}"
            )
            continue

        if isinstance(value, str) and not value.strip():
            problems.append(f"'{spec.name}' is empty")
            continue

        if not spec.is_list:
            continue

        # From here on the field is a list, which is what the brief asks
        # us to verify explicitly rather than assume.
        if spec.item_types:
            bad = [
                f"index {i} is {type(item).__name__}"
                for i, item in enumerate(value)
                if not isinstance(item, spec.item_types)
            ]
            if bad:
                allowed = " or ".join(t.__name__ for t in spec.item_types)
                problems.append(
                    f"'{spec.name}' must contain only {allowed}: {', '.join(bad[:3])}"
                )
        if spec.min_items is not None and len(value) < spec.min_items:
            problems.append(
                f"'{spec.name}' needs at least {spec.min_items} items, got {len(value)}"
            )
        if spec.max_items is not None and len(value) > spec.max_items:
            problems.append(
                f"'{spec.name}' allows at most {spec.max_items} items, got {len(value)}"
            )

    for item in candidate.get("entities", []) if isinstance(candidate.get("entities"), list) else []:
        if isinstance(item, dict) and not str(item.get("name", "")).strip():
            problems.append("every object in 'entities' needs a non-empty 'name'")
            break

    return problems


def normalise_analysis(analysis: dict) -> dict:
    """Coerce the permitted shapes into one consistent structure.

    ``entities`` becomes a list of ``{"name", "type"}`` and
    ``action_items`` a list of ``{"description", "owner", "due"}``,
    whichever form the model chose. Run only after validation passes.
    """
    normalised = dict(analysis)

    entities = []
    for item in analysis.get("entities", []):
        if isinstance(item, str):
            entities.append({"name": item.strip(), "type": "unspecified"})
        else:
            entities.append(
                {
                    "name": str(item.get("name", "")).strip(),
                    "type": str(item.get("type") or item.get("category") or "unspecified").strip(),
                }
            )
    normalised["entities"] = [e for e in entities if e["name"]]

    actions = []
    for item in analysis.get("action_items", []):
        if isinstance(item, str):
            actions.append({"description": item.strip(), "owner": None, "due": None})
        else:
            description = item.get("description") or item.get("action") or item.get("item") or ""
            actions.append(
                {
                    "description": str(description).strip(),
                    "owner": item.get("owner") or item.get("assignee"),
                    "due": item.get("due") or item.get("deadline") or item.get("due_date"),
                }
            )
    normalised["action_items"] = [a for a in actions if a["description"]]

    for key in ("key_points", "topics", "languages", "follow_up_questions"):
        if key in normalised:
            normalised[key] = [str(v).strip() for v in normalised[key] if str(v).strip()]

    for key in ("title", "summary"):
        if key in normalised:
            normalised[key] = str(normalised[key]).strip()

    return normalised


def validate_analysis(candidate: object) -> dict:
    """Validate and normalise, or raise :class:`AnalysisValidationError`."""
    problems = check_analysis(candidate)
    if problems:
        raise AnalysisValidationError(
            "analysis failed schema validation: " + "; ".join(problems),
            problems=problems,
        )
    return normalise_analysis(candidate)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# generation
# --------------------------------------------------------------------------

def _transcript_for_prompt(segments_or_text: list[dict] | str) -> str:
    """Accept either raw text or segments, and cap the length sent."""
    if isinstance(segments_or_text, str):
        text = segments_or_text
    else:
        text = "\n".join(
            f"[{s['start']:.1f}-{s['end']:.1f}] {s['text']}" for s in segments_or_text
        )
    text = text.strip()
    if len(text) <= MAX_TRANSCRIPT_CHARS:
        return text
    head = text[: MAX_TRANSCRIPT_CHARS // 2]
    tail = text[-MAX_TRANSCRIPT_CHARS // 2 :]
    LOGGER.warning(
        "Transcript is %d chars; sending head and tail only (%d chars)",
        len(text), MAX_TRANSCRIPT_CHARS,
    )
    return f"{head}\n\n[... middle of transcript omitted for length ...]\n\n{tail}"


def _build_prompt(transcript_text: str, language: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT.format(language=language)},
        {
            "role": "user",
            "content": (
                "Analyse the transcript below and return exactly this JSON object:\n\n"
                f"{describe_schema()}\n\n"
                "Rules:\n"
                f"- Write every string value in {language}. The transcript is "
                f"partly in another language; translate as you summarise.\n"
                "- Use the transcript only. Do not add facts from outside it.\n"
                "- If the recording contains no action items or decisions, return an "
                "empty array for 'action_items' rather than inventing one.\n"
                "- Fill in 'owner' and 'due' for an action item whenever the "
                "transcript names a person or a date, and use null when it does not.\n"
                "- 'languages' lists the languages actually spoken in the transcript.\n"
                "- Return JSON only.\n\n"
                f"TRANSCRIPT:\n{transcript_text}"
            ),
        },
    ]


def _repair_prompt(problems: list[str]) -> str:
    listed = "\n".join(f"- {p}" for p in problems)
    return (
        "Your previous reply was rejected by the validator for these reasons:\n"
        f"{listed}\n\n"
        "Return the corrected JSON object. Fix only what is listed, keep the rest "
        "of the content, and reply with JSON only."
    )


def analyze_transcript(
    transcript: list[dict] | str,
    llm: LocalLLM | None = None,
    config: Config | None = None,
) -> dict:
    """Turn a transcript into a validated structured analysis.

    Args:
        transcript: transcript segments, or the transcript as one string.
        llm: an already-loaded model to reuse. When omitted a model is
            loaded and released around this call, which is convenient but
            wasteful if you are going to call the LLM again.
        config: run configuration; ``analysis_max_retries`` controls how
            many repair attempts are allowed.

    Returns:
        The validated, normalised analysis, plus a ``_validation`` key
        recording how many attempts it took and which model produced it.

    Raises:
        AnalysisValidationError: still invalid after every retry.
    """
    config = config or Config()
    owns_llm = llm is None
    llm = llm or LocalLLM(config)

    transcript_text = _transcript_for_prompt(transcript)
    messages = _build_prompt(transcript_text, config.analysis_language)
    max_attempts = config.analysis_max_retries + 1
    problems: list[str] = []
    raw = ""

    try:
        for attempt in range(1, max_attempts + 1):
            # First attempt is greedy so a good run is reproducible; later
            # attempts sample, so a bad answer is not simply repeated.
            temperature = 0.0 if attempt == 1 else 0.3
            raw = llm.chat(messages, temperature=temperature)

            try:
                candidate = extract_json(raw)
                problems = check_analysis(candidate)
            except ValueError as exc:
                problems = [str(exc)]
                candidate = None

            if not problems:
                LOGGER.info("Analysis validated on attempt %d/%d", attempt, max_attempts)
                analysis = normalise_analysis(candidate)  # type: ignore[arg-type]
                analysis["_validation"] = {
                    "attempts": attempt,
                    "max_attempts": max_attempts,
                    "model": config.llm_model,
                    "checked": sorted(REQUIRED_ANALYSIS_KEYS),
                }
                return analysis

            LOGGER.warning(
                "Analysis attempt %d/%d rejected: %s",
                attempt, max_attempts, "; ".join(problems),
            )
            if attempt < max_attempts:
                messages = messages + [
                    {"role": "assistant", "content": truncate(raw, 2000)},
                    {"role": "user", "content": _repair_prompt(problems)},
                ]
    finally:
        if owns_llm:
            llm.release()

    raise AnalysisValidationError(
        f"the model did not produce a valid analysis in {max_attempts} attempts; "
        f"last problems: {'; '.join(problems)}",
        attempts=max_attempts,
        problems=problems,
    )
