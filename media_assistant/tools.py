"""Part 4a -- deterministic tools the agent may call.

Every tool is ordinary Python: given the same transcript and the same
arguments it returns the same result, with no model in the loop. The
model only ever *chooses* a tool; the application validates the
arguments and executes it.

Argument validation is declarative rather than defensive coding inside
each function, so an unknown parameter, a missing required one, and a
value of the wrong type all produce the same shape of error -- which the
agent loop can hand back to the model as an observation.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Callable

from .exceptions import ToolError
from .utils import format_span, truncate

# --------------------------------------------------------------------------
# text normalisation for counting
# --------------------------------------------------------------------------

_CN_DIGITS = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
              "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_CN_NUMBER_RE = re.compile(r"[零〇一二两三四五六七八九十]+")


def _cn_number_to_int(text: str) -> int | None:
    """Convert a Chinese numeral in the range 0-99 to an integer.

    Returns None for anything outside that range, which is deliberate:
    a partially-correct conversion of 一千二百 would be worse than
    leaving the original text alone.
    """
    if text in _CN_DIGITS:
        return _CN_DIGITS[text]
    if "十" not in text:
        return None
    tens_text, _, units_text = text.partition("十")
    tens = 1 if tens_text == "" else _CN_DIGITS.get(tens_text)
    units = 0 if units_text == "" else _CN_DIGITS.get(units_text)
    if tens is None or units is None or tens > 9:
        return None
    return tens * 10 + units


def normalise_for_counting(text: str) -> str:
    """Fold the spelling differences that would split a term's count.

    Whisper writes the same spoken number as digits in one segment and
    as Chinese characters in the next, and mixes full-width and
    half-width Latin. Both are normalised here so ``count_mentions``
    measures what was said rather than how it happened to be written.
    Numerals above 99 are left as-is; see :func:`_cn_number_to_int`.
    """
    folded = unicodedata.normalize("NFKC", text).lower()
    return _CN_NUMBER_RE.sub(
        lambda m: str(_cn_number_to_int(m.group()) if _cn_number_to_int(m.group()) is not None else m.group()),
        folded,
    )


# --------------------------------------------------------------------------
# declarative tool specifications
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ParamSpec:
    """One tool parameter and the rules its value must satisfy."""

    name: str
    kind: str  # "string" | "integer" | "number"
    description: str
    required: bool = True
    default: Any = None
    minimum: float | None = None
    maximum: float | None = None

    def coerce(self, value: Any) -> Any:
        """Check and convert one argument value, or raise ToolError."""
        if self.kind == "string":
            if not isinstance(value, str):
                raise ToolError(
                    f"'{self.name}' must be a string, got {type(value).__name__}"
                )
            if not value.strip():
                raise ToolError(f"'{self.name}' must not be empty")
            return value.strip()

        # Numbers arrive as JSON numbers, or as strings when the model
        # quotes them. Booleans are rejected: in Python they would pass
        # an isinstance check against int and silently become 0 or 1.
        if isinstance(value, bool):
            raise ToolError(f"'{self.name}' must be a number, got boolean")
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ToolError(
                f"'{self.name}' must be a number, got {value!r} ({type(value).__name__})"
            )
        if self.kind == "integer":
            if number != int(number):
                raise ToolError(f"'{self.name}' must be a whole number, got {value!r}")
            number = int(number)
        if self.minimum is not None and number < self.minimum:
            raise ToolError(f"'{self.name}' must be at least {self.minimum}, got {number}")
        if self.maximum is not None and number > self.maximum:
            raise ToolError(f"'{self.name}' must be at most {self.maximum}, got {number}")
        return number


@dataclass(frozen=True)
class ToolSpec:
    """A tool's name, purpose, and parameter rules."""

    name: str
    description: str
    parameters: tuple[ParamSpec, ...] = ()

    def validate(self, arguments: dict | None) -> dict:
        """Return validated keyword arguments, or raise ToolError."""
        arguments = arguments or {}
        if not isinstance(arguments, dict):
            raise ToolError(
                f"arguments for '{self.name}' must be a JSON object, "
                f"got {type(arguments).__name__}"
            )

        known = {p.name for p in self.parameters}
        unexpected = sorted(set(arguments) - known)
        if unexpected:
            raise ToolError(
                f"'{self.name}' does not accept {', '.join(unexpected)}; "
                f"valid parameters are {', '.join(sorted(known)) or '(none)'}"
            )

        validated: dict[str, Any] = {}
        for spec in self.parameters:
            if spec.name not in arguments or arguments[spec.name] is None:
                if spec.required:
                    raise ToolError(f"'{self.name}' requires '{spec.name}' ({spec.description})")
                if spec.default is not None:
                    validated[spec.name] = spec.default
                continue
            validated[spec.name] = spec.coerce(arguments[spec.name])
        return validated

    def describe(self) -> str:
        """One-line signature used in the agent's system prompt."""
        if not self.parameters:
            return f"{self.name}() — {self.description}"
        params = ", ".join(
            f"{p.name}: {p.kind}" + ("" if p.required else f" = {p.default!r}")
            for p in self.parameters
        )
        details = "; ".join(f"{p.name} is {p.description}" for p in self.parameters)
        return f"{self.name}({params}) — {self.description}. {details}"


TOOL_SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="search_transcript",
        description="Find the transcript passages most relevant to a query",
        parameters=(
            ParamSpec("query", "string", "what to look for, in any language"),
            ParamSpec(
                "top_k", "integer", "how many passages to return",
                required=False, default=3, minimum=1, maximum=10,
            ),
        ),
    ),
    ToolSpec(
        name="count_mentions",
        description="Count how many times a word or phrase is said in the whole recording",
        parameters=(
            ParamSpec("term", "string", "the word or phrase to count"),
        ),
    ),
    ToolSpec(
        name="get_transcript_range",
        description="Read the verbatim transcript between two times",
        parameters=(
            ParamSpec("start", "number", "start time in seconds", minimum=0),
            ParamSpec("end", "number", "end time in seconds", minimum=0),
        ),
    ),
    ToolSpec(
        name="get_media_metadata",
        description="Get the recording's duration, model, and language information",
    ),
)

TOOL_SPECS_BY_NAME = {spec.name: spec for spec in TOOL_SPECS}


# --------------------------------------------------------------------------
# the toolbox
# --------------------------------------------------------------------------

@dataclass
class Toolbox:
    """Binds the tool specifications to one recording's data.

    ``TOOL_FUNCTIONS`` maps a tool name to the bound implementation, which
    is what the agent loop dispatches through.
    """

    index: Any  # TranscriptIndex; typed loosely to avoid a circular import
    segments: list[dict]
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.TOOL_FUNCTIONS: dict[str, Callable[..., Any]] = {
            "search_transcript": self.search_transcript,
            "count_mentions": self.count_mentions,
            "get_transcript_range": self.get_transcript_range,
            "get_media_metadata": self.get_media_metadata,
        }

    # -- tools ---------------------------------------------------------

    def search_transcript(self, query: str, top_k: int = 3) -> list[dict]:
        """Semantic search over the transcript index."""
        results = self.index.search(query, top_k=top_k)
        return [
            {
                "chunk_id": r["chunk_id"],
                "timespan": r["timespan"],
                "score": r["score"],
                "text": truncate(r["text"], 300),
            }
            for r in results
        ]

    def count_mentions(self, term: str) -> dict:
        """Count occurrences of a term across every transcript segment.

        Counting is done on normalised text (see
        :func:`normalise_for_counting`) so that "19" and "十九" are the
        same mention.
        """
        needle = normalise_for_counting(term)
        total = 0
        occurrences = []
        for segment in self.segments:
            haystack = normalise_for_counting(segment["text"])
            found = haystack.count(needle)
            if found:
                total += found
                occurrences.append(
                    {
                        "segment_id": segment.get("id"),
                        "timespan": format_span(segment["start"], segment["end"]),
                        "count": found,
                        "text": truncate(segment["text"], 120),
                    }
                )
        return {
            "term": term,
            "total_count": total,
            "segments_containing_term": len(occurrences),
            "occurrences": occurrences[:10],
            "note": "Counted on case-folded text with Chinese numerals 0-99 mapped to digits.",
        }

    def get_transcript_range(self, start: float, end: float) -> dict:
        """Return the verbatim segments overlapping ``[start, end]``."""
        if end < start:
            raise ToolError(f"end ({end}) must not be earlier than start ({start})")
        selected = [
            {
                "segment_id": s.get("id"),
                "timespan": format_span(s["start"], s["end"]),
                "language": s.get("language"),
                "text": s["text"],
            }
            for s in self.segments
            if s["end"] >= start and s["start"] <= end
        ]
        return {
            "requested_range": format_span(start, end),
            "segment_count": len(selected),
            "segments": selected,
        }

    def get_media_metadata(self) -> dict:
        """Report what is known about the recording itself."""
        languages = sorted({(s.get("language") or "unknown") for s in self.segments})
        duration = max((s["end"] for s in self.segments), default=0.0)
        return {
            "duration_seconds": round(duration, 2),
            "duration_display": format_span(0, duration),
            "segment_count": len(self.segments),
            "chunk_count": len(self.index) if self.index is not None else 0,
            "languages_detected": languages,
            "asr_model": self.metadata.get("asr_model"),
            "source_file": (self.metadata.get("source_media") or {}).get("filename"),
        }

    # -- dispatch ------------------------------------------------------

    def call(self, name: str, arguments: dict | None = None) -> Any:
        """Validate and execute a tool by name.

        Raises:
            ToolError: unknown tool, or arguments that fail validation.
                The agent loop turns this into an observation rather than
                letting it end the run.
        """
        if name not in TOOL_SPECS_BY_NAME:
            raise ToolError(
                f"unknown tool '{name}'; available tools are "
                f"{', '.join(sorted(TOOL_SPECS_BY_NAME))}"
            )
        validated = TOOL_SPECS_BY_NAME[name].validate(arguments)
        try:
            return self.TOOL_FUNCTIONS[name](**validated)
        except ToolError:
            raise
        except Exception as exc:  # a tool bug must not kill the agent run
            raise ToolError(f"tool '{name}' failed: {type(exc).__name__}: {exc}") from exc

    @staticmethod
    def describe_tools() -> str:
        """The tool catalogue, as shown to the model."""
        return "\n".join(f"- {spec.describe()}" for spec in TOOL_SPECS)
