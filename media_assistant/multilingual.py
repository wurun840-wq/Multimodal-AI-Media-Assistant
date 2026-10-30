"""Part 5 -- multilingual media mode (Option C).

The recording alternates between Chinese and English, so this extension
does four distinct things rather than calling a translate function once:

1. **Detect** the language of every segment and report the distribution
   over the whole recording.
2. **Preserve the original transcript** and hold the English translation
   in separate fields, so retrieval and display never share one string.
3. **Translate twice, by different routes** -- Whisper decoding the audio
   directly into English (``task="translate"``), and the LLM translating
   the transcript text -- and compare the two. They disagree in
   interesting places: Whisper hears the audio but has no document
   context, the LLM has context but only sees text that may already
   contain recognition errors.
4. **Compare retrieval across languages**: ask the same question in
   Chinese and in English and check whether the index returns the same
   evidence. The recording deliberately states some facts in only one
   language, which is what makes this a real test.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .config import Config
from .llm import LocalLLM
from .utils import LOGGER, format_span, truncate

# macOS ships several CJK-capable fonts; matplotlib's default does not
# cover Chinese and would render every character as a box.
_CJK_FONT_CANDIDATES = (
    "PingFang SC", "Hiragino Sans GB", "Heiti SC", "STHeiti",
    "Songti SC", "Arial Unicode MS",
)

TRANSLATION_SYSTEM_PROMPT = (
    "You are a translator. You translate meeting transcript passages into "
    "English, preserving names, numbers, dates and technical terms exactly. "
    "You return only the translation, with no preamble and no quotation marks. "
    "If a passage is already in English, you return it unchanged."
)


# --------------------------------------------------------------------------
# 1. language detection
# --------------------------------------------------------------------------

_HAN = re.compile(r"[一-鿿]")
_LATIN = re.compile(r"[A-Za-z]")


def language_label_disagreements(segments: list[dict]) -> list[dict]:
    """Segments whose writing system contradicts their language label.

    This exists because a whole class of failure is invisible to every
    other check: the transcript can be word-perfect while the ``language``
    field on every segment is wrong. That is what sequential Whisper
    decoding does by default -- it detects one language for the file and
    stamps it on all segments -- and because chunking, the language
    timeline and the cross-lingual comparison all read that field, Part 5
    silently degrades to "1 language, 0 switches" while every timestamp
    and every piece of text still looks correct.

    The test is deliberately narrow. It only judges the Chinese/English
    pair this recording uses, and only when the script is unambiguous:
    text containing Han characters labelled ``english``, or text with
    Latin letters and no Han labelled ``chinese``. Mixed-script text and
    any other language pair are not judged, because writing system alone
    does not determine language in general (romanised Chinese, or English
    quoting a Chinese term, would both be false positives).

    Returns:
        One entry per disagreement, with the segment id, its label, the
        script actually observed and the offending text.
    """
    disagreements = []
    for segment in segments:
        text = segment.get("text") or ""
        label = (segment.get("language") or "").lower()
        has_han, has_latin = bool(_HAN.search(text)), bool(_LATIN.search(text))

        if has_han and not has_latin:
            script = "han"
        elif has_latin and not has_han:
            script = "latin"
        else:
            continue  # mixed or scriptless: no opinion

        if (script == "han" and label == "english") or (
            script == "latin" and label == "chinese"
        ):
            disagreements.append(
                {
                    "segment_id": segment.get("id"),
                    "label": label,
                    "script": script,
                    "text": truncate(text, 80),
                }
            )
    return disagreements


def language_distribution(segments: list[dict]) -> dict:
    """Summarise which languages are spoken, and for how long.

    This aggregates the per-segment labels rather than assuming one
    language per file. Note that those labels come from
    ``SpeechRecognizer.detect_segment_languages``, not from the transcription
    pass: sequential decoding detects a language once and stamps it on every
    segment, which would make any recording look monolingual here.
    """
    totals: dict[str, dict] = {}
    for segment in segments:
        language = (segment.get("language") or "unknown").lower()
        entry = totals.setdefault(language, {"segments": 0, "seconds": 0.0})
        entry["segments"] += 1
        entry["seconds"] += float(segment["end"]) - float(segment["start"])

    spoken = sum(e["seconds"] for e in totals.values()) or 1.0
    for entry in totals.values():
        entry["seconds"] = round(entry["seconds"], 2)
        entry["share"] = round(entry["seconds"] / spoken, 4)

    return {
        "languages": sorted(totals),
        "language_count": len(totals),
        "is_multilingual": len(totals) > 1,
        "per_language": dict(sorted(totals.items(), key=lambda kv: -kv[1]["seconds"])),
        "switch_count": sum(
            1
            for a, b in zip(segments, segments[1:])
            if (a.get("language") or "") != (b.get("language") or "")
        ),
    }


def language_timeline(segments: list[dict]) -> list[dict]:
    """Collapse consecutive same-language segments into spans.

    This is the "who was speaking which language, when" view; it says
    nothing about *who* was speaking, only about the language detected.
    """
    timeline: list[dict] = []
    for segment in segments:
        language = (segment.get("language") or "unknown").lower()
        if timeline and timeline[-1]["language"] == language:
            timeline[-1]["end"] = float(segment["end"])
        else:
            timeline.append(
                {
                    "language": language,
                    "start": float(segment["start"]),
                    "end": float(segment["end"]),
                }
            )
    for span in timeline:
        span["duration"] = round(span["end"] - span["start"], 2)
        span["timespan"] = format_span(span["start"], span["end"])
    return timeline


# --------------------------------------------------------------------------
# 2 & 3. translation by two independent routes
# --------------------------------------------------------------------------

def translate_texts_with_llm(
    texts: list[str],
    llm: LocalLLM,
    config: Config | None = None,
) -> list[str]:
    """Translate passages into English one at a time.

    One call per passage rather than one call for the whole transcript:
    batching invites the model to drop or merge passages, and a
    per-passage failure then costs the entire translation.
    """
    config = config or Config()
    translations = []
    for index, text in enumerate(texts, start=1):
        if not text.strip():
            translations.append("")
            continue
        reply = llm.chat(
            [
                {"role": "system", "content": TRANSLATION_SYSTEM_PROMPT},
                {"role": "user", "content": f"Translate into English:\n\n{text}"},
            ],
            max_new_tokens=320,
        )
        translations.append(reply.strip())
        LOGGER.debug("Translated passage %d/%d", index, len(texts))
    return translations


def build_bilingual_transcript(
    segments: list[dict],
    llm: LocalLLM,
    whisper_translations: list[str] | None = None,
    config: Config | None = None,
) -> list[dict]:
    """Attach English translations to each segment without overwriting it.

    The original text stays in ``text``; translations live in
    ``translation_llm`` and ``translation_whisper``. Keeping them in
    separate fields is what stops a later stage from accidentally
    embedding a translation and retrieving against it.

    ``whisper_translations`` must already be aligned one-to-one with
    ``segments`` — see ``SpeechRecognizer.translate_segments``, which
    guarantees that by translating each segment separately.
    """
    config = config or Config()
    # Keyed by position rather than by object identity: positions stay
    # valid however the caller built the list.
    needs_translation = [
        index
        for index, segment in enumerate(segments)
        if (segment.get("language") or "").lower() != "english"
    ]
    LOGGER.info("Translating %d non-English segments with the LLM", len(needs_translation))

    translated = translate_texts_with_llm(
        [segments[index]["text"] for index in needs_translation], llm, config
    )
    by_position = dict(zip(needs_translation, translated))

    whisper_aligned = list(whisper_translations or [""] * len(segments))
    if len(whisper_aligned) != len(segments):
        # Would silently shift every row of the comparison table by one.
        raise ValueError(
            f"whisper_translations has {len(whisper_aligned)} entries for "
            f"{len(segments)} segments; they must align one-to-one"
        )

    bilingual = []
    for index, (segment, whisper_text) in enumerate(zip(segments, whisper_aligned)):
        is_english = index not in by_position
        bilingual.append(
            {
                **segment,
                "translation_llm": segment["text"] if is_english else by_position[index],
                "translation_whisper": whisper_text,
                "translation_needed": not is_english,
            }
        )
    return bilingual


def compare_translation_routes(bilingual_segments: list[dict], limit: int = 8) -> list[dict]:
    """Rows contrasting the two translation routes, for the report."""
    rows = []
    for segment in bilingual_segments:
        if not segment.get("translation_needed"):
            continue
        rows.append(
            {
                "segment_id": segment.get("id"),
                "timespan": format_span(segment["start"], segment["end"]),
                "original": segment["text"],
                "whisper_translate": segment.get("translation_whisper", ""),
                "llm_translate": segment.get("translation_llm", ""),
                "routes_agree": _rough_similarity(
                    segment.get("translation_whisper", ""), segment.get("translation_llm", "")
                ),
            }
        )
        if len(rows) >= limit:
            break
    return rows


def _rough_similarity(left: str, right: str) -> float:
    """Word-overlap ratio between two translations of the same passage.

    A blunt measure on purpose: it is here to flag rows worth reading in
    the report, not to score translation quality.
    """
    left_words = set(left.lower().split())
    right_words = set(right.lower().split())
    if not left_words or not right_words:
        return 0.0
    return round(len(left_words & right_words) / len(left_words | right_words), 3)


# --------------------------------------------------------------------------
# 4. cross-lingual retrieval comparison
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class CrossLingualQuery:
    """The same question in two languages, with the chunk it should find."""

    label: str
    chinese: str
    english: str
    evidence_language: str
    note: str = ""


# The recording states each of these facts in one language only, so
# querying in the other language is a genuine cross-lingual retrieval
# test rather than a keyword match.
DEFAULT_CROSS_LINGUAL_QUERIES: tuple[CrossLingualQuery, ...] = (
    CrossLingualQuery(
        label="cross-lingual retrieval result",
        chinese="跨语言查询的命中率是多少？",
        english="How well did the embedding model do on cross-lingual retrieval?",
        evidence_language="chinese",
        note="Stated only in the Chinese part of the recording.",
    ),
    CrossLingualQuery(
        label="GPU memory constraint",
        chinese="为什么要在加载语言模型之前删掉 Whisper？",
        english="Why must Whisper be deleted before loading the language model?",
        evidence_language="english",
        note="Stated only in the English part of the recording.",
    ),
    CrossLingualQuery(
        label="submission deadline",
        chinese="项目的提交截止日期是什么时候？",
        english="When is the submission deadline?",
        evidence_language="english",
        note="Stated only in the English part of the recording.",
    ),
)


def cross_lingual_comparison(
    index,
    queries: tuple[CrossLingualQuery, ...] = DEFAULT_CROSS_LINGUAL_QUERIES,
    top_k: int = 3,
) -> list[dict]:
    """Run each question in both languages and compare what comes back.

    Reports whether the two languages agree on the top hit and how much
    their top-k sets overlap -- the concrete evidence that the index is
    cross-lingual rather than lexical.
    """
    rows = []
    for query in queries:
        chinese_hits = index.search(query.chinese, top_k=top_k)
        english_hits = index.search(query.english, top_k=top_k)

        chinese_ids = [h["chunk_id"] for h in chinese_hits]
        english_ids = [h["chunk_id"] for h in english_hits]
        overlap = sorted(set(chinese_ids) & set(english_ids))

        rows.append(
            {
                "label": query.label,
                "note": query.note,
                "evidence_language": query.evidence_language,
                "chinese_query": query.chinese,
                "english_query": query.english,
                "chinese_top_chunks": chinese_ids,
                "english_top_chunks": english_ids,
                "chinese_top_score": chinese_hits[0]["score"] if chinese_hits else 0.0,
                "english_top_score": english_hits[0]["score"] if english_hits else 0.0,
                "same_top_hit": bool(chinese_ids and english_ids and chinese_ids[0] == english_ids[0]),
                "overlap_at_k": len(overlap),
                "overlapping_chunks": overlap,
                "evidence_timespan": chinese_hits[0]["timespan"] if chinese_hits else "",
            }
        )
    return rows


# --------------------------------------------------------------------------
# visual output
# --------------------------------------------------------------------------

def _apply_cjk_font() -> str | None:
    """Point matplotlib at a font that can draw Chinese, if one exists."""
    import matplotlib
    from matplotlib import font_manager

    available = {f.name for f in font_manager.fontManager.ttflist}
    for candidate in _CJK_FONT_CANDIDATES:
        if candidate in available:
            matplotlib.rcParams["font.family"] = [candidate]
            matplotlib.rcParams["axes.unicode_minus"] = False
            return candidate
    LOGGER.warning("No CJK font found; Chinese text in the figure may not render")
    return None


def render_multilingual_figure(
    timeline: list[dict],
    cross_rows: list[dict],
    output_path: str = "visual_output.png",
    title: str = "Multilingual media analysis",
) -> str:
    """Draw the visual output for Part 5.

    Two panels: when each language is spoken across the recording, and
    the cross-lingual retrieval comparison as a table.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    font = _apply_cjk_font()
    colours = {"chinese": "#3b6ea5", "english": "#c0663a", "unknown": "#8a8a8a"}

    row_count = max(len(cross_rows), 1)
    figure, (timeline_axes, table_axes) = plt.subplots(
        2, 1,
        figsize=(12, 3.2 + 0.55 * row_count),
        gridspec_kw={"height_ratios": [1.1, 0.5 + 0.35 * row_count]},
    )

    # -- panel 1: language timeline ------------------------------------
    for span in timeline:
        timeline_axes.barh(
            0,
            span["end"] - span["start"],
            left=span["start"],
            height=0.5,
            color=colours.get(span["language"], colours["unknown"]),
            edgecolor="white",
        )
    total = max((s["end"] for s in timeline), default=0.0)
    timeline_axes.set_xlim(0, total)
    timeline_axes.set_ylim(-0.5, 0.5)
    timeline_axes.set_yticks([])
    timeline_axes.set_xlabel("time (seconds)")
    timeline_axes.set_title(
        f"{title} — detected language over time", fontsize=12, pad=12
    )
    seen = {span["language"] for span in timeline}
    timeline_axes.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color=colours.get(language, colours["unknown"]), label=language)
            for language in sorted(seen)
        ],
        loc="upper right",
        ncol=len(seen),
        frameon=False,
    )

    # -- panel 2: cross-lingual retrieval table ------------------------
    table_axes.axis("off")
    header = ["query (label)", "asked in ZH → chunks", "asked in EN → chunks", "same top hit", "overlap@3"]
    body = [
        [
            truncate(row["label"], 28),
            f"{', '.join(row['chinese_top_chunks'])}  ({row['chinese_top_score']:.2f})",
            f"{', '.join(row['english_top_chunks'])}  ({row['english_top_score']:.2f})",
            "yes" if row["same_top_hit"] else "no",
            str(row["overlap_at_k"]),
        ]
        for row in cross_rows
    ]
    if body:
        # An explicit bbox makes the table fill its axes, instead of sitting
        # in the top of a mostly-empty panel.
        table = table_axes.table(
            cellText=body, colLabels=header, cellLoc="left", bbox=[0, 0, 1, 0.92]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        for column in range(len(header)):
            # Shading rather than a bold weight: the macOS CJK fonts have no
            # bold face, and asking for one makes matplotlib warn per cell.
            table[0, column].set_facecolor("#e8eef5")
    table_axes.set_title(
        "Cross-lingual retrieval: the same question asked in both languages",
        fontsize=11, pad=18,
    )

    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    LOGGER.info("Wrote %s (font: %s)", output_path, font or "matplotlib default")
    return output_path
