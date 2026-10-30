"""Part 3a -- turning transcript segments into retrievable chunks.

Whisper segments are too short to retrieve against on their own: one
sentence rarely carries enough context to answer a question, and a
sentence-level index makes every score look alike. Segments are
therefore grouped into chunks of roughly ``chunk_target_seconds``.

Two rules shape the grouping, and both are deliberate:

1. **Chunks never span a language change.** In a bilingual recording a
   switch from Chinese to English is also a topic boundary, and mixing
   both languages into one vector blurs it. Keeping chunks
   single-language is what makes the cross-lingual retrieval comparison
   in Part 5 meaningful.
2. **Consecutive chunks overlap by one segment**, so a fact stated
   across a chunk boundary is still fully present in at least one chunk.
   The overlap is only applied within a language run, for the same
   reason as rule 1.
"""

from __future__ import annotations

from .config import Config

# Languages written without spaces between words.
_SCRIPTIO_CONTINUA = {"chinese", "japanese", "korean", "yue", "zh", "ja", "ko"}


def _join_text(texts: list[str], language: str | None) -> str:
    """Join segment texts using the spacing convention of the language."""
    parts = [t.strip() for t in texts if t and t.strip()]
    if (language or "").lower() in _SCRIPTIO_CONTINUA:
        return "".join(parts)
    return " ".join(parts)


def _language_of(segment: dict) -> str:
    """Normalised language label, with a stable fallback."""
    return (segment.get("language") or "unknown").lower()


def chunk_segments(segments: list[dict], config: Config | None = None) -> list[dict]:
    """Group timestamped segments into timestamped retrieval chunks.

    Args:
        segments: output of the speech stage; each needs ``start``, ``end``
            and ``text``, and optionally ``language``.
        config: supplies ``chunk_target_seconds``, ``chunk_max_seconds``
            and ``chunk_overlap_segments``.

    Returns:
        Chunks of the form::

            {"chunk_id": "C07", "start": 95.2, "end": 113.8,
             "text": "...", "language": "english",
             "segment_ids": ["S011", "S012"], "duration": 18.6}

        Timestamps stay numeric; they are formatted only for display.
    """
    config = config or Config()
    if not segments:
        return []

    ordered = sorted(segments, key=lambda s: (float(s["start"]), float(s["end"])))
    groups: list[list[dict]] = []
    current: list[dict] = []

    for segment in ordered:
        if not current:
            current = [segment]
            continue

        same_language = _language_of(segment) == _language_of(current[0])
        span = float(segment["end"]) - float(current[0]["start"])
        target_reached = span > config.chunk_target_seconds
        too_long = span > config.chunk_max_seconds

        if not same_language or too_long or (target_reached and len(current) > 1):
            groups.append(current)
            current = [segment]
        else:
            current.append(segment)

    if current:
        groups.append(current)

    # Prepend the tail of the previous group, when it is the same language.
    overlap = max(0, config.chunk_overlap_segments)
    chunks: list[dict] = []
    for index, group in enumerate(groups):
        members = list(group)
        if overlap and index > 0:
            previous = groups[index - 1]
            if _language_of(previous[0]) == _language_of(group[0]):
                members = previous[-overlap:] + members

        language = _language_of(members[0])
        start = float(members[0]["start"])
        end = float(members[-1]["end"])
        chunks.append(
            {
                "chunk_id": f"C{index + 1:02d}",
                "start": round(start, 3),
                "end": round(end, 3),
                "duration": round(end - start, 3),
                "language": language,
                "text": _join_text([m["text"] for m in members], language),
                "segment_ids": [m.get("id", "") for m in members],
            }
        )
    return chunks


def chunking_summary(chunks: list[dict]) -> dict:
    """Descriptive statistics used by the report and the tests."""
    if not chunks:
        return {"chunk_count": 0}
    durations = [c["duration"] for c in chunks]
    languages: dict[str, int] = {}
    for chunk in chunks:
        languages[chunk["language"]] = languages.get(chunk["language"], 0) + 1
    return {
        "chunk_count": len(chunks),
        "mean_duration_seconds": round(sum(durations) / len(durations), 2),
        "min_duration_seconds": round(min(durations), 2),
        "max_duration_seconds": round(max(durations), 2),
        "chunks_per_language": languages,
        "covered_seconds": round(max(c["end"] for c in chunks) - min(c["start"] for c in chunks), 2),
    }
