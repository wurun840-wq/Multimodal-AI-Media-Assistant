"""Part 3b -- semantic search and grounded question answering.

Retrieval and generation are kept separable on purpose: ``search``
returns scored, timestamped evidence with no model involved beyond the
embedder, and ``answer`` reports both the evidence it received and the
answer it produced. The project brief asks for exactly that separation
so the two can be marked independently.

Unanswerable questions are handled twice over. Before generation, a
similarity floor rejects questions the recording has no evidence for.
During generation, the model is required to emit a sentinel rather than
improvise. Either path yields ``answered: False`` with a reason.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from .chunking import chunk_segments, chunking_summary
from .config import Config
from .embeddings import Embedder
from .llm import LocalLLM
from .utils import LOGGER, format_span

INSUFFICIENT = "INSUFFICIENT_EVIDENCE"

# Matches the citation form the model is told to use. The chunk id
# prefix is optional because a bare timespan is still a valid citation,
# and the dash class is wide because models substitute freely between
# the hyphen, en dash, em dash and arrow.
_CITATION_RE = re.compile(
    r"\[\s*(?:(C\d+)\s*[|｜:]\s*)?"
    r"((?:\d{1,2}:)?\d{1,2}:\d{2}(?:\.\d+)?)\s*(?:[–—\-−→~]+|\bto\b)\s*"
    r"((?:\d{1,2}:)?\d{1,2}:\d{2}(?:\.\d+)?)\s*\]"
)

SYSTEM_PROMPT = (
    "You answer questions about a recording. You may use only the transcript "
    "evidence given to you in the prompt; you have no other knowledge of the "
    "recording and must not fill gaps from general knowledge."
)


def _timestamp_to_seconds(text: str) -> float:
    parts = [float(p) for p in text.split(":")]
    seconds = 0.0
    for part in parts:
        seconds = seconds * 60 + part
    return seconds


@dataclass
class TranscriptIndex:
    """Chunks plus their embeddings, with the query side attached.

    Holding the embedder here is what lets ``search(query, top_k)`` take
    a plain string, which is the signature the brief specifies.
    """

    chunks: list[dict]
    embeddings: np.ndarray
    embedder: Embedder
    config: Config = field(default_factory=Config)

    def __len__(self) -> int:
        return len(self.chunks)

    @property
    def summary(self) -> dict:
        stats = chunking_summary(self.chunks)
        stats["embedding_model"] = self.config.embedding_model
        stats["embedding_dimension"] = int(self.embeddings.shape[1]) if len(self) else 0
        return stats

    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        """Return the ``top_k`` most similar chunks, with scores.

        Similarity is cosine, computed as a dot product because both
        sides are L2-normalised at encode time. With a few dozen chunks
        a dense matrix multiply is faster than any index structure would
        be, so there is nothing to build.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if not self.chunks:
            return []

        top_k = self.config.top_k if top_k is None else int(top_k)
        if top_k < 1:
            raise ValueError(f"top_k must be at least 1, got {top_k}")
        top_k = min(top_k, len(self.chunks))

        query_vector = self.embedder.encode_queries([query])[0]
        scores = self.embeddings @ query_vector
        order = np.argsort(-scores)[:top_k]

        results = []
        for rank, position in enumerate(order, start=1):
            chunk = self.chunks[int(position)]
            results.append(
                {
                    "rank": rank,
                    "chunk_id": chunk["chunk_id"],
                    "start": chunk["start"],
                    "end": chunk["end"],
                    "timespan": format_span(chunk["start"], chunk["end"]),
                    "language": chunk["language"],
                    "score": round(float(scores[int(position)]), 4),
                    "text": chunk["text"],
                }
            )
        return results


def build_transcript_index(
    segments: list[dict],
    embedder: Embedder | None = None,
    config: Config | None = None,
) -> TranscriptIndex:
    """Chunk the transcript and embed every chunk."""
    config = config or Config()
    embedder = embedder or Embedder(config)

    chunks = chunk_segments(segments, config)
    if not chunks:
        raise ValueError("cannot build an index from an empty transcript")

    embeddings = embedder.encode_documents([c["text"] for c in chunks])
    LOGGER.info(
        "Indexed %d chunks (%d-dim, %s)",
        len(chunks), embeddings.shape[1], config.embedding_model,
    )
    return TranscriptIndex(chunks=chunks, embeddings=embeddings, embedder=embedder, config=config)


def format_evidence(results: list[dict]) -> str:
    """Render retrieved chunks as the evidence block shown to the model."""
    return "\n\n".join(
        f"[{r['chunk_id']} | {r['timespan']}] {r['text']}" for r in results
    )


def _build_prompt(question: str, results: list[dict]) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "TRANSCRIPT EVIDENCE:\n"
                f"{format_evidence(results)}\n\n"
                f"QUESTION: {question}\n\n"
                "Rules:\n"
                "- Answer using the evidence above and nothing else.\n"
                "- End each statement with a citation copied verbatim from the "
                "header of the passage it came from, including the square "
                "brackets — for example: The prototype is due Friday "
                "[C03 | 01:35.2–01:53.8].\n"
                "- Put the citation after the statement, never before it, and "
                "do not repeat the same citation twice in one sentence.\n"
                "- The evidence may be in a different language from the question. "
                "Answer in the language of the question.\n"
                f"- If the evidence does not answer the question, reply with "
                f"'{INSUFFICIENT}: ' followed by one short sentence saying what is "
                "missing. Do not guess, and do not use outside knowledge.\n"
                "- Be concise: two sentences at most."
            ),
        },
    ]


def _extract_citations(answer: str, results: list[dict], tolerance: float = 2.0) -> dict:
    """Pull timestamp citations out of an answer and check they are real.

    A citation counts as grounded when its span overlaps a chunk that was
    actually retrieved. This catches the failure mode where a model
    invents a plausible-looking timestamp.
    """
    citations = []
    grounded = 0
    for chunk_id, start_text, end_text in _CITATION_RE.findall(answer):
        start = _timestamp_to_seconds(start_text)
        end = _timestamp_to_seconds(end_text)
        matches = [
            r["chunk_id"]
            for r in results
            if start <= r["end"] + tolerance and end >= r["start"] - tolerance
        ]
        if matches:
            grounded += 1
        label = f"[{chunk_id} | {start_text}–{end_text}]" if chunk_id else f"[{start_text}–{end_text}]"
        citations.append(
            {
                "text": label,
                "chunk_id": chunk_id or None,
                "start": round(start, 2),
                "end": round(end, 2),
                "matches_retrieved_chunks": matches,
            }
        )
    return {
        "citations": citations,
        "citation_count": len(citations),
        "grounded_citation_count": grounded,
        "all_citations_grounded": bool(citations) and grounded == len(citations),
    }


def answer_question(
    question: str,
    index: TranscriptIndex,
    llm: LocalLLM | None = None,
    config: Config | None = None,
    top_k: int | None = None,
) -> dict:
    """Answer a question from retrieved transcript evidence.

    Returns a record containing the retrieved chunks and their scores,
    the answer, its timestamp citations, and whether the question was
    answered at all -- so retrieval and generation can be evaluated
    separately.
    """
    config = config or index.config
    results = index.search(question, top_k=config.top_k if top_k is None else top_k)
    top_score = results[0]["score"] if results else 0.0

    record: dict = {
        "question": question,
        "retrieved": results,
        "top_score": top_score,
        "threshold": config.min_similarity,
    }

    # Guard 1: nothing in the recording is close enough to the question.
    if top_score < config.min_similarity:
        record.update(
            answered=False,
            answer=(
                "I cannot answer that from this recording — the transcript does not "
                f"contain relevant evidence (best match scored {top_score:.2f}, below "
                f"the {config.min_similarity:.2f} threshold)."
            ),
            refusal_reason="below_similarity_threshold",
            citations=[],
            citation_count=0,
            grounded_citation_count=0,
            all_citations_grounded=False,
        )
        LOGGER.info("Question refused on similarity (%.3f): %s", top_score, question)
        return record

    owns_llm = llm is None
    llm = llm or LocalLLM(config)
    try:
        raw = llm.chat(_build_prompt(question, results), max_new_tokens=320)
    finally:
        if owns_llm:
            llm.release()

    # Guard 2: the model itself reports that the evidence is not enough.
    if raw.strip().upper().startswith(INSUFFICIENT):
        reason = raw.split(":", 1)[1].strip() if ":" in raw else ""
        record.update(
            answered=False,
            answer=f"I cannot answer that from this recording. {reason}".strip(),
            refusal_reason="model_declined_insufficient_evidence",
            citations=[],
            citation_count=0,
            grounded_citation_count=0,
            all_citations_grounded=False,
        )
        LOGGER.info("Model declined for lack of evidence: %s", question)
        return record

    record.update(answered=True, answer=raw.strip(), refusal_reason=None)
    record.update(_extract_citations(raw, results))
    if not record["citation_count"]:
        LOGGER.warning("Answer carried no timestamp citation: %s", question)
    return record


# --------------------------------------------------------------------------
# names from the project brief's starter code
# --------------------------------------------------------------------------
# The brief sketches `search_transcript(query, top_k=3)` and
# `ask_media(question)` as module-level functions over global state. The
# index is passed explicitly here instead, so two recordings can be open
# at once and nothing depends on import order.

def search_transcript(query: str, top_k: int = 3, *, index: TranscriptIndex) -> list[dict]:
    """Return the top-k timestamped transcript chunks for ``query``."""
    return index.search(query, top_k=top_k)


def ask_media(
    question: str,
    top_k: int = 3,
    *,
    index: TranscriptIndex,
    llm: LocalLLM | None = None,
    config: Config | None = None,
) -> dict:
    """Ground an answer in retrieved chunks and cite timestamps."""
    return answer_question(question, index, llm=llm, config=config, top_k=top_k)
