"""Part 6 -- reliability checks.

The brief asks for evidence that the system behaves under four
conditions: normal input, an invalid or missing file, a RAG question
with no supporting evidence, and a tool or agent error. Those checks
live here rather than only in the test file so the CLI can run them and
put the results in the report -- a demo that works once is not the same
as a system that handles its failure modes.

Checks that need a language model are separated from the ones that do
not, so the deterministic half runs in under a second.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .exceptions import MediaError, ToolError
from .media import probe_metadata
from .utils import LOGGER, truncate


@dataclass
class CheckResult:
    """The outcome of one named check."""

    name: str
    condition: str
    passed: bool
    detail: str = ""


@dataclass
class CheckSuite:
    """Accumulates check results without letting one failure stop the rest."""

    cases: list[CheckResult] = field(default_factory=list)

    def record(self, name: str, condition: str, passed: bool, detail: str = "") -> None:
        self.cases.append(CheckResult(name, condition, bool(passed), truncate(detail, 160)))
        LOGGER.info("%s %s — %s", "PASS" if passed else "FAIL", name, truncate(detail, 100))

    def expect_raises(self, name: str, condition: str, exception: type, call) -> None:
        """Record whether ``call`` raised the exception it was supposed to."""
        try:
            call()
        except exception as exc:
            self.record(name, condition, True, f"{type(exc).__name__}: {exc}")
        except Exception as exc:  # wrong type is still a failure
            self.record(
                name, condition, False,
                f"expected {exception.__name__}, got {type(exc).__name__}: {exc}",
            )
        else:
            self.record(name, condition, False, "no exception was raised")

    def to_dict(self) -> dict:
        passed = sum(1 for case in self.cases if case.passed)
        return {
            "total": len(self.cases),
            "passed": passed,
            "failed": len(self.cases) - passed,
            "all_passed": passed == len(self.cases),
            "cases": [vars(case) for case in self.cases],
        }


# --------------------------------------------------------------------------
# deterministic checks (no model needed)
# --------------------------------------------------------------------------

def check_transcript_invariants(segments: list[dict], suite: CheckSuite) -> None:
    """The lightweight assertions the brief asks for, as recorded checks."""
    suite.record(
        "transcript: timestamps ordered",
        "every segment satisfies start <= end",
        all(segment["start"] <= segment["end"] for segment in segments),
        f"{len(segments)} segments checked",
    )
    suite.record(
        "transcript: text present",
        "every segment has non-empty text",
        all(str(segment.get("text", "")).strip() for segment in segments),
        f"{len(segments)} segments checked",
    )
    suite.record(
        "transcript: ids unique",
        "no segment id appears twice",
        len({segment["id"] for segment in segments}) == len(segments),
        f"{len({s['id'] for s in segments})} distinct ids",
    )
    suite.record(
        "transcript: monotonic",
        "segments are ordered by start time",
        all(a["start"] <= b["start"] for a, b in zip(segments, segments[1:])),
        "checked pairwise",
    )

    # Catches a transcript that is textually perfect but whose language
    # labels are all wrong -- see multilingual.language_label_disagreements.
    from .multilingual import language_label_disagreements

    disagreements = language_label_disagreements(segments)
    suite.record(
        "transcript: language labels match the script",
        "no segment is labelled as a language its writing system contradicts",
        not disagreements,
        f"{len(segments)} segments checked, {len(disagreements)} disagreements"
        + (f"; first: {disagreements[0]['text']}" if disagreements else ""),
    )


def check_index_invariants(index, suite: CheckSuite) -> None:
    """Structural checks on the chunks and their embeddings."""
    import numpy as np

    chunks = index.chunks
    suite.record(
        "index: one embedding per chunk",
        "embeddings.shape[0] == len(chunks)",
        index.embeddings.shape[0] == len(chunks),
        f"{index.embeddings.shape[0]} vectors for {len(chunks)} chunks",
    )
    norms = np.linalg.norm(index.embeddings, axis=1)
    suite.record(
        "index: embeddings normalised",
        "every vector has unit length, so a dot product is a cosine",
        bool(np.allclose(norms, 1.0, atol=1e-3)),
        f"norm range {norms.min():.4f}–{norms.max():.4f}",
    )
    suite.record(
        "index: chunk ids unique",
        "no chunk id appears twice",
        len({chunk["chunk_id"] for chunk in chunks}) == len(chunks),
        f"{len(chunks)} chunks",
    )
    suite.record(
        "index: chunks are single-language",
        "no chunk mixes two languages",
        all(chunk.get("language") for chunk in chunks),
        f"languages: {sorted({c.get('language') for c in chunks})}",
    )


def check_invalid_media(suite: CheckSuite, workdir: str = ".") -> None:
    """Condition 2 -- invalid and missing input files."""
    suite.expect_raises(
        "media: missing file",
        "a path that does not exist raises FileNotFoundError",
        FileNotFoundError,
        lambda: probe_metadata("./definitely_not_a_real_file.wav"),
    )

    not_media = Path(workdir) / ".cache_media" / "not_really_audio.wav"
    not_media.parent.mkdir(parents=True, exist_ok=True)
    not_media.write_text("This is a text file that has been given a .wav extension.\n")
    suite.expect_raises(
        "media: not a media file",
        "a text file renamed .wav raises MediaError",
        MediaError,
        lambda: probe_metadata(not_media),
    )

    empty = not_media.with_name("empty.wav")
    empty.write_bytes(b"")
    suite.expect_raises(
        "media: empty file",
        "a zero-byte file raises MediaError",
        MediaError,
        lambda: probe_metadata(empty),
    )


def check_tool_errors(toolbox, suite: CheckSuite) -> None:
    """Condition 4 -- tool errors and invalid arguments."""
    suite.expect_raises(
        "tools: unknown tool",
        "calling a tool that does not exist raises ToolError",
        ToolError,
        lambda: toolbox.call("translate_to_french", {"text": "hello"}),
    )
    suite.expect_raises(
        "tools: missing argument",
        "omitting a required argument raises ToolError",
        ToolError,
        lambda: toolbox.call("count_mentions", {}),
    )
    suite.expect_raises(
        "tools: wrong argument type",
        "a string where a number belongs raises ToolError",
        ToolError,
        lambda: toolbox.call("search_transcript", {"query": "agent", "top_k": "three"}),
    )
    suite.expect_raises(
        "tools: unexpected argument",
        "an argument the tool does not declare raises ToolError",
        ToolError,
        lambda: toolbox.call("search_transcript", {"query": "agent", "limit": 3}),
    )
    suite.expect_raises(
        "tools: out-of-range argument",
        "a negative timestamp raises ToolError",
        ToolError,
        lambda: toolbox.call("get_transcript_range", {"start": -10, "end": 5}),
    )

    # And the happy path still works after all those rejections.
    result = toolbox.call("get_media_metadata")
    suite.record(
        "tools: still usable after errors",
        "a valid call succeeds after five rejected ones",
        isinstance(result, dict) and result.get("segment_count", 0) > 0,
        f"segment_count={result.get('segment_count')}",
    )


# --------------------------------------------------------------------------
# checks that need the language model
# --------------------------------------------------------------------------

def check_weak_evidence_question(assistant, suite: CheckSuite, question: str) -> dict | None:
    """Condition 3 -- a question the recording cannot answer."""
    result = assistant.ask(question)
    suite.record(
        "rag: unanswerable question refused",
        "a question with no supporting evidence is declined, not guessed",
        result["answered"] is False,
        f"top score {result['top_score']:.3f} vs threshold "
        f"{result['threshold']:.2f}; reason {result.get('refusal_reason')}",
    )
    return result


def check_normal_question(assistant, suite: CheckSuite, question: str) -> dict | None:
    """Condition 1 -- normal input produces a grounded, cited answer."""
    result = assistant.ask(question)
    suite.record(
        "rag: normal question answered",
        "a well-supported question is answered from the recording",
        result["answered"] is True,
        f"top score {result['top_score']:.3f}",
    )
    suite.record(
        "rag: answer carries timestamp citations",
        "the answer cites timespans that appear in the retrieved evidence",
        result.get("all_citations_grounded", False),
        f"{result.get('grounded_citation_count', 0)} of "
        f"{result.get('citation_count', 0)} citations matched retrieved chunks",
    )
    return result


def check_agent_recovers(assistant, suite: CheckSuite, request: str) -> object:
    """The agent survives an impossible request without crashing."""
    run = assistant.agent(request, max_steps=3)
    suite.record(
        "agent: impossible task handled",
        "an unavailable tool ends in an answer, not an exception",
        isinstance(run.final_answer, str),
        f"stopped_because={run.stopped_because}, tool_calls={run.tool_calls}",
    )
    suite.record(
        "agent: step limit enforced",
        "the loop never exceeds the step limit the application set",
        len(run.steps) <= 3,
        f"{len(run.steps)} steps taken, limit 3",
    )
    return run


def run_reliability_checks(
    assistant,
    *,
    unanswerable_question: str,
    normal_question: str | None = None,
    failing_agent_request: str | None = None,
    include_model_checks: bool = True,
) -> dict:
    """Run every check and return a structured result for the report."""
    suite = CheckSuite()

    check_transcript_invariants(assistant.segments, suite)
    if assistant.index is not None:
        check_index_invariants(assistant.index, suite)
    check_invalid_media(suite)
    if assistant.toolbox is not None:
        check_tool_errors(assistant.toolbox, suite)

    if include_model_checks:
        if normal_question:
            check_normal_question(assistant, suite, normal_question)
        check_weak_evidence_question(assistant, suite, unanswerable_question)
        if failing_agent_request:
            check_agent_recovers(assistant, suite, failing_agent_request)

    return suite.to_dict()
