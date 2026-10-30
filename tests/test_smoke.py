"""Part 6 -- the test section.

Covers the four conditions the brief requires:

1. normal input;
2. invalid or missing file;
3. a RAG question with weak or no evidence;
4. a tool/agent error or invalid argument.

Runs two ways:

    python -m tests.test_smoke              # fast: no model downloads, no GPU
    python -m tests.test_smoke --with-models  # also exercises the LLM stages
    pytest tests/test_smoke.py              # pytest collects the same functions

The default mode deliberately avoids every heavy model so the suite is
usable as a pre-commit check. Tests that need one are skipped unless
asked for, and say so rather than passing silently.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow `python tests/test_smoke.py` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from media_assistant.chunking import chunk_segments, chunking_summary  # noqa: E402
from media_assistant.config import Config  # noqa: E402
from media_assistant.demo import DEFAULT_RAG_QUESTIONS, FAILING_AGENT_TASK  # noqa: E402
from media_assistant.exceptions import (  # noqa: E402
    AnalysisValidationError,
    MediaError,
    ToolError,
)
from media_assistant.language import (  # noqa: E402
    REQUIRED_ANALYSIS_KEYS,
    check_analysis,
    validate_analysis,
)
from media_assistant.llm import extract_json  # noqa: E402
from media_assistant.media import probe_metadata  # noqa: E402
from media_assistant.multilingual import (  # noqa: E402
    language_distribution,
    language_label_disagreements,
)
from media_assistant.reliability import CheckSuite, check_invalid_media  # noqa: E402
from media_assistant.retrieval import _extract_citations  # noqa: E402
from media_assistant.tools import Toolbox, normalise_for_counting  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "sample_transcript.json"

VALID_ANALYSIS = {
    "title": "Project meeting",
    "summary": "The team reviewed progress and agreed on next steps.",
    "key_points": ["transcription works", "retrieval works", "the agent works"],
    "topics": ["transcription", "retrieval"],
    "action_items": [{"description": "write the README", "owner": "Lin", "due": "Friday"}],
    "entities": [{"name": "Priya Nair", "type": "person"}, "Colab"],
    "languages": ["Chinese", "English"],
    "follow_up_questions": ["What is the fallback model?"],
}


def load_fixture() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# condition 1 -- normal input (deterministic half)
# --------------------------------------------------------------------------

def test_transcript_invariants() -> None:
    """The lightweight deterministic checks the brief asks for."""
    segments = load_fixture()["segments"]

    assert segments, "fixture transcript is empty"
    assert all(segment["start"] <= segment["end"] for segment in segments)
    assert all("text" in segment and segment["text"].strip() for segment in segments)
    assert all(isinstance(segment["start"], (int, float)) for segment in segments)
    assert len({segment["id"] for segment in segments}) == len(segments)
    assert all(a["start"] <= b["start"] for a, b in zip(segments, segments[1:]))


def test_chunking_preserves_evidence() -> None:
    """Chunks keep timestamps, stay single-language, and cover the recording."""
    segments = load_fixture()["segments"]
    chunks = chunk_segments(segments, Config())

    assert chunks, "chunking produced nothing"
    assert len({chunk["chunk_id"] for chunk in chunks}) == len(chunks)
    assert all(chunk["start"] <= chunk["end"] for chunk in chunks)
    assert all(chunk["text"].strip() for chunk in chunks)

    language_of = {segment["id"]: segment.get("language") for segment in segments}
    for chunk in chunks:
        languages = {language_of[sid] for sid in chunk["segment_ids"] if sid in language_of}
        assert len(languages) == 1, f"{chunk['chunk_id']} mixes languages: {languages}"

    summary = chunking_summary(chunks)
    assert summary["chunk_count"] == len(chunks)
    assert summary["max_duration_seconds"] <= Config().chunk_max_seconds + 0.01

    # The index must cover the whole recording, not just its start.
    assert max(chunk["end"] for chunk in chunks) >= max(s["end"] for s in segments) - 0.01


def test_asr_output_is_normalised_into_sane_segments() -> None:
    """Whatever Whisper emits, the invariants downstream relies on hold.

    Covers the shapes that actually occur: the span as a tuple or a list
    (this has varied between transformers versions), a ``None`` end on
    the final chunk, punctuation-only text, and a range running past the
    end of the audio.
    """
    from media_assistant.speech import _normalise_chunks

    cases = {
        "tuple spans": [
            {"timestamp": (0.0, 5.0), "text": "one", "language": "english"},
            {"timestamp": (5.0, None), "text": "two", "language": "english"},
        ],
        "list spans": [
            {"timestamp": [0.0, 5.0], "text": "one", "language": "english"},
            {"timestamp": [5.0, 9.0], "text": "two", "language": "english"},
        ],
        "missing span": [{"timestamp": None, "text": "one", "language": "english"}],
        "punctuation only": [
            {"timestamp": (0.0, 1.0), "text": "...", "language": "english"},
            {"timestamp": (1.0, 4.0), "text": "real", "language": "english"},
        ],
        "span past the end": [
            {"timestamp": (0.0, 999.0), "text": "overrun", "language": "english"}
        ],
    }
    for label, chunks in cases.items():
        segments = _normalise_chunks(chunks, audio_seconds=10.0)
        assert segments, f"{label}: everything was dropped"
        assert all(0 <= s["start"] <= s["end"] <= 10.0 for s in segments), label
        assert all(s["text"].strip() for s in segments), label
        assert [s["id"] for s in segments] == [
            f"S{i:03d}" for i in range(1, len(segments) + 1)
        ], label

    # Punctuation-only chunks are dropped, not kept as empty segments.
    assert len(_normalise_chunks(cases["punctuation only"], 10.0)) == 1
    # No confidence field is invented when the decoder reports none.
    assert "avg_logprob" not in _normalise_chunks(cases["list spans"], 10.0)[0]


def test_timestamps_format_for_humans_only() -> None:
    """Timestamps stay numeric internally and are formatted at the edge."""
    from media_assistant.utils import format_span, format_timestamp

    segments = load_fixture()["segments"]
    assert all(isinstance(segment["start"], (int, float)) for segment in segments)
    assert format_timestamp(95.24) == "01:35.2"
    assert format_timestamp(95.24, decimals=0) == "01:35"
    assert format_timestamp(3725.0).startswith("01:02:")
    assert format_span(95.2, 113.8) == "01:35.2–01:53.8"


# --------------------------------------------------------------------------
# condition 2 -- invalid / missing input
# --------------------------------------------------------------------------

def test_missing_file_raises_file_not_found() -> None:
    try:
        probe_metadata("./this_file_does_not_exist.wav")
    except FileNotFoundError as exc:
        assert "not found" in str(exc)
    else:
        raise AssertionError("a missing file should raise FileNotFoundError")


def test_non_media_file_raises_media_error(tmp_path: Path | None = None) -> None:
    """A text file with an audio extension must be rejected, not decoded."""
    directory = tmp_path or Path(".cache_media")
    directory.mkdir(parents=True, exist_ok=True)
    fake = directory / "definitely_not_audio.wav"
    fake.write_text("plain text pretending to be a WAV file\n", encoding="utf-8")
    try:
        probe_metadata(fake)
    except MediaError as exc:
        assert "decode" in str(exc) or "duration" in str(exc)
    else:
        raise AssertionError("a non-media file should raise MediaError")


def test_invalid_media_suite_all_pass() -> None:
    """The reliability suite's media checks agree with the tests above."""
    suite = CheckSuite()
    check_invalid_media(suite)
    failures = [case.name for case in suite.cases if not case.passed]
    assert not failures, f"media checks failed: {failures}"


# --------------------------------------------------------------------------
# condition 3 -- structured output and weak evidence
# --------------------------------------------------------------------------

def test_valid_analysis_accepted() -> None:
    validated = validate_analysis(VALID_ANALYSIS)
    assert REQUIRED_ANALYSIS_KEYS <= validated.keys()
    # Both permitted entity shapes normalise to objects.
    assert all(isinstance(entity, dict) and entity["name"] for entity in validated["entities"])
    assert all(isinstance(item, dict) for item in validated["action_items"])
    assert validated["action_items"][0]["owner"] == "Lin"


def test_invalid_analysis_rejected() -> None:
    """Missing keys, wrong types, and non-list list fields are all caught."""
    cases = {
        "missing keys": {"title": "only a title"},
        "list field is a string": {**VALID_ANALYSIS, "key_points": "a, b, c"},
        "too few key points": {**VALID_ANALYSIS, "key_points": ["one", "two"]},
        "too many key points": {**VALID_ANALYSIS, "key_points": list("abcdefg")},
        "wrong item type": {**VALID_ANALYSIS, "topics": [1, 2]},
        "empty string field": {**VALID_ANALYSIS, "title": "   "},
        "not an object": ["not a dict at all"],
    }
    for label, candidate in cases.items():
        problems = check_analysis(candidate)
        assert problems, f"{label} should have been rejected"
        try:
            validate_analysis(candidate)
        except AnalysisValidationError as exc:
            assert exc.problems
        else:
            raise AssertionError(f"{label} should have raised AnalysisValidationError")


def test_json_extraction_survives_model_habits() -> None:
    """Code fences, preambles, reasoning blocks and trailing commas."""
    assert extract_json('```json\n{"a": 1,}\n```') == {"a": 1}
    assert extract_json('Sure! Here it is:\n{"a": {"b": "}"}}') == {"a": {"b": "}"}}
    assert extract_json("<think>reasoning</think>{\"a\": 2}") == {"a": 2}
    for broken in ("", "   ", "no json anywhere", "{unclosed"):
        try:
            extract_json(broken)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{broken!r} should not have parsed")


def test_citations_must_match_retrieved_evidence() -> None:
    """A fabricated timestamp is not counted as grounded."""
    retrieved = [
        {"chunk_id": "C02", "start": 30.8, "end": 52.6},
        {"chunk_id": "C03", "start": 41.9, "end": 70.5},
    ]
    real = _extract_citations("Forty-seven segments [C02 | 00:30.8–00:52.6].", retrieved)
    assert real["citation_count"] == 1
    assert real["all_citations_grounded"] is True

    invented = _extract_citations("Something happened [C09 | 08:00.0–08:30.0].", retrieved)
    assert invented["citation_count"] == 1
    assert invented["grounded_citation_count"] == 0
    assert invented["all_citations_grounded"] is False

    none_at_all = _extract_citations("No timestamps here.", retrieved)
    assert none_at_all["citation_count"] == 0
    assert none_at_all["all_citations_grounded"] is False


def test_single_language_stamping_is_caught() -> None:
    """A bilingual transcript labelled with one language must be rejected.

    This is the regression test for the real defect: sequential Whisper
    decoding runs language detection once, on the first 30 seconds, and
    reuses that label for every segment. The transcript below is exactly
    what that produced -- correct English text carrying a ``chinese``
    label -- and every other invariant (timestamps, ids, ordering,
    non-empty text) holds, which is why nothing else caught it.
    """
    stamped = [
        {"id": "S001", "start": 0.0, "end": 5.0, "language": "chinese",
         "text": "大家好，今天是第三次碰头。"},
        {"id": "S002", "start": 5.0, "end": 10.0, "language": "chinese",
         "text": "Let's start with the transcription pipeline."},
        {"id": "S003", "start": 10.0, "end": 15.0, "language": "chinese",
         "text": "The submission deadline is August fifteenth."},
    ]

    # The bug is invisible to the distribution summary, which is the point.
    assert language_distribution(stamped)["is_multilingual"] is False

    flagged = language_label_disagreements(stamped)
    assert {d["segment_id"] for d in flagged} == {"S002", "S003"}
    assert all(d["script"] == "latin" for d in flagged)

    for segment, language in zip(stamped, ["chinese", "english", "english"]):
        segment["language"] = language
    assert language_label_disagreements(stamped) == []
    assert language_distribution(stamped)["is_multilingual"] is True

    # Mixed-script and romanised text must not be judged, or correct
    # transcripts would start failing.
    assert language_label_disagreements([
        {"id": "S004", "start": 0.0, "end": 1.0, "language": "chinese",
         "text": "我们用 Qwen embedding 测了 12 个查询"},
        {"id": "S005", "start": 1.0, "end": 2.0, "language": "chinese",
         "text": "1234"},
    ]) == []


def test_unanswerable_question_is_declared_in_the_demo_set() -> None:
    """The required five questions cover the four categories the brief lists."""
    categories = [question.category for question in DEFAULT_RAG_QUESTIONS]
    assert len(DEFAULT_RAG_QUESTIONS) >= 5
    assert categories.count("factual") >= 2
    assert "paraphrased" in categories
    assert "multi-chunk" in categories
    assert categories.count("unanswerable") == 1


# --------------------------------------------------------------------------
# condition 4 -- tool and agent errors
# --------------------------------------------------------------------------

class _StubIndex:
    """Stands in for the real index so tool tests need no embedding model."""

    def __init__(self, chunks: list[dict]):
        self.chunks = chunks

    def __len__(self) -> int:
        return len(self.chunks)

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        return [
            {
                "rank": rank,
                "chunk_id": chunk["chunk_id"],
                "start": chunk["start"],
                "end": chunk["end"],
                "timespan": f"{chunk['start']}-{chunk['end']}",
                "language": chunk["language"],
                "score": 0.5,
                "text": chunk["text"],
            }
            for rank, chunk in enumerate(self.chunks[:top_k], start=1)
        ]


def _stub_toolbox() -> Toolbox:
    fixture = load_fixture()
    chunks = chunk_segments(fixture["segments"], Config())
    return Toolbox(
        index=_StubIndex(chunks),
        segments=fixture["segments"],
        metadata=fixture["metadata"],
    )


def test_tool_errors_are_typed_and_informative() -> None:
    toolbox = _stub_toolbox()
    bad_calls = [
        ("translate_to_french", {"text": "hi"}),   # unknown tool
        ("count_mentions", {}),                     # missing required argument
        ("count_mentions", {"term": 123}),          # wrong type
        ("count_mentions", {"term": "  "}),         # empty after stripping
        ("search_transcript", {"query": "x", "top_k": "three"}),
        ("search_transcript", {"query": "x", "top_k": 99}),      # out of range
        ("search_transcript", {"query": "x", "limit": 3}),        # unknown parameter
        ("search_transcript", {"query": "x", "top_k": True}),     # bool is not a number
        ("get_transcript_range", {"start": 50, "end": 10}),       # end before start
        ("get_transcript_range", {"start": -1, "end": 10}),       # below minimum
    ]
    for name, arguments in bad_calls:
        try:
            toolbox.call(name, arguments)
        except ToolError as exc:
            assert str(exc), f"{name} raised an empty ToolError"
        else:
            raise AssertionError(f"{name}({arguments}) should have raised ToolError")

    # Still usable afterwards: a rejected call must not corrupt any state.
    assert toolbox.call("get_media_metadata")["segment_count"] == 33


def test_tools_return_expected_shapes() -> None:
    toolbox = _stub_toolbox()

    mentions = toolbox.call("count_mentions", {"term": "agent"})
    assert mentions["total_count"] >= 1
    assert all("timespan" in occurrence for occurrence in mentions["occurrences"])

    # Numeral normalisation: the transcript says 十九, the query says 19.
    assert toolbox.call("count_mentions", {"term": "19"})["total_count"] >= 1
    assert normalise_for_counting("十九个") == "19个"
    assert normalise_for_counting("ＪＳＯＮ") == "json"

    window = toolbox.call("get_transcript_range", {"start": 130, "end": 145})
    assert window["segment_count"] >= 1
    assert all(segment["text"] for segment in window["segments"])

    metadata = toolbox.call("get_media_metadata")
    assert metadata["languages_detected"] == ["chinese", "english"]
    assert metadata["duration_seconds"] > 0


def test_agent_step_limit_is_enforced_by_the_application() -> None:
    """A model that only ever calls tools still stops at max_steps."""
    from media_assistant.agent import run_agent

    class _AlwaysCallsATool:
        """Stand-in LLM that never produces a final answer."""

        def __init__(self):
            self.calls = 0

        def chat(self, messages, **kwargs):
            self.calls += 1
            return '{"thought": "again", "tool": "count_mentions", "args": {"term": "agent"}}'

    stub_llm = _AlwaysCallsATool()
    run = run_agent("count things forever", _stub_toolbox(), llm=stub_llm, max_steps=3)

    assert run.tool_calls == 3, f"expected exactly 3 tool calls, got {run.tool_calls}"
    assert run.stopped_because == "max_steps_reached"
    assert len(run.steps) == 3


def test_agent_recovers_from_a_bad_tool_call() -> None:
    """An unknown tool becomes an observation, not an exception."""
    from media_assistant.agent import run_agent

    class _WrongThenRight:
        def __init__(self):
            self.replies = [
                '{"tool": "translate_to_french", "args": {"text": "hi"}}',
                'not json at all',
                '{"tool": "count_mentions", "args": {"term": "agent"}}',
                '{"final_answer": "The word agent is said three times."}',
            ]

        def chat(self, messages, **kwargs):
            return self.replies.pop(0)

    run = run_agent("count agent mentions", _stub_toolbox(), llm=_WrongThenRight(), max_steps=5)

    assert run.stopped_because == "final_answer"
    assert "three times" in run.final_answer
    assert any(step.error for step in run.steps), "the bad calls should be recorded as errors"
    assert "count_mentions" in run.distinct_tools_used


def test_agent_rejects_bad_arguments_of_its_own() -> None:
    from media_assistant.agent import run_agent

    try:
        run_agent("", _stub_toolbox(), llm=object(), max_steps=3)
    except ValueError:
        pass
    else:
        raise AssertionError("an empty request should raise ValueError")

    try:
        run_agent("do something", _stub_toolbox(), llm=object(), max_steps=0)
    except ValueError:
        pass
    else:
        raise AssertionError("max_steps=0 should raise ValueError")


# --------------------------------------------------------------------------
# model-backed tests (opt in)
# --------------------------------------------------------------------------

def test_with_models() -> None:
    """End-to-end checks that need the real models. Opt in with --with-models."""
    from media_assistant.assistant import MediaAssistant
    from media_assistant.reliability import run_reliability_checks

    fixture = load_fixture()
    with MediaAssistant(Config(), verbose=True) as assistant:
        assistant.use_transcript(fixture["segments"], fixture["metadata"])

        unanswerable = next(
            q.question for q in DEFAULT_RAG_QUESTIONS if q.category == "unanswerable"
        )
        normal = next(q.question for q in DEFAULT_RAG_QUESTIONS if q.category == "factual")

        results = run_reliability_checks(
            assistant,
            unanswerable_question=unanswerable,
            normal_question=normal,
            failing_agent_request=FAILING_AGENT_TASK,
            include_model_checks=True,
        )

    failures = [case["name"] for case in results["cases"] if not case["passed"]]
    assert not failures, f"model-backed checks failed: {failures}"


# --------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------

def _collect() -> list:
    return [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    with_models = "--with-models" in argv

    passed, failed, skipped = 0, 0, 0
    for test in _collect():
        name = test.__name__
        if name == "test_with_models" and not with_models:
            print(f"SKIP  {name}  (pass --with-models to run it)")
            skipped += 1
            continue
        try:
            test()
        except AssertionError as exc:
            print(f"FAIL  {name}\n        {exc}")
            failed += 1
        except Exception as exc:  # noqa: BLE001 - a crash is a test failure
            print(f"ERROR {name}\n        {type(exc).__name__}: {exc}")
            failed += 1
        else:
            print(f"ok    {name}")
            passed += 1

    print(f"\n{passed} passed, {failed} failed, {skipped} skipped")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
