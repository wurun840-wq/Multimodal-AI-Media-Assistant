"""The demonstrations the project brief requires, defined in one place.

Keeping the five RAG questions and the agent task here means the CLI,
the notebook and the tests all exercise exactly the same cases, and the
report cannot quietly drift from what was actually run.

The questions are written against the recording described in
``sample_media/recording_script.md``. Swap them for your own if you
record something else -- the categories are what the brief asks for, not
the specific wording.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DemoQuestion:
    """One required RAG test case."""

    category: str
    question: str
    expectation: str

    @property
    def should_be_answerable(self) -> bool:
        return self.category != "unanswerable"


# Two straightforward factual questions, one paraphrase, one that needs
# more than one chunk, and one the recording cannot answer.
DEFAULT_RAG_QUESTIONS: tuple[DemoQuestion, ...] = (
    DemoQuestion(
        category="factual",
        question="How many timestamped segments did Whisper produce on the test recording?",
        expectation="Stated directly in the English part of the recording.",
    ),
    DemoQuestion(
        category="factual",
        question="整段录音一共切出了多少个 chunk？",
        expectation="Stated directly in the Chinese part of the recording.",
    ),
    DemoQuestion(
        category="paraphrased",
        question="agent 最多能跑几轮？",
        expectation=(
            "The recording says 'the loop is capped at five steps' in English. "
            "The question shares no wording with it and is in the other language."
        ),
    ),
    DemoQuestion(
        category="multi-chunk",
        question="agent 的多工具演示能在提交截止日期之前做完吗？",
        expectation=(
            "Needs the 9 August action-item deadline (Chinese) and the "
            "15 August submission deadline (English), which are in different "
            "chunks and different languages."
        ),
    ),
    DemoQuestion(
        category="unanswerable",
        question="这个项目的预算是多少？",
        expectation=(
            "The recording never mentions money, cost or budget. The assistant "
            "must decline rather than guess."
        ),
    ),
)

# Requires at least two different tools: one search to locate the passage,
# one count over the whole transcript.
DEFAULT_AGENT_TASK = (
    "Find the passage where the agent's step limit is discussed, then count how "
    "many times the word 'agent' is said in the whole recording. Report both "
    "results with timestamps."
)

# Deliberately impossible with the available tools, to demonstrate that the
# loop degrades gracefully instead of inventing an answer.
FAILING_AGENT_TASK = (
    "Use the translate_to_french tool to translate the whole recording, then "
    "tell me the speaker's phone number."
)
