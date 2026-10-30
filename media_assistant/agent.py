"""Part 4b -- a bounded, transparent tool-using agent.

The loop is deliberately small and fully observable:

    model proposes  ->  application validates  ->  application executes
                    <-  observation           <-

The model never runs anything. It emits a JSON object naming a tool; the
application checks the name and every argument, calls the Python
function itself, and feeds the result back. Unknown tools, malformed
JSON, and bad arguments all become observations the model can recover
from, so a wrong guess costs a step rather than the run.

The step limit is enforced by the loop, not requested of the model. When
it is reached the agent stops calling tools and produces a final answer
from whatever it has already observed.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

from .config import Config
from .exceptions import ToolError
from .llm import LocalLLM, extract_json
from .tools import Toolbox
from .utils import LOGGER, truncate

MAX_OBSERVATION_CHARS = 1_200

SYSTEM_PROMPT = """You are an assistant that answers questions about a recording by calling tools.

Available tools:
{tools}

Reply with a single JSON object and nothing else. To call a tool:

  {{"thought": "why this tool", "tool": "<name>", "args": {{...}}}}

To finish, when you have enough information:

  {{"thought": "why I can answer now", "final_answer": "<your answer>"}}

Rules:
- One tool call per reply.
- You have at most {max_steps} tool calls. Use them deliberately.
- Quote timestamps from the observations in your final answer.
- Base the final answer only on what the observations actually returned.
- If an observation reports an error, read it and correct your next call.
"""


@dataclass
class AgentStep:
    """One iteration of the loop, recorded for the report."""

    step: int
    thought: str | None = None
    tool: str | None = None
    args: dict | None = None
    observation: str | None = None
    error: str | None = None
    raw_reply: str | None = None

    def summary(self) -> str:
        if self.tool is None:
            label = "malformed reply" if self.error else "produced the final answer"
            return f"step {self.step}: {label}"
        status = f"error: {self.error}" if self.error else truncate(self.observation or "", 100)
        return f"step {self.step}: {self.tool}({json.dumps(self.args, ensure_ascii=False)}) -> {status}"


@dataclass
class AgentRun:
    """The full record of one agent run."""

    request: str
    final_answer: str
    steps: list[AgentStep] = field(default_factory=list)
    tool_calls: int = 0
    successful_tool_calls: int = 0
    stopped_because: str = "final_answer"
    max_steps: int = 0

    @property
    def distinct_tools_used(self) -> list[str]:
        return sorted({s.tool for s in self.steps if s.tool and not s.error})

    def to_dict(self) -> dict:
        return {
            "request": self.request,
            "final_answer": self.final_answer,
            "tool_calls": self.tool_calls,
            "successful_tool_calls": self.successful_tool_calls,
            "distinct_tools_used": self.distinct_tools_used,
            "stopped_because": self.stopped_because,
            "max_steps": self.max_steps,
            "steps": [asdict(s) for s in self.steps],
        }


def _observation_text(payload: object) -> str:
    """Serialise a tool result for the model, bounded in size."""
    text = json.dumps(payload, ensure_ascii=False, indent=None)
    if len(text) <= MAX_OBSERVATION_CHARS:
        return text
    return text[:MAX_OBSERVATION_CHARS] + " …(truncated)"


def run_agent(
    request: str,
    toolbox: Toolbox,
    llm: LocalLLM | None = None,
    config: Config | None = None,
    max_steps: int | None = None,
) -> AgentRun:
    """Run the model/tool/observation loop until an answer or the step limit.

    Args:
        request: what the user wants done.
        toolbox: bound tools for this recording.
        llm: a loaded model to reuse; one is created and released if omitted.
        config: supplies ``max_agent_steps`` when ``max_steps`` is not given.
        max_steps: hard cap on tool calls, enforced by this function.

    Returns:
        An :class:`AgentRun` holding the answer and every step taken, so
        the agent's behaviour can be inspected rather than trusted.
    """
    if not isinstance(request, str) or not request.strip():
        raise ValueError("request must be a non-empty string")

    config = config or Config()
    # `is None` rather than a truthiness test: max_steps=0 is a caller
    # error worth reporting, not a request for the default.
    max_steps = int(config.max_agent_steps if max_steps is None else max_steps)
    if max_steps < 1:
        raise ValueError(f"max_steps must be at least 1, got {max_steps}")

    owns_llm = llm is None
    llm = llm or LocalLLM(config)

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT.format(
                tools=Toolbox.describe_tools(), max_steps=max_steps
            ),
        },
        {"role": "user", "content": request},
    ]
    run = AgentRun(request=request, final_answer="", max_steps=max_steps)

    try:
        for step_number in range(1, max_steps + 1):
            raw = llm.chat(messages, max_new_tokens=400)
            step = AgentStep(step=step_number, raw_reply=truncate(raw, 400))

            try:
                decision = extract_json(raw)
                if not isinstance(decision, dict):
                    raise ValueError("reply must be a JSON object")
            except ValueError as exc:
                step.error = f"malformed reply: {exc}"
                step.observation = (
                    "Your reply could not be parsed. Reply with a single JSON object "
                    'containing either "tool" and "args", or "final_answer".'
                )
                run.steps.append(step)
                messages += [
                    {"role": "assistant", "content": truncate(raw, 400)},
                    {"role": "user", "content": f"OBSERVATION: {step.observation}"},
                ]
                LOGGER.warning("Agent step %d: %s", step_number, step.error)
                continue

            step.thought = str(decision.get("thought") or "").strip() or None

            if "final_answer" in decision:
                run.final_answer = str(decision["final_answer"]).strip()
                run.steps.append(step)
                run.stopped_because = "final_answer"
                LOGGER.info("Agent finished after %d step(s)", step_number)
                return run

            step.tool = decision.get("tool")
            step.args = decision.get("args") if isinstance(decision.get("args"), dict) else {}
            run.tool_calls += 1

            try:
                result = toolbox.call(step.tool, step.args)
                step.observation = _observation_text(result)
                run.successful_tool_calls += 1
            except ToolError as exc:
                # Recoverable by design: hand the message back and let the
                # model try again with the remaining steps.
                step.error = str(exc)
                step.observation = f"ERROR: {exc}"
                LOGGER.warning("Agent step %d tool error: %s", step_number, exc)

            run.steps.append(step)
            messages += [
                {"role": "assistant", "content": truncate(raw, 400)},
                {"role": "user", "content": f"OBSERVATION: {step.observation}"},
            ]

        # Step limit reached. Stop calling tools and answer from what we have.
        run.stopped_because = "max_steps_reached"
        LOGGER.warning("Agent hit the %d-step limit; asking for a final answer", max_steps)
        messages.append(
            {
                "role": "user",
                "content": (
                    f"You have used all {max_steps} tool calls. Answer now using only "
                    "the observations above. If they are not enough, say exactly what "
                    "is missing. Reply with plain text, not JSON."
                ),
            }
        )
        run.final_answer = llm.chat(messages, max_new_tokens=300).strip()
        return run
    finally:
        if owns_llm:
            llm.release()
