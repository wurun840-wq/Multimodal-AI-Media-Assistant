"""Local instruction-tuned LLM, shared by the analysis, RAG, and agent stages.

One loaded model serves all three, because loading a 4B model three
times on a 16 GB machine is the difference between a pipeline that runs
and one that swaps. Like the ASR wrapper, it exposes an explicit
``release()`` so the caller controls when the memory goes back.

``extract_json`` lives here rather than in the analysis module because
both the structured-analysis stage and the agent loop have to recover a
JSON object from free-form model output, and they should fail the same
way when they cannot.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

from .config import Config
from .utils import LOGGER, free_memory, set_seed

_CODE_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


def _resolve_dtype(name: str):
    import torch

    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]


class LocalLLM:
    """A chat-style wrapper over a local causal LM."""

    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self._model = None
        self._tokenizer = None

    # -- lifecycle -----------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        """Load weights and tokenizer. Idempotent."""
        if self._model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer

        LOGGER.info(
            "Loading LLM %s on %s (%s)",
            self.config.llm_model, self.config.device, self.config.llm_dtype,
        )
        started = time.perf_counter()
        self._tokenizer = AutoTokenizer.from_pretrained(self.config.llm_model)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.llm_model,
            dtype=_resolve_dtype(self.config.llm_dtype),
        ).to(self.config.device)
        self._model.eval()
        LOGGER.info("LLM ready in %.1fs", time.perf_counter() - started)

    def release(self) -> None:
        """Drop the model and empty the allocator cache."""
        if self._model is None:
            return
        LOGGER.info("Releasing LLM %s", self.config.llm_model)
        del self._model
        del self._tokenizer
        self._model = None
        self._tokenizer = None
        free_memory(self.config.device)

    def __enter__(self) -> "LocalLLM":
        self.load()
        return self

    def __exit__(self, *exc_info) -> None:
        self.release()

    # -- generation ----------------------------------------------------

    def chat(
        self,
        messages: list[dict],
        *,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        stop_at: str | None = None,
    ) -> str:
        """Run a chat completion and return only the newly generated text.

        Temperature 0 (the default) means greedy decoding, which keeps a
        run reproducible. The analysis stage raises it slightly on retry
        so a malformed answer is not simply reproduced verbatim.
        """
        self.load()
        import torch

        temperature = self.config.llm_temperature if temperature is None else temperature
        max_new_tokens = max_new_tokens or self.config.llm_max_new_tokens
        set_seed(self.config.seed)

        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self.config.device)

        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self._tokenizer.pad_token_id or self._tokenizer.eos_token_id,
        }
        if temperature and temperature > 0:
            generate_kwargs.update(do_sample=True, temperature=temperature, top_p=0.9)
        else:
            generate_kwargs.update(do_sample=False)

        started = time.perf_counter()
        with torch.no_grad():
            output = self._model.generate(**inputs, **generate_kwargs)
        generated = output[0][inputs["input_ids"].shape[-1] :]
        text = self._tokenizer.decode(generated, skip_special_tokens=True).strip()

        LOGGER.debug(
            "Generated %d tokens in %.1fs", len(generated), time.perf_counter() - started
        )
        if stop_at and stop_at in text:
            text = text.split(stop_at)[0].strip()
        return text

    def chat_json(self, messages: list[dict], **kwargs) -> Any:
        """Chat, then parse the reply as JSON.

        Raises ValueError (from :func:`extract_json`) when no JSON object
        can be recovered, so callers can feed the error back as a retry.
        """
        raw = self.chat(messages, **kwargs)
        return extract_json(raw)


def strip_reasoning(text: str) -> str:
    """Remove ``<think>`` blocks emitted by hybrid reasoning models.

    Qwen3-4B-Instruct-2507 does not produce them, but the smaller cached
    Qwen3-1.7B fallback does, and they would otherwise sit in front of
    the JSON object we are trying to parse.
    """
    return _THINK_RE.sub("", text).strip()


def _find_json_object(text: str) -> str | None:
    """Return the first balanced ``{...}`` span, ignoring braces in strings."""
    start = None
    depth = 0
    in_string = False
    escaped = False

    for index, char in enumerate(text):
        if start is None:
            if char == "{":
                start, depth = index, 1
            continue
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def extract_json(text: str) -> Any:
    """Recover a JSON object from a model reply.

    Handles the three things instruction-tuned models actually do when
    asked for JSON: wrap it in a code fence, prefix it with a sentence of
    commentary, and leave a trailing comma. Anything beyond that is
    reported as an error rather than guessed at.

    Raises:
        ValueError: no JSON object could be parsed. The message names the
            problem so it can be fed back to the model on retry.
    """
    if not text or not text.strip():
        raise ValueError("model returned an empty response")

    cleaned = strip_reasoning(text)

    fenced = _CODE_FENCE_RE.search(cleaned)
    if fenced:
        cleaned = fenced.group(1).strip()

    candidate = _find_json_object(cleaned)
    if candidate is None:
        raise ValueError(
            "no JSON object found in the response (expected text starting with '{')"
        )

    try:
        return json.loads(candidate)
    except json.JSONDecodeError as first_error:
        repaired = _TRAILING_COMMA_RE.sub(r"\1", candidate)
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            raise ValueError(
                f"response is not valid JSON: {first_error.msg} "
                f"at line {first_error.lineno} column {first_error.colno}"
            ) from first_error
