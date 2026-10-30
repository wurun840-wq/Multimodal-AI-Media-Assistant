"""Central configuration.

Every tunable value lives here. A run is reproducible from a single
``Config`` object, and no magic number is buried inside a pipeline stage.
``Config.describe()`` is what the README and report quote as the exact
configuration a result came from.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


def _default_device() -> str:
    """Pick the best available accelerator without importing torch eagerly."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass(frozen=True)
class Config:
    """Model choices and pipeline parameters for one run."""

    # --- models -------------------------------------------------------
    asr_model: str = "openai/whisper-large-v3-turbo"
    # A separate checkpoint for the Part 5 translate route, on purpose.
    # large-v3-turbo is a pruned large-v3 (32 decoder layers cut to 4)
    # re-finetuned on multilingual *transcription* data only, so it was
    # never taught the translation task: given task="translate" it emits
    # the source language instead of English, which was verified on this
    # recording. large-v3 keeps the full decoder and does translate.
    translate_model: str = "openai/whisper-large-v3"
    llm_model: str = "Qwen/Qwen3-4B-Instruct-2507"
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"

    # --- hardware -----------------------------------------------------
    device: str = field(default_factory=_default_device)
    # Whisper is numerically comfortable in fp16. Qwen3 is trained in
    # bf16 and can overflow to inf in fp16, so the two are set apart.
    asr_dtype: str = "float16"
    llm_dtype: str = "bfloat16"
    embedding_dtype: str = "float32"

    # --- audio / ASR --------------------------------------------------
    target_sample_rate: int = 16_000
    # False selects Whisper's sequential long-form algorithm, which gives
    # noticeably better timestamps than the chunked one. Chunked is
    # faster and is kept available for long inputs.
    asr_use_chunked: bool = False
    asr_chunk_length_s: int = 30
    asr_batch_size: int = 1

    # --- chunking / retrieval ----------------------------------------
    chunk_target_seconds: float = 30.0
    chunk_max_seconds: float = 45.0
    chunk_overlap_segments: int = 1
    top_k: int = 3
    # Cosine similarity below which we treat the recording as having no
    # evidence for the question. Calibrated on the fixture transcript:
    # questions the recording cannot answer topped out at 0.352, while
    # the weakest answerable question scored 0.520. 0.42 sits between
    # them, leaning low because a false refusal costs more than a false
    # accept -- the model-side sentinel is the second line of defence.
    min_similarity: float = 0.42

    # --- generation ---------------------------------------------------
    llm_max_new_tokens: int = 1024
    llm_temperature: float = 0.0
    analysis_max_retries: int = 2
    max_agent_steps: int = 5
    # The recording is bilingual, so the report would otherwise come out
    # in whichever language happened to dominate the transcript.
    analysis_language: str = "English"

    # --- reproducibility ---------------------------------------------
    seed: int = 20_260_811

    @classmethod
    def small(cls, **overrides) -> "Config":
        """A fallback profile using only the smaller cached models.

        Intended for machines that cannot hold the recommended stack.
        Transcription quality on Chinese drops noticeably, and the 1.7B
        LLM is a hybrid reasoning model whose ``<think>`` blocks the JSON
        extractor has to strip, so this is a fallback and not the
        configuration the submitted results were produced with.
        """
        defaults = dict(
            asr_model="openai/whisper-base",
            llm_model="Qwen/Qwen3-1.7B",
            embedding_model="Qwen/Qwen3-Embedding-0.6B",
        )
        defaults.update(overrides)
        return cls(**defaults)

    def describe(self) -> dict:
        """A JSON-serialisable record of this configuration."""
        return asdict(self)
