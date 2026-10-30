"""Part 1b -- timestamped speech recognition.

Wraps Whisper in a small object whose model can be released on demand.
That matters on a 16 GB unified-memory machine: the ASR model and the
4B language model do not fit at the same time, so the pipeline loads
one, frees it, then loads the other.

A note on confidence, which the project brief calls out explicitly: the
per-segment ``avg_logprob`` reported here is the model's mean token log
probability. It is **not** a calibrated probability that the text is
correct, and it is never rescaled into a "percent confidence". It is
useful only for ranking segments against each other within one run.
"""

from __future__ import annotations

import os
import time
from typing import Any

from .config import Config
from .exceptions import TranscriptionError
from .media import load_audio
from .utils import LOGGER, free_memory, set_seed

# Whisper emits these when it hears no speech; keeping them would create
# empty segments that pollute chunking and retrieval.
_NOISE_ONLY = {"", "-", "...", "…", "。", ".", "!", "?", "Thank you.", "字幕由 Amara.org 社群提供"}

# Below this, a segment's audio window is mostly feature-extractor padding
# and the language argmax over it is unreliable, so the window is widened.
_MIN_DETECT_SECONDS = 3.0


def _resolve_dtype(name: str):
    import torch

    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def _language_name(tokenizer, token_id: int) -> str:
    """Map a Whisper language token id to the pipeline's own label style.

    ``detect_language`` returns token ids whose text form is ``<|zh|>``,
    while the ASR pipeline reports ``"chinese"``. Both flow into the same
    ``segment["language"]`` field, so they have to agree or chunking would
    split on a difference of spelling.
    """
    from transformers.models.whisper.tokenization_whisper import TO_LANGUAGE_CODE

    token = tokenizer.decode([token_id]).strip()
    code = token.removeprefix("<|").removesuffix("|>")
    for name, mapped in TO_LANGUAGE_CODE.items():
        if mapped == code:
            return name
    return code


class SpeechRecognizer:
    """Whisper transcription with an explicit model lifecycle.

    Usage:
        asr = SpeechRecognizer(config)
        segments, meta = asr.transcribe("meeting.wav")
        asr.release()          # frees ~1.6 GB before the LLM is loaded
    """

    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self._pipe = None
        self._loaded_dtype: str | None = None

    # -- lifecycle -----------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        return self._pipe is not None

    def load(self, dtype: str | None = None) -> None:
        """Build the ASR pipeline. Idempotent."""
        dtype = dtype or self.config.asr_dtype
        if self._pipe is not None and self._loaded_dtype == dtype:
            return
        if self._pipe is not None:
            self.release()

        from transformers import pipeline

        LOGGER.info(
            "Loading ASR model %s on %s (%s)", self.config.asr_model, self.config.device, dtype
        )
        started = time.perf_counter()
        self._pipe = pipeline(
            task="automatic-speech-recognition",
            model=self.config.asr_model,
            device=self.config.device,
            # transformers 5.x renamed `torch_dtype` to `dtype`.
            dtype=_resolve_dtype(dtype),
        )
        self._loaded_dtype = dtype
        LOGGER.info("ASR model ready in %.1fs", time.perf_counter() - started)

    def release(self) -> None:
        """Drop the model and empty the allocator cache."""
        if self._pipe is None:
            return
        LOGGER.info("Releasing ASR model %s", self.config.asr_model)
        del self._pipe
        self._pipe = None
        self._loaded_dtype = None
        free_memory(self.config.device)

    def __enter__(self) -> "SpeechRecognizer":
        self.load()
        return self

    def __exit__(self, *exc_info) -> None:
        self.release()

    # -- transcription -------------------------------------------------

    def transcribe(
        self,
        audio_path: str | os.PathLike,
        *,
        task: str = "transcribe",
        language: str | None = None,
        media_metadata: dict | None = None,
    ) -> tuple[list[dict], dict]:
        """Transcribe an audio file into timestamped segments.

        Args:
            audio_path: mono 16 kHz WAV, as produced by ``media.prepare_audio``.
            task: ``"transcribe"`` keeps the spoken language; ``"translate"``
                makes Whisper decode straight into English. The multilingual
                extension uses both on the same audio.
            language: force a source language (e.g. ``"zh"``). Left as None
                the model detects one language for the whole file; see
                :meth:`detect_segment_languages` for why that is not enough
                for a bilingual recording.
            media_metadata: output of ``media.probe_metadata``, folded into
                the returned metadata so the report can quote it.

        Returns:
            ``(segments, metadata)`` where each segment is
            ``{"id", "start", "end", "text", "language", "avg_logprob"}``
            with numeric timestamps in seconds.

        Raises:
            TranscriptionError: the model returned no usable speech.
        """
        set_seed(self.config.seed)
        audio = load_audio(audio_path, self.config.target_sample_rate)
        audio_seconds = len(audio) / self.config.target_sample_rate

        segments, elapsed, used_dtype = self._run(audio, task=task, language=language)

        # fp16 on MPS occasionally degenerates to empty output. Rather than
        # returning an empty transcript, fall back to fp32 once and say so.
        if not segments and used_dtype != "float32":
            LOGGER.warning("ASR produced no segments in %s; retrying in float32", used_dtype)
            self.load(dtype="float32")
            segments, elapsed, used_dtype = self._run(audio, task=task, language=language)

        if not segments:
            raise TranscriptionError(
                f"Whisper returned no speech segments for {audio_path}. "
                "The recording may be silent, or the audio track may be corrupt."
            )

        metadata = {
            "asr_model": self.config.asr_model,
            "asr_task": task,
            "asr_forced_language": language,
            "asr_dtype": used_dtype,
            "device": self.config.device,
            "sample_rate": self.config.target_sample_rate,
            "audio_duration_seconds": round(audio_seconds, 3),
            "transcription_seconds": round(elapsed, 2),
            "realtime_factor": round(elapsed / audio_seconds, 3) if audio_seconds else None,
            "segment_count": len(segments),
            "long_form_algorithm": "chunked" if self.config.asr_use_chunked else "sequential",
            "confidence_note": (
                "No per-segment confidence score is exposed. Where the decoder "
                "reports avg_logprob it is passed through unchanged as a mean "
                "token log probability; it is not calibrated and is never "
                "rescaled into a percentage."
            ),
        }
        if media_metadata:
            metadata["source_media"] = media_metadata
        return segments, metadata

    def detect_segment_languages(
        self,
        audio_path: str | os.PathLike,
        segments: list[dict],
        *,
        batch_size: int = 8,
    ) -> list[dict]:
        """Re-label each segment with Whisper's prediction for its own audio.

        Sequential long-form decoding detects the language exactly once.
        ``WhisperForConditionalGeneration.generate`` calls
        ``_retrieve_init_tokens`` before the segment loop starts, and that
        runs ``detect_language`` on the first ``num_segment_frames`` — the
        opening 30 seconds — then reuses the resulting token for every
        segment in the file. A recording that opens in Mandarin and later
        switches to English therefore comes back labelled ``chinese``
        throughout, even though the English is transcribed correctly as
        English text.

        Chunked mode (``config.asr_use_chunked``) does detect per chunk,
        because each chunk is a separate ``generate`` call, but it also
        degrades long-form timestamps, and the citation format depends on
        those. So the transcript is still decoded sequentially and the
        language labels are recovered here instead: Whisper's own
        ``detect_language`` is applied to each segment's audio window.

        Segments shorter than ``_MIN_DETECT_SECONDS`` are widened
        symmetrically before detection — a 0.8 s window is mostly padding
        and the prediction from it is close to a coin flip.

        Args:
            audio_path: the same mono 16 kHz WAV passed to ``transcribe``.
            segments: segments from ``transcribe``, modified in place.
            batch_size: segments per forward pass.

        Returns:
            The same list, with ``language`` replaced and the model's
            original whole-file guess preserved as ``language_file_level``.
        """
        if not segments:
            return segments

        self.load()
        model = self._pipe.model
        extractor = self._pipe.feature_extractor
        sample_rate = self.config.target_sample_rate

        audio = load_audio(audio_path, sample_rate)
        audio_seconds = len(audio) / sample_rate

        windows = []
        for segment in segments:
            start, end = float(segment["start"]), float(segment["end"])
            if end - start < _MIN_DETECT_SECONDS:
                pad = (_MIN_DETECT_SECONDS - (end - start)) / 2
                start, end = max(0.0, start - pad), min(audio_seconds, end + pad)
            window = audio[int(start * sample_rate) : int(end * sample_rate)]
            # An empty slice would make the feature extractor emit all-padding
            # frames, and argmax over those is meaningless.
            windows.append(window if len(window) else audio[: sample_rate])

        started = time.perf_counter()
        detected: list[str] = []
        for offset in range(0, len(windows), batch_size):
            batch = windows[offset : offset + batch_size]
            features = extractor(
                batch, sampling_rate=sample_rate, return_tensors="pt"
            ).input_features.to(model.device, dtype=model.dtype)
            lang_ids = model.detect_language(
                input_features=features, generation_config=model.generation_config
            )
            detected.extend(_language_name(self._pipe.tokenizer, int(i)) for i in lang_ids)

        changed = 0
        for segment, language in zip(segments, detected):
            if segment.get("language") != language:
                changed += 1
            segment["language_file_level"] = segment.get("language")
            segment["language"] = language

        LOGGER.info(
            "Per-segment language detection: %d/%d segments relabelled in %.1fs (%s)",
            changed, len(segments), time.perf_counter() - started,
            ", ".join(sorted(set(detected))),
        )
        return segments

    def translate_segments(
        self,
        audio_path: str | os.PathLike,
        segments: list[dict],
    ) -> list[str]:
        """Translate each segment's own audio into English, one at a time.

        The obvious implementation — one ``task="translate"`` pass over the
        whole file, then match its output back to the transcript by time
        overlap — does not work here. The translate pass re-segments the
        audio and, on this recording, returned 7 segments for 270 seconds
        with one of them spanning 162 seconds, so any overlap-based
        alignment silently pairs a segment with text from elsewhere in the
        recording.

        Translating each segment separately makes the alignment exact by
        construction: row *i* of the result is the translation of segment
        *i*, with no matching step to get wrong. It also puts this route on
        the same footing as the LLM route, which already translates one
        segment at a time, so the two are actually comparable.

        The cost is context: a 4-second window translated in isolation has
        no surrounding discourse to disambiguate against. That is a real
        limitation of the comparison and is stated in the report rather
        than hidden.

        Args:
            audio_path: the same mono 16 kHz WAV passed to ``transcribe``.
            segments: segments to translate.

        Returns:
            One English string per segment, in the same order. A segment
            whose decode yields nothing becomes an empty string rather
            than dropping a row and shifting every later one.
        """
        if not segments:
            return []

        self.load()
        sample_rate = self.config.target_sample_rate
        audio = load_audio(audio_path, sample_rate)
        audio_seconds = len(audio) / sample_rate

        translations: list[str] = []
        started = time.perf_counter()
        for segment in segments:
            start = max(0.0, float(segment["start"]))
            end = min(audio_seconds, float(segment["end"]))
            window = audio[int(start * sample_rate) : int(end * sample_rate)]
            if not len(window):
                translations.append("")
                continue

            raw = self._pipe(
                {"array": window, "sampling_rate": sample_rate},
                return_timestamps=False,
                generate_kwargs={"task": "translate"},
            )
            translations.append((raw.get("text") or "").strip())

        LOGGER.info(
            "Whisper translate route: %d segments in %.1fs using %s",
            len(segments), time.perf_counter() - started, self.config.asr_model,
        )
        return translations

    def _run(self, audio, *, task: str, language: str | None) -> tuple[list[dict], float, str]:
        """One decode pass. Returns (segments, elapsed_seconds, dtype_used)."""
        self.load()
        generate_kwargs: dict[str, Any] = {"task": task}
        if language:
            generate_kwargs["language"] = language

        call_kwargs: dict[str, Any] = {
            "return_timestamps": True,
            "return_language": True,
            "generate_kwargs": generate_kwargs,
        }
        if self.config.asr_use_chunked:
            call_kwargs["chunk_length_s"] = self.config.asr_chunk_length_s
            call_kwargs["batch_size"] = self.config.asr_batch_size

        started = time.perf_counter()
        raw = self._pipe(
            {"array": audio, "sampling_rate": self.config.target_sample_rate},
            **call_kwargs,
        )
        elapsed = time.perf_counter() - started

        audio_seconds = len(audio) / self.config.target_sample_rate
        segments = _normalise_chunks(raw.get("chunks") or [], audio_seconds)
        return segments, elapsed, self._loaded_dtype or self.config.asr_dtype


def _normalise_chunks(chunks: list[dict], audio_seconds: float) -> list[dict]:
    """Turn raw pipeline chunks into clean, numerically sane segments.

    Whisper can emit a ``None`` end timestamp on the final chunk, ranges
    that run past the end of the audio, and text that is punctuation
    only. All three are fixed here so that everything downstream can
    assume ``0 <= start <= end <= duration`` and non-empty text.
    """
    segments: list[dict] = []
    for chunk in chunks:
        text = (chunk.get("text") or "").strip()
        if text in _NOISE_ONLY:
            continue

        # The pipeline reports the span as a 2-sequence, but whether it is
        # a tuple or a list has varied between transformers versions, and
        # the final chunk's end can be None.
        timestamp = chunk.get("timestamp")
        if isinstance(timestamp, (tuple, list)) and len(timestamp) >= 2:
            start, end = timestamp[0], timestamp[1]
        else:
            start, end = None, None

        start = 0.0 if start is None else float(start)
        end = audio_seconds if end is None else float(end)
        start = min(max(0.0, start), audio_seconds)
        end = min(max(start, end), audio_seconds)

        segment = {
            "id": "",  # assigned below, once the list is final
            "start": round(start, 3),
            "end": round(end, 3),
            "text": text,
            "language": chunk.get("language"),
        }
        # Only carried through when the decoder actually reports it. We do
        # not synthesise a confidence number when one is unavailable.
        if chunk.get("avg_logprob") is not None:
            segment["avg_logprob"] = float(chunk["avg_logprob"])
        segments.append(segment)

    segments.sort(key=lambda s: (s["start"], s["end"]))
    for index, segment in enumerate(segments, start=1):
        segment["id"] = f"S{index:03d}"
    return segments


def transcribe(
    audio_path: str | os.PathLike,
    config: Config | None = None,
    **kwargs,
) -> tuple[list[dict], dict]:
    """One-shot transcription that loads and releases the model.

    Convenient for scripts and tests. Pipelines that transcribe more than
    once should hold a :class:`SpeechRecognizer` instead, so the model is
    not reloaded per call.
    """
    recognizer = SpeechRecognizer(config)
    try:
        return recognizer.transcribe(audio_path, **kwargs)
    finally:
        recognizer.release()


def transcribe_audio(audio_path: str | os.PathLike, config: Config | None = None, **kwargs):
    """Return timestamped transcript segments and metadata.

    The name from the project brief's starter code; an alias for
    :func:`transcribe`.
    """
    return transcribe(audio_path, config, **kwargs)
