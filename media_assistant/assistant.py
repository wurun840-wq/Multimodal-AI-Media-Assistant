"""The facade that ties the stages together and owns model lifetimes.

Ordering here is driven by memory. On a 16 GB unified-memory machine the
recommended stack does not fit all at once, so ``process`` runs every
Whisper pass first, releases the ASR model, and only then loads the LLM.
The embedding model is small enough to stay resident, and has to, since
every later search re-encodes a query.

The class also holds the run's state -- segments, index, analysis,
toolbox -- so that ``search``, ``ask`` and ``agent`` are one-liners
rather than plumbing.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from .agent import AgentRun, run_agent
from .config import Config
from .embeddings import Embedder
from .language import analyze_transcript
from .llm import LocalLLM
from .media import prepare_audio
from .multilingual import (
    build_bilingual_transcript,
    compare_translation_routes,
    cross_lingual_comparison,
    language_distribution,
    language_timeline,
    render_multilingual_figure,
)
from .retrieval import TranscriptIndex, answer_question, build_transcript_index
from .speech import SpeechRecognizer
from .tools import Toolbox
from .utils import LOGGER, configure_logging, total_duration


class MediaAssistant:
    """End-to-end assistant for one recording."""

    def __init__(self, config: Config | None = None, *, verbose: bool = False):
        self.config = config or Config()
        if verbose:
            configure_logging()

        # Heavy components, created on demand and released explicitly.
        self._recognizer: SpeechRecognizer | None = None
        self._llm: LocalLLM | None = None
        self._embedder: Embedder | None = None

        # Run state.
        self.media_metadata: dict = {}
        self.transcript_metadata: dict = {}
        self.segments: list[dict] = []
        # One English string per segment, aligned by position.
        self.translation_texts: list[str] = []
        self.index: TranscriptIndex | None = None
        self.analysis: dict = {}
        self.toolbox: Toolbox | None = None

    # -- components ----------------------------------------------------

    @property
    def recognizer(self) -> SpeechRecognizer:
        if self._recognizer is None:
            self._recognizer = SpeechRecognizer(self.config)
        return self._recognizer

    @property
    def llm(self) -> LocalLLM:
        if self._llm is None:
            self._llm = LocalLLM(self.config)
        return self._llm

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = Embedder(self.config)
        return self._embedder

    def release_all(self) -> None:
        """Free every loaded model."""
        for component in (self._recognizer, self._llm, self._embedder):
            if component is not None:
                component.release()

    def __enter__(self) -> "MediaAssistant":
        return self

    def __exit__(self, *exc_info) -> None:
        self.release_all()

    # -- pipeline ------------------------------------------------------

    def process(
        self,
        input_path: str | os.PathLike,
        *,
        workdir: str | os.PathLike = ".cache_media",
        translate_pass: bool = True,
        analyse: bool = True,
    ) -> dict:
        """Run ingestion, transcription, indexing and analysis.

        Args:
            input_path: audio or video file.
            workdir: where the converted WAV is written.
            translate_pass: also decode the audio straight into English
                with Whisper, for the Part 5 translation comparison.
                Skipped automatically for an English-only recording.
            analyse: run the structured LLM analysis (Part 2).

        Returns:
            A dictionary with the transcript, its metadata, the index
            summary and the analysis.
        """
        wav_path, media_metadata = prepare_audio(
            input_path, workdir=workdir, sample_rate=self.config.target_sample_rate
        )
        self.media_metadata = media_metadata

        LOGGER.info("Transcribing %s", media_metadata["filename"])
        self.segments, self.transcript_metadata = self.recognizer.transcribe(
            wav_path, media_metadata=media_metadata
        )

        # Sequential decoding labels every segment with the language it
        # detected from the first 30 seconds, so the labels have to be
        # recovered per segment before anything reads them. Chunking splits
        # on language, so this must happen before the index is built.
        self.recognizer.detect_segment_languages(wav_path, self.segments)

        self.recognizer.release()  # ~1.6 GB back before the next model loads

        # The translate route needs a different checkpoint -- see
        # Config.translate_model -- so it runs after the transcription
        # model is released rather than sharing its load.
        languages = language_distribution(self.segments)
        if translate_pass and languages["is_multilingual"]:
            LOGGER.info(
                "Recording is multilingual %s; running the translate route with %s",
                languages["languages"], self.config.translate_model,
            )
            self.translation_texts = self._translate_route(wav_path)
        elif translate_pass:
            LOGGER.info("Recording is monolingual; skipping the Whisper translate route")

        self.index = build_transcript_index(self.segments, self.embedder, self.config)
        self.toolbox = Toolbox(
            index=self.index, segments=self.segments, metadata=self.transcript_metadata
        )

        if analyse:
            self.analysis = analyze_transcript(self.segments, llm=self.llm, config=self.config)

        return {
            "media": self.media_metadata,
            "transcript_metadata": self.transcript_metadata,
            "segments": self.segments,
            "index": self.index.summary,
            "analysis": self.analysis,
            "languages": languages,
        }

    def use_transcript(self, segments: list[dict], metadata: dict | None = None) -> None:
        """Load a transcript directly, skipping ingestion and ASR.

        Used by the tests and by any re-run that should not pay for
        transcription again.
        """
        if not segments:
            raise ValueError("segments must not be empty")
        self.segments = segments
        self.transcript_metadata = metadata or {}
        self.media_metadata = self.transcript_metadata.get("source_media", {})
        self.index = build_transcript_index(segments, self.embedder, self.config)
        self.toolbox = Toolbox(
            index=self.index, segments=segments, metadata=self.transcript_metadata
        )

    def load_transcript(self, path: str | os.PathLike) -> None:
        """Load a transcript from a ``transcript.json`` written earlier."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        self.use_transcript(payload["segments"], payload.get("metadata"))

    def _translate_route(self, wav_path: str | os.PathLike) -> list[str]:
        """Run the Whisper translate route on its own checkpoint.

        Loaded and released inside this method: ``translate_model`` is the
        full large-v3, which is considerably heavier than the turbo
        checkpoint used for transcription, and holding both plus the 4B
        language model would not fit in 16 GB.
        """
        from dataclasses import replace

        # Segments already in English pass straight through, exactly as the
        # LLM route treats them. Decoding them again would cost a large-v3
        # forward pass each and make the two routes differ on rows where
        # there is nothing to compare.
        pending = [
            (index, segment)
            for index, segment in enumerate(self.segments)
            if (segment.get("language") or "").lower() != "english"
        ]

        translator = SpeechRecognizer(replace(self.config, asr_model=self.config.translate_model))
        try:
            decoded = translator.translate_segments(wav_path, [s for _, s in pending])
        finally:
            translator.release()

        translations = [segment["text"] for segment in self.segments]
        for (index, _), text in zip(pending, decoded):
            translations[index] = text
        return translations

    def _require_index(self, action: str) -> TranscriptIndex:
        if self.index is None:
            raise RuntimeError(
                f"cannot {action} before a transcript exists; call process() or "
                "use_transcript() first"
            )
        return self.index

    # -- Part 3 --------------------------------------------------------

    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        """Semantic transcript retrieval with timestamps and scores."""
        return self._require_index("search").search(query, top_k=top_k)

    def ask(self, question: str, top_k: int | None = None) -> dict:
        """Grounded question answering with timestamp citations."""
        return answer_question(
            question, self._require_index("ask"), llm=self.llm, config=self.config, top_k=top_k
        )

    # -- Part 4 --------------------------------------------------------

    def agent(self, request: str, max_steps: int | None = None) -> AgentRun:
        """Bounded tool-using agent loop."""
        self._require_index("run the agent")
        return run_agent(
            request, self.toolbox, llm=self.llm, config=self.config, max_steps=max_steps
        )

    # -- Part 2 --------------------------------------------------------

    def analyse(self) -> dict:
        """Run (or re-run) the structured analysis on the loaded transcript."""
        if not self.segments:
            raise RuntimeError("cannot analyse before a transcript exists")
        self.analysis = analyze_transcript(self.segments, llm=self.llm, config=self.config)
        return self.analysis

    # -- Part 5 --------------------------------------------------------

    def multilingual_analysis(
        self,
        *,
        figure_path: str | os.PathLike = "visual_output.png",
        translate: bool = True,
    ) -> dict:
        """Run the Option C extension and render its visual output."""
        index = self._require_index("run the multilingual analysis")

        distribution = language_distribution(self.segments)
        timeline = language_timeline(self.segments)
        cross_rows = cross_lingual_comparison(index, top_k=self.config.top_k)

        bilingual: list[dict] = []
        comparison: list[dict] = []
        if translate and distribution["is_multilingual"]:
            bilingual = build_bilingual_transcript(
                self.segments, self.llm, self.translation_texts, self.config
            )
            comparison = compare_translation_routes(bilingual)

        figure = render_multilingual_figure(
            timeline,
            cross_rows,
            output_path=str(figure_path),
            title=self.media_metadata.get("filename", "recording"),
        )

        return {
            "extension": "Option C — multilingual media mode",
            "language_distribution": distribution,
            "language_timeline": timeline,
            "cross_lingual_comparison": cross_rows,
            "translation_comparison": comparison,
            "bilingual_segments": bilingual,
            "figure_path": figure,
        }

    # -- artifacts -----------------------------------------------------

    def transcript_payload(self) -> dict:
        """The object written to ``transcript.json``."""
        return {
            "metadata": {
                **self.transcript_metadata,
                "duration_seconds": round(total_duration(self.segments), 2),
                "config": self.config.describe(),
            },
            "segments": self.segments,
        }
