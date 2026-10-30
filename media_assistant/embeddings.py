"""Text embedding model wrapper.

The third heavy model, with the same explicit lifecycle as the ASR and
LLM wrappers. It is by far the smallest of the three (0.6B), so the
pipeline keeps it resident while the other two are swapped in and out.

Qwen3-Embedding is asymmetric: queries are meant to carry a short task
instruction and documents are not. Getting that right matters most for
the cross-lingual retrieval this project is evaluated on, so the query
side always goes through :meth:`Embedder.encode_queries`.
"""

from __future__ import annotations

import time

import numpy as np

from .config import Config
from .utils import LOGGER, free_memory

# Used when the model does not ship a "query" prompt of its own.
QUERY_INSTRUCTION = (
    "Instruct: Given a question about a recorded meeting, retrieve the "
    "transcript passages that answer it\nQuery: "
)


class Embedder:
    """Encodes transcript chunks and queries into a shared vector space."""

    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self._model = None
        self._has_query_prompt = False

    # -- lifecycle -----------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        """Load the sentence-transformers model. Idempotent."""
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer

        LOGGER.info(
            "Loading embedding model %s on %s",
            self.config.embedding_model, self.config.device,
        )
        started = time.perf_counter()
        self._model = SentenceTransformer(
            self.config.embedding_model,
            device=self.config.device,
            model_kwargs={"dtype": self.config.embedding_dtype},
        )
        prompts = getattr(self._model, "prompts", None) or {}
        self._has_query_prompt = "query" in prompts
        LOGGER.info(
            "Embedding model ready in %.1fs (built-in query prompt: %s)",
            time.perf_counter() - started, self._has_query_prompt,
        )

    def release(self) -> None:
        if self._model is None:
            return
        LOGGER.info("Releasing embedding model %s", self.config.embedding_model)
        del self._model
        self._model = None
        free_memory(self.config.device)

    def __enter__(self) -> "Embedder":
        self.load()
        return self

    def __exit__(self, *exc_info) -> None:
        self.release()

    # -- encoding ------------------------------------------------------

    @property
    def dimension(self) -> int:
        self.load()
        return int(self._model.get_sentence_embedding_dimension())

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        """Embed passages. No instruction prefix, by design."""
        return self._encode(texts, query=False)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        """Embed questions, with the model's query instruction applied."""
        return self._encode(texts, query=True)

    def _encode(self, texts: list[str], *, query: bool) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        self.load()

        kwargs = {"normalize_embeddings": True, "convert_to_numpy": True, "show_progress_bar": False}
        if query:
            if self._has_query_prompt:
                kwargs["prompt_name"] = "query"
            else:
                kwargs["prompt"] = QUERY_INSTRUCTION

        vectors = self._model.encode(list(texts), **kwargs)
        return np.asarray(vectors, dtype=np.float32)
