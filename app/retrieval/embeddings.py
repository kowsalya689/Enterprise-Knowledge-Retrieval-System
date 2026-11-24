"""Embeddings with Sentence Transformers (default model: Nomic Embed Text v2 MoE).

Nomic models expect a task prefix.  Chunks are embedded as
``search_document: <title path>\\n<chunk text>`` (the same as
``prompt_name="passage"`` in the model card) and questions, later, as
``search_query: <question>``.  Prefixes are settings, so another model
(for example BGE-small, which uses none) is a change in ``.env`` only.
"""
from __future__ import annotations

import logging
from typing import Optional, Protocol, Sequence

import numpy as np

from app.config import Settings, get_settings
from app.ingestion.models import Chunk

logger = logging.getLogger(__name__)


class EmbeddingModel(Protocol):
    """What the rest of the code needs from an embedding model."""
    name: str

    @property
    def dimension(self) -> int: ...
    def embed_documents(self, texts: Sequence[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


def chunk_embedding_text(chunk: Chunk) -> str:
    """Text that is embedded: title path first, then the chunk text."""
    return f"{chunk.title_path}\n{chunk.text}"


def _normalise(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)


class SentenceTransformerEmbedder:
    """Lazy wrapper: the model is downloaded and loaded on first use."""

    def __init__(self, settings: Optional[Settings] = None, model=None):
        self.settings = settings or get_settings()
        self.name = self.settings.embedding_model
        self._model = model                       # a ready model can be injected (tests)

    # ----------------------------------------------------------------- loading
    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError("sentence-transformers is not installed: pip install -r requirements.txt") from exc
            s = self.settings
            logger.info("Loading embedding model '%s' (first run downloads the model files)...", s.embedding_model)
            kwargs = {"trust_remote_code": s.embedding_trust_remote_code}
            if s.embedding_dim:
                kwargs["truncate_dim"] = s.embedding_dim
            if s.embedding_device:
                kwargs["device"] = s.embedding_device
            self._model = SentenceTransformer(s.embedding_model, **kwargs)
            self._model.max_seq_length = s.embedding_max_seq_length
            if getattr(self._model, "default_prompt_name", None):
                logger.warning("Model has a default prompt; the configured prefix would be added twice")
            logger.info("Model ready: %d dimensions, max %d tokens", self.dimension, self._model.max_seq_length)
        return self._model

    @property
    def dimension(self) -> int:
        model = self._load()
        dim = model.get_sentence_embedding_dimension()
        configured = self.settings.embedding_dim
        return min(dim, configured) if configured else dim

    # --------------------------------------------------------------- embedding
    @staticmethod
    def _prefixed(prefix: str, text: str) -> str:
        return f"{prefix.rstrip()} {text}" if prefix.strip() else text

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        model = self._load()
        out = []
        size = max(1, self.settings.embedding_batch_size)
        for i in range(0, len(texts), size):
            batch = list(texts[i:i + size])
            vectors = model.encode(batch, batch_size=size, convert_to_numpy=True, normalize_embeddings=False,
                                   show_progress_bar=False)
            out.append(np.asarray(vectors, dtype=np.float32))
            logger.info("  embedded %d/%d chunks", min(i + size, len(texts)), len(texts))
        vectors = np.vstack(out) if out else np.zeros((0, self.dimension), dtype=np.float32)
        if self.settings.embedding_dim and vectors.shape[1] > self.settings.embedding_dim:
            vectors = vectors[:, : self.settings.embedding_dim]       # Matryoshka truncation
        return _normalise(vectors)

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Embed chunk texts with the document prefix; returns unit-length vectors."""
        return self._encode([self._prefixed(self.settings.embedding_doc_prefix, t) for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode([self._prefixed(self.settings.embedding_query_prefix, text)])[0]

    # --------------------------------------------------------------- diagnostics
    def token_stats(self, texts: Sequence[str]) -> dict:
        """How many inputs are longer than the model limit (they would be truncated)."""
        model = self._load()
        tokenizer = getattr(model, "tokenizer", None)
        if tokenizer is None or not texts:
            return {"max_tokens": None, "over_limit": 0, "limit": self.settings.embedding_max_seq_length}
        prefixed = [self._prefixed(self.settings.embedding_doc_prefix, t) for t in texts]
        lengths = [len(ids) for ids in tokenizer(prefixed, add_special_tokens=True, truncation=False)["input_ids"]]
        limit = self.settings.embedding_max_seq_length
        return {"max_tokens": max(lengths), "over_limit": sum(1 for n in lengths if n > limit), "limit": limit}


class ThreadSafeEmbedder:
    """Serialises calls to one model so uploads and questions can arrive at the same time."""

    def __init__(self, inner):
        import threading
        self._inner = inner
        self._lock = threading.Lock()
        self.name = inner.name
        self._dimension = None

    @property
    def dimension(self) -> int:
        if self._dimension is None:                 # asked once; afterwards /health never waits for a running embedding
            with self._lock:
                self._dimension = self._inner.dimension
        return self._dimension

    def embed_documents(self, texts):
        with self._lock:
            return self._inner.embed_documents(texts)

    def embed_query(self, text):
        with self._lock:
            return self._inner.embed_query(text)

    def token_stats(self, texts):
        with self._lock:
            fn = getattr(self._inner, "token_stats", None)
            return fn(texts) if fn else {}
