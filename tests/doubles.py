"""Test doubles: they replace the heavy parts (embedding model, Ollama) so tests run offline.

* ``BagOfWordsEmbedder``: hashes words into a vector, so cosine similarity reflects shared words.
  It is NOT a semantic model; it only lets retrieval tests behave in a predictable way.
* ``FakeLLM``: answers from the first context block (extractive) and records what it was given.
"""
from __future__ import annotations

import hashlib
import re

import numpy as np

from app.rag.llm import LLMUnavailableError

_STOP = set("a an and are as at be by do for from how i in is it of on or the to what when who which with you your my can may must "
            "about per each this that there their them they we our".split())


class BagOfWordsEmbedder:
    name = "bow-test-embedder"
    dimension = 4096

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dimension, dtype=np.float32)
        for w in re.findall(r"[a-z0-9]+", text.lower()):
            if w in _STOP:
                continue
            v[int.from_bytes(hashlib.md5(w.encode()).digest()[:4], "little") % self.dimension] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_documents(self, texts):
        return np.vstack([self._vec(t) for t in texts])

    def embed_query(self, text):
        return self._vec(text)


class FakeLLM:
    model = "fake-llm"

    def __init__(self, mode: str = "extractive"):
        self.mode = mode
        self.calls: list[tuple[str, str]] = []

    def generate(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        if self.mode == "error":
            raise LLMUnavailableError("Ollama is not reachable (test)")
        if self.mode == "not_found":
            return "NOT_FOUND"
        m = re.search(r"\[1\] \([^)]*\)\n(.+)", user)
        sentence = (m.group(1) if m else "").split(". ")[0][:200]
        if self.mode == "no_markers":
            return sentence
        return f"{sentence} [1]"

    def health(self) -> dict:
        return {"reachable": True, "model": self.model, "model_available": True}
