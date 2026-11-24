"""Ollama client: sends the question and the retrieved context to a local model."""
from __future__ import annotations

import logging
from typing import Optional, Protocol

import httpx

from app.config import Settings

logger = logging.getLogger(__name__)


class LLMUnavailableError(Exception):
    """Ollama is not reachable, the model is missing, or the call failed."""


class LLMClient(Protocol):
    model: str
    def generate(self, system: str, user: str) -> str: ...
    def health(self) -> dict: ...


class OllamaClient:
    def __init__(self, settings: Settings, client: Optional[httpx.Client] = None):
        self.model = settings.ollama_model
        self._settings = settings
        self._client = client or httpx.Client(base_url=settings.ollama_base_url.rstrip("/"),
                                              timeout=httpx.Timeout(settings.ollama_timeout, connect=10.0))

    def generate(self, system: str, user: str) -> str:
        payload = {
            "model": self.model,
            "stream": False,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "keep_alive": self._settings.ollama_keep_alive,
            "options": {"temperature": self._settings.ollama_temperature, "num_ctx": self._settings.ollama_num_ctx,
                        "num_predict": self._settings.ollama_num_predict},
        }
        try:
            response = self._client.post("/api/chat", json=payload)
        except httpx.HTTPError as exc:
            raise LLMUnavailableError(f"Ollama is not reachable at {self._settings.ollama_base_url}: {exc}") from exc
        if response.status_code != 200:
            detail = _error_text(response)
            hint = f" Pull it first: ollama pull {self.model}" if response.status_code == 404 else ""
            raise LLMUnavailableError(f"Ollama returned {response.status_code}: {detail}.{hint}")
        try:
            return response.json()["message"]["content"].strip()
        except (KeyError, ValueError, TypeError) as exc:
            raise LLMUnavailableError(f"Unexpected Ollama response: {response.text[:200]}") from exc

    def warm_up(self) -> bool:
        """Load the model into memory now (an empty prompt loads it and generates nothing),
        so the first real question does not pay the loading time. Never raises."""
        try:
            response = self._client.post("/api/generate", json={"model": self.model, "keep_alive": self._settings.ollama_keep_alive,
                                                                       "options": {"num_ctx": self._settings.ollama_num_ctx}})
            return response.status_code == 200
        except httpx.HTTPError as exc:
            logger.warning("Ollama warm-up failed: %s", exc)
            return False

    def health(self) -> dict:
        """Is Ollama reachable and is the configured model installed?"""
        try:
            response = self._client.get("/api/tags", timeout=5)
            names = [m.get("name", "") for m in response.json().get("models", [])]
        except (httpx.HTTPError, ValueError) as exc:
            return {"reachable": False, "model": self.model, "model_available": False, "detail": str(exc)}
        wanted = self.model if ":" in self.model else self.model + ":latest"
        return {"reachable": True, "model": self.model, "model_available": wanted in names or self.model in names}


def _error_text(response: httpx.Response) -> str:
    try:
        return str(response.json().get("error", response.text))[:200]
    except ValueError:
        return response.text[:200]