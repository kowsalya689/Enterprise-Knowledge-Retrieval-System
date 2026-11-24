"""Ollama client: request format and error handling (HTTP is simulated)."""
import json

import httpx
import pytest

from app.config import Settings
from app.rag.llm import LLMUnavailableError, OllamaClient


def _client(handler, **kw):
    cfg = Settings(_env_file=None, ollama_model="llama3.2:3b", ollama_temperature=0.0, **kw)
    return OllamaClient(cfg, httpx.Client(transport=httpx.MockTransport(handler), base_url="http://ollama:11434"))


def test_chat_request_shape_and_response_parsing():
    seen = {}

    def handler(request):
        seen["url"], seen["body"] = str(request.url), json.loads(request.content)
        return httpx.Response(200, json={"message": {"role": "assistant", "content": " 18 days [1] "}})

    answer = _client(handler).generate("SYS", "USER")
    assert answer == "18 days [1]"
    assert seen["url"].endswith("/api/chat")
    b = seen["body"]
    assert b["model"] == "llama3.2:3b" and b["stream"] is False and b["options"]["temperature"] == 0.0
    assert [m["role"] for m in b["messages"]] == ["system", "user"]


def test_missing_model_gives_a_helpful_message():
    c = _client(lambda r: httpx.Response(404, json={"error": "model 'llama3.2:3b' not found"}))
    with pytest.raises(LLMUnavailableError, match="ollama pull llama3.2:3b"):
        c.generate("s", "u")


def test_unreachable_server_is_reported():
    def handler(request):
        raise httpx.ConnectError("refused")
    with pytest.raises(LLMUnavailableError, match="not reachable"):
        _client(handler).generate("s", "u")


def test_health_checks_that_the_model_is_installed():
    ok = _client(lambda r: httpx.Response(200, json={"models": [{"name": "llama3.2:3b"}]})).health()
    assert ok == {"reachable": True, "model": "llama3.2:3b", "model_available": True}
    missing = _client(lambda r: httpx.Response(200, json={"models": [{"name": "other:7b"}]})).health()
    assert missing["reachable"] and not missing["model_available"]
