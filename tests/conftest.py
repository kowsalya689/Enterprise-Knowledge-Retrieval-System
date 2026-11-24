"""Shared fixtures: dataset paths, a fake embedding model and an in-memory Qdrant."""
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import numpy as np
import pytest
from qdrant_client import QdrantClient

from app.config import PROJECT_ROOT, Settings
from app.ingestion.metadata import load_inventory

RAW_DIR = PROJECT_ROOT / "data" / "raw"
INVENTORY = PROJECT_ROOT / "data" / "processed" / "document_inventory.csv"
HAS_SOFFICE = bool(shutil.which("soffice") or shutil.which("libreoffice"))


class FakeEmbedder:
    """Deterministic stand-in for the real model (no download needed in tests).

    The vector is derived from the SHA-256 of the input, so the same text always
    gives the same vector and different texts give different vectors.
    """
    name = "fake-embedder"
    dimension = 32

    def __init__(self):
        self.calls: list[list[str]] = []

    def _vec(self, text: str) -> np.ndarray:
        seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")
        v = np.random.default_rng(seed).normal(size=self.dimension).astype(np.float32)
        return v / np.linalg.norm(v)

    def embed_documents(self, texts):
        self.calls.append(list(texts))
        return np.vstack([self._vec(t) for t in texts])

    def embed_query(self, text):
        return self._vec(text)


@pytest.fixture(scope="session")
def settings() -> Settings:
    return Settings(_env_file=None, qdrant_collection="test_kb", embedding_dim=None)


@pytest.fixture(scope="session")
def inventory():
    return load_inventory(INVENTORY)


@pytest.fixture()
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture()
def memory_client():
    client = QdrantClient(":memory:")
    yield client
    client.close()


@pytest.fixture(scope="session")
def raw_dir() -> Path:
    return RAW_DIR


# ----------------------------------------------------------------------------- API-level fixtures
from fastapi.testclient import TestClient  # noqa: E402
from pydantic import SecretStr  # noqa: E402

from app.cache.redis_cache import RedisAnswerCache  # noqa: E402
from app.dependencies import assemble_services  # noqa: E402
from app.ingestion.indexer import Indexer  # noqa: E402
from app.main import create_app  # noqa: E402
from app.rag.prompts import PROMPT_VERSION  # noqa: E402
from app.retrieval.vector_store import QdrantStore  # noqa: E402
from app.security.permissions import Authenticator, load_roles, parse_api_keys  # noqa: E402
from tests.doubles import BagOfWordsEmbedder, FakeLLM  # noqa: E402

KEYS = "k-emp:employee,k-hr:hr,k-fin:finance,k-mgr:manager,k-it:it,k-sales:sales"
H = {role: {"X-API-Key": key} for key, role in (p.split(":") for p in KEYS.split(","))}


def make_settings(tmp_path, **kw) -> Settings:
    base = dict(_env_file=None, qdrant_collection="api_test", embedding_dim=None, docx_page_mode="off",
                relevance_threshold=0.15, api_keys=SecretStr(KEYS), upload_dir=tmp_path / "uploads",
                cache_key_prefix="t")
    base.update(kw)
    return Settings(**base)


@pytest.fixture(scope="session")
def kb():
    """All 11 dataset documents indexed once (bag-of-words vectors, in-memory Qdrant)."""
    client = QdrantClient(":memory:")
    cfg = Settings(_env_file=None, qdrant_collection="api_test", embedding_dim=None, docx_page_mode="off")
    store = QdrantStore(cfg, client=client)
    indexer = Indexer(cfg, BagOfWordsEmbedder(), store, load_inventory(INVENTORY))
    results = indexer.ingest_directory(RAW_DIR)
    assert all(r.status == "indexed" for r in results)
    yield store
    client.close()


def build_services(tmp_path, store, llm=None, cache_client=None, **kw):
    import fakeredis
    cfg = make_settings(tmp_path, **kw)
    cache = RedisAnswerCache(cache_client or fakeredis.FakeRedis(decode_responses=True), "t", PROMPT_VERSION, "test-model")
    auth = Authenticator(load_roles(cfg.roles_path), parse_api_keys(KEYS))
    return assemble_services(cfg, embedder=BagOfWordsEmbedder(), store=store, cache=cache, llm=llm or FakeLLM(), authenticator=auth)


@pytest.fixture()
def services(tmp_path, kb):
    return build_services(tmp_path, kb)


@pytest.fixture()
def client(services):
    return TestClient(create_app(services.settings, services), raise_server_exceptions=False)


@pytest.fixture()
def empty_client(tmp_path):
    """An application with an empty Qdrant (nothing uploaded yet)."""
    c = QdrantClient(":memory:")
    cfg = make_settings(tmp_path)
    store = QdrantStore(cfg, client=c)
    svc = build_services(tmp_path, store)
    yield TestClient(create_app(svc.settings, svc), raise_server_exceptions=False), svc
    c.close()
