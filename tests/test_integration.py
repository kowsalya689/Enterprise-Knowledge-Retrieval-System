"""Against REAL services (not mocks): a Qdrant server and a Redis server.

Run them locally, then::

    QDRANT_TEST_URL=http://localhost:6333 REDIS_TEST_URL=redis://localhost:6379/1 python -m pytest tests/test_integration.py -v

Skipped when the variables are not set.
"""
import os
import uuid

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from app.config import Settings
from app.ingestion.indexer import Indexer
from app.ingestion.metadata import load_inventory
from app.main import create_app
from app.retrieval.vector_store import QdrantStore
from tests.conftest import H, INVENTORY, RAW_DIR, build_services, make_settings
from tests.doubles import BagOfWordsEmbedder

pytestmark = [pytest.mark.integration,
              pytest.mark.skipif(not (os.getenv("QDRANT_TEST_URL") and os.getenv("REDIS_TEST_URL")),
                                 reason="set QDRANT_TEST_URL and REDIS_TEST_URL")]


@pytest.fixture(scope="module")
def real_store():
    name = f"it_{uuid.uuid4().hex[:8]}"
    cfg = Settings(_env_file=None, qdrant_url=os.environ["QDRANT_TEST_URL"], qdrant_collection=name,
                   embedding_dim=None, docx_page_mode="off")
    store = QdrantStore(cfg)
    Indexer(cfg, BagOfWordsEmbedder(), store, load_inventory(INVENTORY)).ingest_directory(RAW_DIR)
    yield store
    store.client.delete_collection(name)


def test_real_qdrant_stores_points_with_payload_and_indexes(real_store):
    info = real_store.collection_info()
    assert info["points_count"] == 163 and info["vector_size"] == 4096 and info["distance"] == "Cosine"
    assert info["indexed_payload_fields"]["access_tags"] == "keyword"
    assert {"document_id", "file_name", "file_type", "department"} <= set(info["indexed_payload_fields"])
    point = real_store.sample(limit=1)[0]
    assert point["vector_size"] == 4096 and {"text", "page_number", "access_tags", "department"} <= set(point["payload"])


def test_real_qdrant_search_applies_the_access_filter(real_store):
    emb = BagOfWordsEmbedder()
    hits = real_store.search(emb.embed_query("salary band grade G5 maximum"), ["all_employees"], 20)
    assert hits and all("all_employees" in h.payload["access_tags"] for h in hits)
    hr = real_store.search(emb.embed_query("salary band grade G5 maximum"), ["all_employees", "hr"], 3)
    assert hr[0].payload["document_id"] == "DOC-011"


def test_real_redis_and_qdrant_through_the_api(tmp_path, real_store):
    import redis
    rc = redis.Redis.from_url(os.environ["REDIS_TEST_URL"], decode_responses=True)
    rc.flushdb()
    svc = build_services(tmp_path, real_store, cache_client=rc)
    client = TestClient(create_app(svc.settings, svc), raise_server_exceptions=False)
    q = {"question": "What is the maximum number of consecutive days of casual leave?"}
    first = client.post("/query", headers=H["employee"], json=q).json()
    second = client.post("/query", headers=H["employee"], json=q).json()
    assert first["status"] == "answered" and not first["cached"] and second["cached"]
    assert any(k.startswith("t:v1:") for k in rc.keys("t:*"))
    assert client.get("/health").json()["components"]["redis_cache"]["status"] == "ok"
    rc.flushdb()
