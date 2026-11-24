"""Embeddings wrapper, Qdrant storage and the full pipeline (fake model, in-memory Qdrant)."""
from __future__ import annotations

import numpy as np
import pytest

from app.config import Settings
from app.ingestion.indexer import Indexer
from app.ingestion.metadata import load_inventory
from app.ingestion.models import Chunk
from app.retrieval.embeddings import SentenceTransformerEmbedder, chunk_embedding_text
from app.retrieval.vector_store import (
    ChunkPoint, QdrantStore, VectorSizeMismatchError, VectorStoreError, create_client, point_id,
)
from tests.conftest import HAS_SOFFICE, INVENTORY, RAW_DIR, FakeEmbedder


# ------------------------------------------------------------------ embeddings wrapper
class _RecordingModel:
    """Looks like a SentenceTransformer: records the inputs, returns 6 numbers per text."""
    max_seq_length = 512

    def __init__(self):
        self.seen: list[str] = []

    def get_sentence_embedding_dimension(self):
        return 6

    def encode(self, texts, **kwargs):
        self.seen += list(texts)
        return np.array([[len(t), 1, 2, 3, 4, 5] for t in texts], dtype=np.float32)


def _settings(**kw) -> Settings:
    return Settings(_env_file=None, **kw)


def test_document_prefix_is_added_and_vectors_are_unit_length():
    model = _RecordingModel()
    emb = SentenceTransformerEmbedder(_settings(embedding_dim=None, embedding_batch_size=2), model=model)
    vectors = emb.embed_documents(["alpha", "beta", "gamma"])
    assert model.seen == ["search_document: alpha", "search_document: beta", "search_document: gamma"]
    assert vectors.shape == (3, 6)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


def test_query_prefix_and_no_prefix_for_other_models():
    model = _RecordingModel()
    SentenceTransformerEmbedder(_settings(embedding_dim=None), model=model).embed_query("hello")
    assert model.seen == ["search_query: hello"]
    model2 = _RecordingModel()
    SentenceTransformerEmbedder(_settings(embedding_dim=None, embedding_doc_prefix=""), model=model2).embed_documents(["x"])
    assert model2.seen == ["x"]


def test_dimension_truncation_matryoshka():
    model = _RecordingModel()
    emb = SentenceTransformerEmbedder(_settings(embedding_dim=4), model=model)
    vectors = emb.embed_documents(["abc"])
    assert emb.dimension == 4 and vectors.shape == (1, 4)
    assert np.isclose(np.linalg.norm(vectors[0]), 1.0, atol=1e-5)


def test_embedding_text_has_title_path_first():
    c = Chunk("D-C001", 0, "body text", "Doc > 1. Intro", "1. Intro", "text", 2)
    assert chunk_embedding_text(c) == "Doc > 1. Intro\nbody text"


# ------------------------------------------------------------------ Qdrant store
def _payload(doc="DOC-X", version="1.0", index=0, tags=("hr",)):
    return {"chunk_id": f"{doc}-C{index + 1:03d}", "chunk_index": index, "document_id": doc, "document_version": version,
            "file_name": f"{doc}.pdf", "file_type": "pdf", "access_tags": list(tags), "section_path": "1. A",
            "page_number": 1, "file_sha256": "abc", "text": "t"}


def _points(n, doc="DOC-X", dim=32, **kw):
    emb = FakeEmbedder()
    vecs = emb.embed_documents([f"{doc}-{i}" for i in range(n)])
    return [ChunkPoint(f"c{i}", vecs[i], _payload(doc, index=i, **kw)) for i in range(n)]


@pytest.fixture()
def store(memory_client):
    s = QdrantStore(Settings(_env_file=None, qdrant_collection="t", embedding_dim=None), client=memory_client)
    s.ensure_collection(32)
    return s


def test_ensure_collection_is_idempotent_and_checks_vector_size(store):
    store.ensure_collection(32)                              # second call: nothing breaks
    with pytest.raises(VectorSizeMismatchError):
        store.ensure_collection(768)


def test_upsert_is_idempotent_and_ids_are_deterministic(store):
    store.upsert(_points(5))
    store.upsert(_points(5))                                 # same ids -> still 5 points
    assert store.count() == 5
    assert point_id("DOC-X", "1.0", 2) == point_id("DOC-X", "1.0", 2) != point_id("DOC-X", "1.1", 2)


def test_payload_and_vector_are_stored_on_every_point(store):
    store.upsert(_points(3, tags=("hr", "management")))
    for item in store.sample(limit=3):
        assert item["vector_size"] == 32
        pl = item["payload"]
        assert pl["file_name"] and pl["section_path"] and pl["page_number"] == 1
        assert pl["access_tags"] == ["hr", "management"]


def test_delete_document_and_list_documents(store):
    store.upsert(_points(4, "DOC-A"))
    store.upsert(_points(2, "DOC-B"))
    assert store.list_documents()["DOC-A"]["chunks"] == 4
    store.delete_document("DOC-A")
    assert store.count("DOC-A") == 0 and store.count("DOC-B") == 2
    assert store.fingerprint("DOC-B") == "abc" and store.fingerprint("DOC-A") is None


def test_client_needs_configuration():
    with pytest.raises(VectorStoreError):
        create_client(Settings(_env_file=None, qdrant_url=None, qdrant_local_path=None))


def test_local_disk_mode_persists(tmp_path):
    cfg = Settings(_env_file=None, qdrant_local_path=tmp_path / "q", qdrant_collection="t", embedding_dim=None)
    s1 = QdrantStore(cfg); s1.ensure_collection(32); s1.upsert(_points(3)); s1.client.close()
    s2 = QdrantStore(cfg)
    assert s2.count() == 3


# ------------------------------------------------------------------ whole pipeline on the dataset
@pytest.fixture()
def indexer(memory_client, fake_embedder):
    cfg = Settings(_env_file=None, qdrant_collection="e2e", embedding_dim=None)
    store = QdrantStore(cfg, client=memory_client)
    return Indexer(cfg, fake_embedder, store, load_inventory(INVENTORY))


def test_ingest_all_documents_then_skip_unchanged_then_force(indexer):
    results = indexer.ingest_directory(RAW_DIR)
    assert len(results) == 11 and all(r.status == "indexed" for r in results), [(r.file_name, r.status, r.message) for r in results]
    total = sum(r.chunks for r in results)
    assert indexer.store.count() == total == 163
    docs = indexer.store.list_documents()
    assert len(docs) == 11 and docs["DOC-011"]["access_tags"] == ["hr", "management"]

    again = indexer.ingest_directory(RAW_DIR)
    assert all(r.status == "unchanged" for r in again) and indexer.store.count() == total

    forced = indexer.ingest_directory(RAW_DIR, force=True)
    assert all(r.status == "indexed" for r in forced) and indexer.store.count() == total   # replaced, not duplicated


def test_every_stored_chunk_has_required_metadata(indexer):
    indexer.ingest_directory(RAW_DIR)
    points, _ = indexer.store.client.scroll("e2e", limit=1000, with_payload=True, with_vectors=True)
    assert len(points) == 163
    for p in points:
        pl = p.payload
        assert pl["file_name"] and pl["section_path"] and pl["access_tags"] and pl["chunk_id"].startswith(pl["document_id"])
        assert len(p.vector) == 32
        if pl["file_type"] == "pdf" or (pl["file_type"] == "docx" and HAS_SOFFICE):
            assert pl["page_number"]
        elif pl["file_type"] not in ("pdf", "docx"):
            assert pl["page_number"] is None


def test_file_not_in_inventory_is_rejected_and_nothing_stored(indexer, tmp_path):
    f = tmp_path / "secret.txt"; f.write_text("1. Title\n\nsome text")
    r = indexer.ingest_file(f)
    assert r.status == "rejected" and "inventory" in r.message
    exists = indexer.store.client.collection_exists("e2e")
    assert (not exists) or indexer.store.count() == 0


def test_empty_and_unsupported_files_are_rejected(indexer, tmp_path):
    e = tmp_path / "e.pdf"; e.write_bytes(b"")
    x = tmp_path / "x.exe"; x.write_bytes(b"MZ")
    assert indexer.ingest_file(e).status == "rejected"
    assert indexer.ingest_file(x).status == "rejected"


def test_changed_document_replaces_old_chunks(indexer, tmp_path):
    inv = load_inventory(INVENTORY)
    name = "DOC-007_Support_Escalation_SOP.txt"
    work = tmp_path / name
    work.write_text((RAW_DIR / name).read_text(encoding="utf-8"), encoding="utf-8")
    indexer.inventory = inv
    first = indexer.ingest_file(work)
    n1 = indexer.store.count("DOC-007")
    work.write_text(work.read_text(encoding="utf-8") + "\n\n9. EXTRA SECTION\n-----------------\n\nA brand new paragraph.\n", encoding="utf-8")
    second = indexer.ingest_file(work)
    assert first.status == second.status == "indexed"
    assert indexer.store.count("DOC-007") == n1 + 1


def test_dry_run_needs_no_model_and_no_database():
    cfg = Settings(_env_file=None)
    r = Indexer(cfg, None, None, load_inventory(INVENTORY)).ingest_file(RAW_DIR / "DOC-005_IT_Incident_Response_SOP.md")
    assert r.status == "dry_run" and r.chunks == 19 and r.payloads[0]["access_tags"] == ["it", "security", "management"]
