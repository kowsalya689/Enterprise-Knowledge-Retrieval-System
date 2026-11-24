"""Qdrant similarity search, collection facts and the one-model-per-collection rule."""
import numpy as np
import pytest
from qdrant_client import QdrantClient

from app.config import Settings
from app.retrieval.vector_store import EmbeddingModelMismatchError, QdrantStore
from tests.doubles import BagOfWordsEmbedder

ALL = ["all_employees", "hr", "finance", "procurement", "it", "security", "sales", "support", "management"]


def test_similarity_search_ranks_the_matching_chunk_first(kb):
    emb = BagOfWordsEmbedder()
    hits = kb.search(emb.embed_query("hotel limit per night Tier 1 cities grade G4 to G6"), ["all_employees"], 5)
    top = hits[0]
    assert top.payload["document_id"] == "DOC-003" and top.payload["section_path"] == "3. Eligibility by Grade"
    assert top.payload["page_number"] == 2 and top.score >= hits[-1].score
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_hit_carries_the_metadata_needed_for_citations(kb):
    hit = kb.search(BagOfWordsEmbedder().embed_query("casual leave maximum consecutive days"), ["all_employees"], 1)[0]
    for key in ("document_id", "file_name", "text", "section_path", "department", "access_tags", "chunk_id", "embedding_model"):
        assert key in hit.payload and hit.payload[key] not in (None, "")
    assert hit.payload["embedding_model"] == "bow-test-embedder"


def test_collection_info_reports_dimension_and_count(kb):
    info = kb.collection_info()
    assert info["points_count"] == 163 and info["vector_size"] == 4096 and info["distance"] == "Cosine"
    assert info["name"] == "api_test"


def test_every_stored_point_has_the_required_fields(kb):
    points = kb.sample(limit=200, with_vectors=True)
    assert len(points) == 163
    for p in points:
        pl = p["payload"]
        assert p["vector_size"] == 4096
        for key in ("document_id", "file_name", "text", "page_number", "department", "access_tags"):
            assert key in pl


def test_list_documents_respects_access_tags(kb):
    assert len(kb.list_documents()) == 11
    employee = kb.list_documents(["all_employees"])
    assert set(employee) == {"DOC-001", "DOC-002", "DOC-003", "DOC-006", "DOC-007"}
    assert kb.find_document_by_file_name("DOC-011_HR_Compensation_and_Salary_Bands.pdf")["document_id"] == "DOC-011"


def test_vectors_of_different_models_cannot_share_a_collection(memory_client):
    store = QdrantStore(Settings(_env_file=None, qdrant_collection="m", embedding_dim=None), client=memory_client)
    store.ensure_collection(512, "model-a")
    from app.retrieval.vector_store import ChunkPoint
    store.upsert([ChunkPoint("c", np.ones(512) / 22.6, {"document_id": "D", "document_version": "1", "chunk_index": 0,
                                                       "embedding_model": "model-a", "access_tags": ["x"]})])
    store.ensure_collection(512, "model-a")                       # same model: fine
    with pytest.raises(EmbeddingModelMismatchError):
        store.ensure_collection(512, "model-b")


def test_retrieval_evaluation_runs_and_reports_zero_leaks(kb, tmp_path):
    from app.evaluation.retrieval_eval import evaluate
    from tests.conftest import make_settings
    result = evaluate(make_settings(tmp_path), BagOfWordsEmbedder(), kb)
    assert result["access_control_leaks"] == 0 and result["answerable_questions"] == 55
    assert 0 <= result["recall_at_5"] <= 1 and result["mrr"] > 0.3        # a bag-of-words stand-in, not a real model
    assert result["top1_score"]["answerable_median"] > result["top1_score"]["unanswerable_median"]
