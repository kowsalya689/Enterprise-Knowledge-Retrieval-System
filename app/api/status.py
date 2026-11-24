"""Health and Qdrant collection information."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse

from app.api.schemas import (
    ChunkingInfo, CollectionInfoResponse, ComponentHealth, DocumentInfo, EmbeddingInfo, ErrorResponse, HealthResponse,
    PipelineInfoResponse, StoredPoint,
)
from app.dependencies import Services, chunking_info, embedding_info, get_principal, get_services
from app.security.permissions import Principal

router = APIRouter()


def _doc_info(doc_id: str, d: dict) -> DocumentInfo:
    return DocumentInfo(document_id=doc_id, file_name=d["file_name"], title=d.get("title"), file_type=d.get("file_type"),
                        department=d.get("department"), access_tags=d.get("access_tags") or [], version=d.get("version"),
                        chunks=d["chunks"], ingested_at=d.get("ingested_at"))


@router.get("/health", response_model=HealthResponse, tags=["System"], summary="Check application health",
            responses={503: {"model": HealthResponse, "description": "Qdrant is not reachable"}})
def health(services: Services = Depends(get_services)):
    c: dict[str, ComponentHealth] = {"api": ComponentHealth(status="ok", detail="running")}
    try:
        c["embedding_model"] = ComponentHealth(status="ok", detail=f"{services.embedder.name}, {services.embedder.dimension} dimensions")
    except Exception as exc:
        c["embedding_model"] = ComponentHealth(status="unavailable", detail=str(exc))
    try:
        store = services.store
        if store.client.collection_exists(store.collection):
            c["qdrant"] = ComponentHealth(status="ok", detail=f"collection '{store.collection}': {store.count()} points")
        else:
            c["qdrant"] = ComponentHealth(status="degraded", detail=f"collection '{store.collection}' does not exist yet; upload a document")
    except Exception as exc:
        c["qdrant"] = ComponentHealth(status="unavailable", detail=f"Qdrant not reachable: {exc}")
    if not services.cache.enabled:
        c["redis_cache"] = ComponentHealth(status="disabled", detail="cache switched off or REDIS_URL not set")
    elif services.cache.ping():
        c["redis_cache"] = ComponentHealth(status="ok", detail="connected")
    else:
        c["redis_cache"] = ComponentHealth(status="degraded", detail="Redis not reachable; answers are not cached")
    llm = services.llm.health()
    if llm.get("reachable") and llm.get("model_available"):
        c["ollama"] = ComponentHealth(status="ok", detail=f"model {llm['model']} available")
    elif llm.get("reachable"):
        c["ollama"] = ComponentHealth(status="degraded", detail=f"model {llm['model']} is not installed (ollama pull {llm['model']})")
    else:
        c["ollama"] = ComponentHealth(status="degraded", detail=f"Ollama not reachable: {llm.get('detail', '')}")

    statuses = {x.status for x in c.values()}
    overall = "unavailable" if c["qdrant"].status == "unavailable" or c["embedding_model"].status == "unavailable" \
        else "degraded" if "degraded" in statuses else "ok"
    body = HealthResponse(status=overall, components=c)
    return JSONResponse(status_code=503 if overall == "unavailable" else 200, content=body.model_dump())


@router.get("/collection/info", response_model=CollectionInfoResponse, tags=["System"],
            summary="Show the Qdrant collection and how documents are stored",
            responses={401: {"model": ErrorResponse}, 503: {"model": ErrorResponse}})
def collection_info(sample: int = Query(0, ge=0, le=10, description="Also return this many stored points (payload and vector size)"),
                    principal: Principal = Depends(get_principal), services: Services = Depends(get_services)):
    store = services.store
    info = store.collection_info()
    visible = store.list_documents(sorted(principal.access_tags))
    models_seen = {d.get("embedding_model") for d in visible.values() if d.get("embedding_model")}
    points: Optional[list[StoredPoint]] = None
    if sample:
        points = [StoredPoint(**p) for p in store.sample(limit=sample, access_tags=sorted(principal.access_tags))]
    return CollectionInfoResponse(
        **info, embedding_model=", ".join(sorted(models_seen)) or None,
        documents_visible_to_you=[_doc_info(k, v) for k, v in visible.items()], sample_points=points)


@router.get("/pipeline/info", response_model=PipelineInfoResponse, tags=["System"],
            summary="How documents are embedded and chunked, and the main query settings")
def pipeline_info(services: Services = Depends(get_services)):
    """Answers: which embedding model turns text into vectors, how documents are cut into chunks,
    and which settings decide whether a question is answered. No key is needed: it shows settings, not data."""
    st = services.settings
    return PipelineInfoResponse(
        embedding=EmbeddingInfo(**embedding_info(st, services.embedder)), chunking=ChunkingInfo(**chunking_info(st)),
        retrieval_top_k=st.retrieval_top_k, relevance_threshold=st.relevance_threshold, llm_model=st.ollama_model,
        answer_cache_ttl_seconds=st.cache_ttl_answer_seconds, semantic_cache_threshold=st.semantic_cache_threshold,
        web_ui_role=st.web_ui_role or "(web pages off)")
