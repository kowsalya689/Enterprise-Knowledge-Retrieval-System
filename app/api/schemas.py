"""Pydantic request and response models (shown in Swagger UI at /docs)."""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator


# ------------------------------------------------------------------ errors
class ErrorBody(BaseModel):
    code: str = Field(examples=["permission_denied"])
    message: str
    details: Optional[list[Any]] = None


class ErrorResponse(BaseModel):
    error: ErrorBody


# ------------------------------------------------------------------ health
class ComponentHealth(BaseModel):
    status: Literal["ok", "degraded", "unavailable", "disabled"]
    detail: str = ""


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "unavailable"]
    components: dict[str, ComponentHealth]


# ------------------------------------------------------------------ documents
class ChunkingInfo(BaseModel):
    strategy: str
    max_words_per_chunk: int
    overlap_words: int
    table_rows_per_chunk: int


class EmbeddingInfo(BaseModel):
    model: str
    dimension: Optional[int] = None
    document_prefix: str
    query_prefix: str


class PipelineInfoResponse(BaseModel):
    embedding: EmbeddingInfo
    chunking: ChunkingInfo
    retrieval_top_k: int
    relevance_threshold: float
    llm_model: str
    answer_cache_ttl_seconds: int
    semantic_cache_threshold: float
    web_ui_role: str


class UploadResponse(BaseModel):
    status: Literal["indexed", "unchanged"]
    document_id: str
    file_name: str
    file_type: str
    department: str
    access_tags: list[str]
    version: str
    chunks: int
    pages: Optional[int] = None
    seconds: float
    message: str = ""
    embedding_model: Optional[str] = None
    embedding_dimension: Optional[int] = None
    chunking: Optional[ChunkingInfo] = None
    chunk_words_max: Optional[int] = None
    chunk_words_avg: Optional[float] = None


class DocumentInfo(BaseModel):
    document_id: str
    file_name: str
    title: Optional[str] = None
    file_type: Optional[str] = None
    department: Optional[str] = None
    access_tags: list[str]
    version: Optional[str] = None
    chunks: int
    ingested_at: Optional[str] = None


class DocumentListResponse(BaseModel):
    count: int
    documents: list[DocumentInfo]


# ------------------------------------------------------------------ query
class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000, examples=["What is the hotel limit per night for a G5 manager in Chennai?"])
    top_k: Optional[int] = Field(None, ge=1, le=20, description="Chunks to retrieve (default from settings)")
    debug: bool = Field(False, description="Return the retrieval trace (scores, kept/dropped chunks). Skips the cache lookup.")

    @field_validator("question")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 3:
            raise ValueError("question is too short")
        return v


class CitationOut(BaseModel):
    number: int
    document_id: str
    document_name: str
    document_title: str = ""
    section: str
    page: Optional[int] = None
    sheet: Optional[str] = None
    chunk_id: str
    score: float
    snippet: str


class RetrievedChunkOut(BaseModel):
    rank: int
    score: float
    kept: bool = Field(description="False = below the relevance cutoff")
    chunk_id: str
    document_id: str
    document_name: str
    section: str
    page: Optional[int] = None


class QueryResponse(BaseModel):
    status: Literal["answered", "not_found"]
    answer: str
    citations: list[CitationOut] = []
    citation_source: Optional[str] = Field(None, description="model_markers = cited by the model with [n]; retrieved_context = all context blocks")
    reason: Optional[str] = Field(None, description="Why the answer is 'not_found'")
    cached: bool
    cache_match: Optional[Literal["exact", "semantic"]] = Field(None, description="How the cached answer was found")
    cache_similarity: Optional[float] = Field(None, description="Question-to-question cosine similarity (semantic hits only)")
    top_score: Optional[float] = Field(None, description="Best retrieval score for the question (cutoff is RELEVANCE_THRESHOLD)")
    access_scope: str
    timings_ms: dict[str, float] = {}
    retrieved: Optional[list[RetrievedChunkOut]] = None


# ------------------------------------------------------------------ collection
class StoredPoint(BaseModel):
    id: str
    vector_size: Optional[int] = None
    payload: dict[str, Any]


class CollectionInfoResponse(BaseModel):
    name: str
    status: str
    points_count: int
    vector_size: Optional[int]
    distance: Optional[str]
    embedding_model: Optional[str] = None
    indexed_payload_fields: dict[str, str]
    documents_visible_to_you: list[DocumentInfo]
    sample_points: Optional[list[StoredPoint]] = None
