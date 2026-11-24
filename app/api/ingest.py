
"""Document ingestion API."""
from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, UploadFile, status

from app.api.schemas import UploadResponse
from app.dependencies import (
    Services,
    chunking_info,
    embedding_info,
    get_principal,
    get_services,
)
from app.security.permissions import Principal

router = APIRouter(tags=["Ingest"])


@router.post(
    "/ingest",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest a document into Qdrant",
)
def ingest_document(
    file: UploadFile = File(..., description="PDF, DOCX, CSV, TXT, or supported document"),
    access_tags: str = Form(
        "all_employees",
        description="Comma-separated access tags",
    ),
    department: str = Form(
        "General",
        description="Owning department",
    ),
    principal: Principal = Depends(get_principal),
    services: Services = Depends(get_services),
):
    """Extract, chunk, embed, and store the uploaded document."""

    outcome = services.documents.upload(
        principal,
        file.filename or "",
        file.file,
        department,
        access_tags,
    )

    result = outcome.result
    record = outcome.record

    embedding = embedding_info(
        services.settings,
        services.embedder,
    )

    return UploadResponse(
        status=outcome.status,
        document_id=record.document_id,
        file_name=record.file_name,
        file_type=record.file_type,
        department=record.department,
        access_tags=list(record.access_tags),
        version=record.version,
        chunks=result.chunks,
        pages=result.pages,
        seconds=round(result.seconds, 2),
        message=(
            "Already indexed with identical content"
            if outcome.status == "unchanged"
            else "Document ingested successfully"
        ),
        embedding_model=embedding["model"],
        embedding_dimension=embedding["dimension"],
        chunking=ChunkingInfo(
            **chunking_info(services.settings)
        ),
        chunk_words_max=result.max_words or None,
        chunk_words_avg=(
            round(result.avg_words, 1)
            if result.avg_words
            else None
        ),
    )
