"""Document upload and listing."""
from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, UploadFile, status

from app.api.schemas import ChunkingInfo, DocumentInfo, DocumentListResponse, ErrorResponse, UploadResponse
from app.dependencies import Services, chunking_info, embedding_info, get_principal, get_services
from app.security.permissions import Principal

router = APIRouter(prefix="/documents", tags=["Documents"])

_ERRORS = {401: {"model": ErrorResponse}, 403: {"model": ErrorResponse}, 413: {"model": ErrorResponse},
           415: {"model": ErrorResponse}, 422: {"model": ErrorResponse}}


@router.post("/upload", response_model=UploadResponse, status_code=status.HTTP_201_CREATED, responses=_ERRORS,
             summary="Upload a document (PDF, DOCX, XLSX, XLS, HTML, CSV, TXT or Markdown)")
def upload_document(
    file: UploadFile = File(..., description="The document (max 50 MB, not empty)"),
    access_tags: str = Form(..., description="Comma-separated access tags, e.g. 'hr,management'. You can only use tags you hold.",
                            examples=["all_employees"]),
    department: str = Form("General", description="Owning department"),
    principal: Principal = Depends(get_principal),
    services: Services = Depends(get_services),
):
    """Validates the file, extracts text and tables, chunks it, embeds the chunks and stores them in Qdrant.

    Uploading a file with the same name replaces the old version (version number +1) and
    removes cached answers that used it.

    The response tells you **how** the file was processed: the embedding model and vector size, the chunking
    settings, how many chunks were made and how long the longest and the average chunk are (in words).
    """
    outcome = services.documents.upload(principal, file.filename or "", file.file, department, access_tags)
    r, rec = outcome.result, outcome.record
    return UploadResponse(
        status=outcome.status, document_id=rec.document_id, file_name=rec.file_name, file_type=rec.file_type,
        department=rec.department, access_tags=list(rec.access_tags), version=rec.version, chunks=r.chunks,
        pages=r.pages, seconds=round(r.seconds, 2),
        message="Already indexed with identical content" if outcome.status == "unchanged" else "Document indexed",
        embedding_model=embedding_info(services.settings, services.embedder)["model"],
        embedding_dimension=embedding_info(services.settings, services.embedder)["dimension"],
        chunking=ChunkingInfo(**chunking_info(services.settings)),
        chunk_words_max=r.max_words or None, chunk_words_avg=round(r.avg_words, 1) if r.avg_words else None)


@router.get("", response_model=DocumentListResponse, responses={401: {"model": ErrorResponse}},
            summary="List the documents you are allowed to see")
def list_documents(principal: Principal = Depends(get_principal), services: Services = Depends(get_services)):
    store = services.store
    docs = store.list_documents(sorted(principal.access_tags)) if store.client.collection_exists(store.collection) else {}
    items = [DocumentInfo(document_id=k, file_name=v["file_name"], title=v.get("title"), file_type=v.get("file_type"),
                          department=v.get("department"), access_tags=v.get("access_tags") or [], version=v.get("version"),
                          chunks=v["chunks"], ingested_at=v.get("ingested_at")) for k, v in docs.items()]
    return DocumentListResponse(count=len(items), documents=items)
