"""FastAPI application: REST API over the RAG pipeline.

Run:  uvicorn app.main:app --reload
      http://localhost:8000/chat (ask)   /upload (upload)   /docs (Swagger)
"""
from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager
from typing import Optional
from app.api import ingest

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api import pages, query, status, upload
from app.config import Settings, get_settings
from app.dependencies import Services, build_services
from app.ingestion.models import (
    EmptyFileError, ExtractionError, FileTooLargeError, IngestionError, MissingAccessTagsError, UnsupportedFileError,
)
from app.rag.llm import LLMUnavailableError
from app.retrieval.vector_store import VectorStoreError
from app.security.permissions import AuthenticationError, PermissionDeniedError

logger = logging.getLogger("app")

DESCRIPTION = """
Ask questions about company documents. Answers come **only** from documents you are allowed to see and always
carry the document, section and page. If the documents do not contain the answer, you get
*Not found in the available documents*.

**How to try it:** click **Authorize**, enter your API key (the role behind the key decides what you can see),
upload a document, then ask a question with `POST /query`. Use `debug: true` to see the retrieval trace.
"""

TAGS = [
    {"name": "System", "description": "Health and Qdrant collection information"},
    {"name": "Documents", "description": "Upload and list documents"},
    {"name": "Query", "description": "Retrieval-augmented question answering"},
]

# exception -> (HTTP status, error code); the most specific class wins
_ERRORS: list[tuple[type[Exception], int, str]] = [
    (AuthenticationError, 401, "authentication_failed"),
    (PermissionDeniedError, 403, "permission_denied"),
    (UnsupportedFileError, 415, "unsupported_file_type"),
    (FileTooLargeError, 413, "file_too_large"),
    (EmptyFileError, 422, "empty_file"),
    (MissingAccessTagsError, 422, "invalid_access_tags"),
    (ExtractionError, 422, "extraction_failed"),
    (VectorStoreError, 503, "vector_store_error"),
    (LLMUnavailableError, 503, "llm_unavailable"),
    (IngestionError, 422, "ingestion_error"),
]


def _envelope(status_code: int, code: str, message: str, details: Optional[list] = None) -> JSONResponse:
    body = {"error": {"code": code, "message": message}}
    if details:
        body["error"]["details"] = details
    return JSONResponse(status_code=status_code, content=body)


def _warm_up(services: Services) -> None:
    """Run once in the background after start-up: the first embedding and the first Ollama call are slow
    (model weights are paged in), so do them now instead of during the first question."""
    try:
        services.embedder.embed_query("warm up")
        warm = getattr(services.llm, "warm_up", None)
        if warm:
            warm()
        logger.info("Warm-up finished")
    except Exception:                                # never stop the app because of the warm-up
        logger.warning("Warm-up failed (the first question will be slower)", exc_info=True)


def create_app(settings: Optional[Settings] = None, services: Optional[Services] = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if not hasattr(app.state, "services"):
            app.state.services = build_services(settings)
            logger.info("Services ready")
            threading.Thread(target=_warm_up, args=(app.state.services,), daemon=True).start()
        yield

    app = FastAPI(title="Enterprise Knowledge Retrieval API", version="1.0.0", description=DESCRIPTION,
                  openapi_tags=TAGS, lifespan=lifespan)
    if services is not None:                       # tests and demos inject ready-made services
        app.state.services = services

    for exc_class, code_status, code in _ERRORS:
        def handler(request: Request, exc: Exception, _s=code_status, _c=code):
            return _envelope(_s, _c, str(exc))
        app.add_exception_handler(exc_class, handler)

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError):
        details = [{"field": ".".join(str(p) for p in e["loc"] if p != "body"), "problem": e["msg"]} for e in exc.errors()]
        return _envelope(422, "validation_error", "The request is not valid", details)

    @app.exception_handler(Exception)
    async def unexpected_handler(request: Request, exc: Exception):
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return _envelope(500, "internal_error", "Unexpected server error. See the server log.")

    app.mount("/static", StaticFiles(directory=pages.WEB_DIR / "static"), name="static")
    app.include_router(pages.router)
    app.include_router(status.router)
    app.include_router(upload.router)
    app.include_router(query.router)
    app.include_router(ingest.router)
    return app


app = create_app()
