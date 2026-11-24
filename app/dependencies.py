"""Shared objects (services) and FastAPI dependencies."""
from __future__ import annotations

import logging
from dataclasses import dataclass

from fastapi import Depends, Request, Security
from fastapi.security import APIKeyHeader

from app.cache.redis_cache import Cache, build_cache
from app.config import Settings
from app.ingestion.indexer import Indexer
from app.rag.llm import LLMClient, OllamaClient
from app.rag.pipeline import RagPipeline
from app.rag.prompts import PROMPT_VERSION
from app.retrieval.embeddings import SentenceTransformerEmbedder, ThreadSafeEmbedder
from app.retrieval.vector_store import QdrantStore
from app.security.permissions import Authenticator, Principal
from app.services.documents import DocumentService

logger = logging.getLogger(__name__)


@dataclass
class Services:
    settings: Settings
    authenticator: Authenticator
    embedder: object
    store: QdrantStore
    cache: Cache
    llm: LLMClient
    pipeline: RagPipeline
    documents: DocumentService


def assemble_services(settings: Settings, *, embedder, store: QdrantStore, cache: Cache, llm: LLMClient,
                      authenticator: Authenticator) -> Services:
    """Wire the parts together (tests pass fakes; production passes the real ones)."""
    indexer = Indexer(settings, embedder, store, inventory={})      # uploads bring their own metadata
    return Services(
        settings=settings, authenticator=authenticator, embedder=embedder, store=store, cache=cache, llm=llm,
        pipeline=RagPipeline(settings, embedder, store, cache, llm),
        documents=DocumentService(settings, indexer, store, cache, authenticator.known_tags),
    )


def build_services(settings: Settings) -> Services:
    """Create the real services at start-up and check that the collection fits the model."""
    embedder = ThreadSafeEmbedder(SentenceTransformerEmbedder(settings))
    logger.info("Loading embedding model %s ...", embedder.name)
    dimension = embedder.dimension                                  # loads the model now, not on the first request
    store = QdrantStore(settings)
    store.ensure_collection(dimension, embedder.name)               # fails if the model or size differs
    llm = OllamaClient(settings)
    model_version = f"{settings.ollama_model}|{settings.embedding_model}|{dimension}|{settings.chunk_max_words}"
    cache = build_cache(settings, PROMPT_VERSION, model_version)
    return assemble_services(settings, embedder=embedder, store=store, cache=cache, llm=llm,
                             authenticator=Authenticator.from_settings(settings))


def chunking_info(settings: Settings) -> dict:
    return dict(strategy="structure-aware: one chunk per heading section (title path first), tables kept whole, "
                         "spreadsheets in row groups with column names; overlap only when a section must be split",
                max_words_per_chunk=settings.chunk_max_words, overlap_words=settings.chunk_overlap_words,
                table_rows_per_chunk=settings.table_rows_per_chunk)


def embedding_info(settings: Settings, embedder) -> dict:
    try:
        dim = int(embedder.dimension)
    except Exception:
        dim = settings.embedding_dim
    return dict(model=getattr(embedder, "name", settings.embedding_model), dimension=dim,
                document_prefix=settings.embedding_doc_prefix, query_prefix=settings.embedding_query_prefix)


# ------------------------------------------------------------------ FastAPI dependencies
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False,
                              description="Your API key. It decides your role and therefore which documents you may see.")


def get_services(request: Request) -> Services:
    return request.app.state.services


def get_principal(request: Request, api_key: str | None = Security(api_key_header),
                  services: Services = Depends(get_services)) -> Principal:
    """A request with a key gets that key's role. The two web pages send no key but the header
    ``X-Web-UI: 1`` and get the server-side ``WEB_UI_ROLE`` (never a role chosen by the browser).
    The custom header makes a browser block calls from other websites (they would need CORS permission,
    which this app never grants)."""
    if not api_key and request.headers.get("x-web-ui") == "1" and services.settings.web_ui_role:
        return services.authenticator.principal_for_role(services.settings.web_ui_role)
    return services.authenticator.authenticate(api_key)
