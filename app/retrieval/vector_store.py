"""Qdrant storage: one point per chunk (id + vector + payload).

Connection modes (set in ``.env``):

* **Qdrant Cloud** (nothing to install): ``QDRANT_URL`` + ``QDRANT_API_KEY``;
* **local disk** (no server, no account): ``QDRANT_LOCAL_PATH=./qdrant_data``.

Everything is created automatically: the collection, its vector size and the
payload indexes for the fields used in filters.  Point ids are deterministic
(UUID v5 of document id, version and chunk index), so re-ingesting a document
never creates duplicates.

Search always requires the caller's access tags and applies them **inside** the Qdrant
query, so chunks the user may not see are never returned (fail closed).

Check what is stored::

    python -m app.retrieval.vector_store --stats
    python -m app.retrieval.vector_store --document DOC-003 --sample 3
"""
from __future__ import annotations

import argparse
import logging
import uuid
import warnings
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
from qdrant_client import QdrantClient, models

from app.config import Settings, get_settings
from app.ingestion.models import IngestionError

logger = logging.getLogger(__name__)

_NAMESPACE = uuid.UUID("6f1c2e7a-9d0b-5c53-8a41-1f3b4c5d6e7f")

# Fields that queries filter on: these get a payload index.
PAYLOAD_INDEXES: dict[str, models.PayloadSchemaType] = {
    "access_tags": models.PayloadSchemaType.KEYWORD,
    "document_id": models.PayloadSchemaType.KEYWORD,
    "file_name": models.PayloadSchemaType.KEYWORD,
    "file_type": models.PayloadSchemaType.KEYWORD,
    "department": models.PayloadSchemaType.KEYWORD,
}


class VectorStoreError(IngestionError):
    """Connection or configuration problem with the vector database."""


class VectorSizeMismatchError(VectorStoreError):
    """The collection exists with a different vector size than the model produces."""


class EmbeddingModelMismatchError(VectorStoreError):
    """The collection holds vectors from another embedding model: never mix models."""


@dataclass
class Hit:
    """One search result."""
    score: float
    point_id: str
    payload: dict[str, Any]


@dataclass
class ChunkPoint:
    chunk_id: str
    vector: np.ndarray
    payload: dict[str, Any]

    @property
    def point_id(self) -> str:
        p = self.payload
        return point_id(p["document_id"], p["document_version"], p["chunk_index"])


def point_id(document_id: str, version: str, chunk_index: int) -> str:
    return str(uuid.uuid5(_NAMESPACE, f"{document_id}:{version}:{chunk_index}"))


def _doc_filter(document_id: str) -> models.Filter:
    return models.Filter(must=[models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id))])


def _tags_condition(access_tags: Sequence[str]) -> models.FieldCondition:
    tags = sorted(set(access_tags))
    if not tags:
        raise VectorStoreError("Refusing to query without access tags (fail closed)")
    return models.FieldCondition(key="access_tags", match=models.MatchAny(any=tags))


def _filter(access_tags: Optional[Sequence[str]] = None, document_id: Optional[str] = None,
            file_name: Optional[str] = None) -> Optional[models.Filter]:
    must = []
    if access_tags is not None:
        must.append(_tags_condition(access_tags))
    if document_id:
        must.append(models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id)))
    if file_name:
        must.append(models.FieldCondition(key="file_name", match=models.MatchValue(value=file_name)))
    return models.Filter(must=must) if must else None


def create_client(settings: Settings) -> QdrantClient:
    """Build a client from the settings (cloud URL, or local disk folder)."""
    if settings.qdrant_url:
        url = settings.qdrant_url.strip()
        if "://" not in url:
            url = "https://" + url
        key = settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None
        return QdrantClient(url=url, api_key=key, timeout=settings.qdrant_timeout)
    if settings.qdrant_local_path:
        path = settings.qdrant_local_path
        path.mkdir(parents=True, exist_ok=True)
        return QdrantClient(path=str(path))
    raise VectorStoreError(
        "No vector database configured. Set QDRANT_URL and QDRANT_API_KEY (Qdrant Cloud) "
        "or QDRANT_LOCAL_PATH (local folder) in the .env file."
    )


class QdrantStore:
    def __init__(self, settings: Optional[Settings] = None, client: Optional[QdrantClient] = None):
        self.settings = settings or get_settings()
        self.collection = self.settings.qdrant_collection
        self.client = client or create_client(self.settings)

    # --------------------------------------------------------------- setup
    def ensure_collection(self, vector_size: int, embedding_model: Optional[str] = None) -> None:
        """Create the collection and payload indexes if they do not exist (safe to repeat).

        Raises when the collection already holds vectors of another size or from another
        embedding model, because vectors of different models must never share a collection.
        """
        try:
            if not self.client.collection_exists(self.collection):
                logger.info("Creating collection '%s' (%d dimensions, cosine)", self.collection, vector_size)
                self.client.create_collection(
                    self.collection,
                    vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
                )
            else:
                existing = self.client.get_collection(self.collection).config.params.vectors
                size = getattr(existing, "size", None)
                if size is not None and size != vector_size:
                    raise VectorSizeMismatchError(
                        f"Collection '{self.collection}' has {size} dimensions but the embedding model produces "
                        f"{vector_size}. Use another QDRANT_COLLECTION name for this model."
                    )
            if embedding_model:
                self._check_model(embedding_model)
            with warnings.catch_warnings():              # local mode ignores payload indexes
                warnings.simplefilter("ignore")
                for field, schema in PAYLOAD_INDEXES.items():
                    self.client.create_payload_index(self.collection, field_name=field, field_schema=schema)
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError(f"Could not set up Qdrant collection '{self.collection}': {exc}") from exc

    def _check_model(self, embedding_model: str) -> None:
        points, _ = self.client.scroll(self.collection, limit=1, with_payload=["embedding_model"], with_vectors=False)
        stored = points[0].payload.get("embedding_model") if points else None
        if stored and stored != embedding_model:
            raise EmbeddingModelMismatchError(
                f"Collection '{self.collection}' was filled with model '{stored}', but the configured model is "
                f"'{embedding_model}'. Use another QDRANT_COLLECTION name or delete the collection."
            )

    # --------------------------------------------------------------- write
    def upsert(self, points: Sequence[ChunkPoint]) -> int:
        size = self.settings.qdrant_batch_size
        try:
            for i in range(0, len(points), size):
                batch = points[i:i + size]
                self.client.upsert(
                    self.collection,
                    points=[models.PointStruct(id=p.point_id, vector=p.vector.astype(float).tolist(), payload=p.payload)
                            for p in batch],
                    wait=True,
                )
        except Exception as exc:
            raise VectorStoreError(f"Upsert into '{self.collection}' failed: {exc}") from exc
        return len(points)

    def delete_document(self, document_id: str) -> None:
        self.client.delete(self.collection, points_selector=models.FilterSelector(filter=_doc_filter(document_id)), wait=True)

    # ---------------------------------------------------------------- read
    def count(self, document_id: Optional[str] = None) -> int:
        flt = _doc_filter(document_id) if document_id else None
        return self.client.count(self.collection, count_filter=flt, exact=True).count

    def fingerprint(self, document_id: str) -> Optional[str]:
        """file_sha256 stored for a document, or None when it is not indexed."""
        points, _ = self.client.scroll(self.collection, scroll_filter=_doc_filter(document_id), limit=1,
                                       with_payload=["file_sha256"], with_vectors=False)
        return points[0].payload.get("file_sha256") if points else None

    def list_documents(self, access_tags: Optional[Sequence[str]] = None) -> dict[str, dict]:
        """Chunks per document read from the stored payloads (optionally only what ``access_tags`` may see)."""
        docs: dict[str, dict] = {}
        offset = None
        while True:
            points, offset = self.client.scroll(
                self.collection, scroll_filter=_filter(access_tags), limit=256, offset=offset, with_vectors=False,
                with_payload=["document_id", "file_name", "file_type", "access_tags", "document_version",
                              "department", "document_title", "ingested_at", "embedding_model"])
            for p in points:
                pl = p.payload
                d = docs.setdefault(pl["document_id"], {
                    "chunks": 0, "file_name": pl.get("file_name"), "file_type": pl.get("file_type"),
                    "access_tags": pl.get("access_tags"), "version": pl.get("document_version"),
                    "department": pl.get("department"), "title": pl.get("document_title"),
                    "ingested_at": pl.get("ingested_at"), "embedding_model": pl.get("embedding_model")})
                d["chunks"] += 1
            if offset is None:
                return dict(sorted(docs.items()))

    def find_document_by_file_name(self, file_name: str) -> Optional[dict]:
        """(document_id, version, access_tags) of an indexed file with this name, if any."""
        points, _ = self.client.scroll(self.collection, scroll_filter=_filter(file_name=file_name), limit=1,
                                       with_payload=["document_id", "document_version", "access_tags"], with_vectors=False)
        if not points:
            return None
        pl = points[0].payload
        return {"document_id": pl["document_id"], "version": pl.get("document_version"), "access_tags": pl.get("access_tags", [])}

    def search(self, query_vector: np.ndarray, access_tags: Sequence[str], limit: int = 8) -> list[Hit]:
        """Vector similarity search restricted to chunks the caller may see.

        The access-tag filter is part of the Qdrant query itself; there is no way to
        call this method without tags.
        """
        flt = _filter(access_tags)
        try:
            result = self.client.query_points(self.collection, query=np.asarray(query_vector, dtype=float).tolist(),
                                              query_filter=flt, limit=limit, with_payload=True, with_vectors=False)
        except Exception as exc:
            raise VectorStoreError(f"Search in '{self.collection}' failed: {exc}") from exc
        return [Hit(score=float(p.score), point_id=str(p.id), payload=p.payload or {}) for p in result.points]

    def collection_info(self) -> dict:
        """Facts about the collection: size, vector configuration, indexed payload fields."""
        info = self.client.get_collection(self.collection)
        vectors = info.config.params.vectors
        schema = {k: getattr(v.data_type, "value", str(v.data_type)) for k, v in (info.payload_schema or {}).items()}
        return {
            "name": self.collection,
            "status": getattr(info.status, "value", str(info.status)),
            "points_count": self.count(),
            "vector_size": getattr(vectors, "size", None),
            "distance": getattr(getattr(vectors, "distance", None), "value", None),
            "indexed_payload_fields": schema,
        }

    def sample(self, document_id: Optional[str] = None, limit: int = 3, access_tags: Optional[Sequence[str]] = None,
               with_vectors: bool = True) -> list[dict]:
        """Stored points with their payload (and vector size) for inspection."""
        points, _ = self.client.scroll(self.collection, scroll_filter=_filter(access_tags, document_id),
                                       limit=limit, with_payload=True, with_vectors=with_vectors)
        return [{"id": str(p.id), "vector_size": len(p.vector) if p.vector is not None else None, "payload": p.payload}
                for p in points]


def _main() -> int:
    parser = argparse.ArgumentParser(description="Inspect the stored chunks")
    parser.add_argument("--stats", action="store_true", help="chunks per document")
    parser.add_argument("--document", help="document id, e.g. DOC-003")
    parser.add_argument("--sample", type=int, default=0, help="print N stored points (payload) for the document")
    args = parser.parse_args()
    store = QdrantStore()
    if not store.client.collection_exists(store.collection):
        print(f"Collection '{store.collection}' does not exist yet. Run the ingestion first.")
        return 1
    if args.stats or not (args.document or args.sample):
        docs = store.list_documents()
        print(f"Collection '{store.collection}': {store.count()} chunks in {len(docs)} documents\n")
        print(f"{'document':10s} {'chunks':>6s}  {'type':5s} {'tags':28s} file")
        for doc_id, d in docs.items():
            print(f"{doc_id:10s} {d['chunks']:6d}  {d['file_type']:5s} {','.join(d['access_tags'] or []):28s} {d['file_name']}")
    if args.document or args.sample:
        for item in store.sample(args.document, args.sample or 3):
            pl = item["payload"]
            print("\n" + "-" * 78)
            print(f"id {item['id']}   vector: {item['vector_size']} dimensions")
            for key in ("chunk_id", "file_name", "section_path", "page_number", "sheet_name", "access_tags", "word_count"):
                print(f"  {key:14s}: {pl.get(key)}")
            print(f"  text          : {pl.get('text', '')[:200].replace(chr(10), ' / ')}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
