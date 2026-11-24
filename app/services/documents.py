"""Upload workflow: validate, save, index and invalidate the cache.

The heavy lifting (extraction, chunking, embeddings, Qdrant) is done by the existing
``Indexer``; this service only adds the rules that belong to an upload:
safe file names, size limit while streaming, tag permissions, versioning of a
re-uploaded file and cache invalidation.
"""
from __future__ import annotations

import hashlib
import os
import re
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import BinaryIO, Optional

from app.cache.redis_cache import Cache
from app.config import Settings
from app.ingestion.extractors import SUPPORTED_EXTENSIONS
from app.ingestion.indexer import IngestResult, Indexer
from app.ingestion.metadata import DocumentRecord, parse_access_tags
from app.ingestion.models import (
    EmptyFileError, ExtractionError, FileTooLargeError, IngestionError, UnsupportedFileError,
)
from app.retrieval.vector_store import QdrantStore
from app.security.permissions import PermissionDeniedError, Principal

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_filename(name: str) -> str:
    """Keep only the file name (no folders) and safe characters."""
    base = re.split(r"[\\/]", name or "")[-1].strip()
    stem, dot, ext = base.rpartition(".")
    if not dot:
        stem, ext = base, ""
    stem = _UNSAFE.sub("_", stem).strip("._-")[:100]
    ext = _UNSAFE.sub("", ext).lower()[:8]
    if not stem:
        raise UnsupportedFileError("Invalid file name")
    return f"{stem}.{ext}" if ext else stem


def next_version(current: Optional[str]) -> str:
    try:
        return f"{int(float(current or '0')) + 1}.0"
    except ValueError:
        return "2.0"


@dataclass
class UploadOutcome:
    status: str                      # indexed | unchanged
    record: DocumentRecord
    result: IngestResult


class DocumentService:
    def __init__(self, settings: Settings, indexer: Indexer, store: QdrantStore, cache: Cache, known_tags: frozenset[str]):
        self.settings, self.indexer, self.store, self.cache, self.known_tags = settings, indexer, store, cache, known_tags

    def upload(self, principal: Principal, original_name: str, stream: BinaryIO, department: str,
               access_tags: str) -> UploadOutcome:
        name = safe_filename(original_name)
        ext = Path(name).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise UnsupportedFileError(f"Unsupported type '{ext or 'none'}'. Allowed: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")

        tags = parse_access_tags(access_tags)                       # at least one tag, valid characters
        not_yours = [t for t in tags if t not in principal.access_tags]
        if not_yours:
            raise PermissionDeniedError(f"You can only assign access tags you hold. Not allowed: {', '.join(not_yours)}")

        existing = self.store.find_document_by_file_name(name) if self._collection_exists() else None
        if existing and not set(existing["access_tags"]) & principal.access_tags:
            raise PermissionDeniedError("A document with this name exists and you may not replace it")

        upload_dir = Path(self.settings.upload_dir)
        upload_dir.mkdir(parents=True, exist_ok=True)
        tmp = upload_dir / f".{uuid.uuid4().hex}__{name}"
        try:
            sha = self._save(stream, tmp)
            document_id = existing["document_id"] if existing else f"UPL-{sha[:8].upper()}"
            record = DocumentRecord(
                document_id=document_id, file_name=name, title=Path(name).stem.replace("_", " "),
                file_type=SUPPORTED_EXTENSIONS[ext], department=(department or "General").strip()[:80],
                access_level="Uploaded", access_tags=tags,
                version=next_version(existing["version"]) if existing else "1.0")
            result = self.indexer.ingest_file(tmp, record=record)
            if result.status == "unchanged":                        # identical content: nothing changed, same version
                return UploadOutcome("unchanged", replace(record, version=existing["version"] if existing else record.version), result)
            if result.status != "indexed":
                if result.status == "rejected":
                    raise IngestionError(result.message)
                raise ExtractionError(result.message or "Processing failed")
            os.replace(tmp, upload_dir / name)                        # keep the original file
        finally:
            tmp.unlink(missing_ok=True)

        self.cache.invalidate_document(document_id)
        self.cache.bump_index_version()
        return UploadOutcome("indexed", record, result)

    # ----------------------------------------------------------------- helpers
    def _collection_exists(self) -> bool:
        return self.store.client.collection_exists(self.store.collection)

    def _save(self, stream: BinaryIO, target: Path) -> str:
        limit = self.settings.max_file_size_bytes
        size, digest = 0, hashlib.sha256()
        with target.open("wb") as out:
            while chunk := stream.read(1 << 20):
                size += len(chunk)
                if size > limit:
                    raise FileTooLargeError(f"File is larger than {self.settings.max_file_size_mb} MB")
                digest.update(chunk)
                out.write(chunk)
        if size == 0:
            raise EmptyFileError("The uploaded file is empty")
        return digest.hexdigest()
