"""Metadata for every chunk: document record, access tags and the stored payload.

The document inventory (``data/processed/document_inventory.csv``) is the single
source of truth for access tags, department and version.  A document that is
not in the inventory, or that has no access tag, is **rejected** (fail closed).
"""
from __future__ import annotations

import csv
import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.ingestion.models import Chunk, IngestionError, MissingAccessTagsError

_TAG = re.compile(r"^[a-z0-9_]+$")


@dataclass(frozen=True)
class DocumentRecord:
    document_id: str
    file_name: str
    title: str
    file_type: str
    department: str
    access_level: str
    access_tags: tuple[str, ...]
    version: str


def parse_access_tags(raw: str) -> tuple[str, ...]:
    """'hr|management' or 'hr, management' -> ('hr', 'management')."""
    tags = tuple(dict.fromkeys(t.strip().lower() for t in re.split(r"[|,;]", raw or "") if t.strip()))
    if not tags:
        raise MissingAccessTagsError("document has no access tags")
    bad = [t for t in tags if not _TAG.match(t)]
    if bad:
        raise MissingAccessTagsError(f"invalid access tag(s): {', '.join(bad)} (use a-z, 0-9 and _)")
    return tags


def load_inventory(path: Path) -> dict[str, DocumentRecord]:
    """Read the inventory CSV and return records keyed by file name."""
    path = Path(path)
    if not path.is_file():
        raise IngestionError(f"Document inventory not found: {path}")
    records: dict[str, DocumentRecord] = {}
    with path.open(newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            name = (row.get("file_name") or "").strip()
            if not name:
                continue
            try:
                tags = parse_access_tags(row.get("access_tags", ""))
            except MissingAccessTagsError:
                continue          # left out of the inventory, so the file is rejected at ingestion
            records[name] = DocumentRecord(
                document_id=row["document_id"].strip(),
                file_name=name,
                title=(row.get("title") or "").strip(),
                file_type=(row.get("file_type") or "").strip().lower(),
                department=(row.get("department") or "").strip(),
                access_level=(row.get("access_level") or "").strip(),
                access_tags=tags,
                version=(row.get("version") or "").strip(),
            )
    return records


def resolve_record(file_name: str, inventory: dict[str, DocumentRecord]) -> DocumentRecord:
    try:
        return inventory[file_name]
    except KeyError:
        raise MissingAccessTagsError(
            f"'{file_name}' is not in the inventory or has no access tags; it is rejected (fail closed)"
        ) from None


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def build_payload(chunk: Chunk, record: DocumentRecord, sha256: str, ingested_at: Optional[str] = None) -> dict:
    """The metadata stored next to the vector of one chunk."""
    return {
        # identity
        "chunk_id": chunk.chunk_id,
        "chunk_index": chunk.chunk_index,
        "document_id": record.document_id,
        "document_title": record.title,
        "document_version": record.version,
        "file_name": record.file_name,
        "file_type": record.file_type,
        "department": record.department,
        "access_level": record.access_level,
        # location (what the user sees as the source)
        "section_path": chunk.section_path,
        "title_path": chunk.title_path,
        "page_number": chunk.page_start,
        "page_start": chunk.page_start,
        "page_end": chunk.page_end,
        "sheet_name": chunk.sheet_name,
        "row_start": chunk.row_start,
        "row_end": chunk.row_end,
        # permission
        "access_tags": list(record.access_tags),
        # content
        "text": chunk.text,
        "chunk_type": chunk.chunk_type,
        "word_count": chunk.word_count,
        # housekeeping
        "file_sha256": sha256,
        "ingested_at": ingested_at or datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
