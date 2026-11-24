"""Ingestion pipeline: file -> chunks -> metadata -> embeddings -> vector database.

Steps for every file (progress is logged):

 1. validate     allowed type, not empty, within the size limit
 2. metadata     look the file up in the inventory (access tags are mandatory)
 3. extract      structure reader for the file type
 4. preprocess   clean text, remove repeated headers and footers
 5. chunk        one method per document type, max 380 words with the title path
 6. embed        document prefix + title path + chunk text
 7. store        one point per chunk; old points of the same document are replaced

Run from the project folder::

    python -m app.ingestion.indexer --dry-run                  # steps 1-5 only, nothing stored
    python -m app.ingestion.indexer                            # all steps, whole data/raw folder
    python -m app.ingestion.indexer --path data/raw/DOC-003_Finance_Travel_Reimbursement_Policy.pdf
    python -m app.ingestion.indexer --force                    # re-embed unchanged files
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.config import Settings, get_settings
from app.ingestion.chunking import chunk_document
from app.ingestion.extractors import SUPPORTED_EXTENSIONS, extract, validate_source_file
from app.ingestion.metadata import DocumentRecord, build_payload, file_sha256, load_inventory, resolve_record
from app.ingestion.models import Chunk, IngestionError
from app.ingestion.preprocess import preprocess

logger = logging.getLogger("app.ingestion.indexer")


@dataclass
class IngestResult:
    file_name: str
    status: str                          # indexed | unchanged | dry_run | rejected | failed
    document_id: str = ""
    file_type: str = ""
    pages: Optional[int] = None
    chunks: int = 0
    max_words: int = 0
    avg_words: float = 0.0
    seconds: float = 0.0
    message: str = ""
    payloads: list[dict] = field(default_factory=list, repr=False)


def build_chunks(path: Path, record: DocumentRecord, settings: Settings) -> tuple[list[Chunk], Optional[int], str]:
    """Steps 3-5: extract, preprocess and chunk. Returns (chunks, page_count, file_type)."""
    doc = preprocess(extract(path, settings))
    if record.title:
        doc.title = record.title                      # the inventory title is authoritative
    chunks = chunk_document(doc, record.document_id, settings)
    if not chunks:
        raise IngestionError(f"'{path.name}' produced no chunks")
    return chunks, doc.page_count, doc.file_type


class Indexer:
    def __init__(self, settings: Optional[Settings] = None, embedder=None, store=None,
                 inventory: Optional[dict[str, DocumentRecord]] = None):
        self.settings = settings or get_settings()
        self.embedder = embedder              # None => dry run (steps 1-5 only)
        self.store = store
        self.inventory = inventory if inventory is not None else load_inventory(self.settings.inventory_path)
        self._collection_ready = False

    def _ensure_ready(self) -> None:
        """Create the collection (size taken from the model) before the first write."""
        if self.store is not None and self.embedder is not None and not self._collection_ready:
            self.store.ensure_collection(self.embedder.dimension, getattr(self.embedder, "name", None))
            self._collection_ready = True

    # ------------------------------------------------------------------ one file
    def ingest_file(self, path: Path, force: bool = False, record: Optional[DocumentRecord] = None) -> IngestResult:
        path = Path(path)
        started = time.perf_counter()
        result = IngestResult(file_name=path.name, status="failed")
        try:
            logger.info("[%s] 1/7 validate", path.name)
            result.file_type = validate_source_file(path, self.settings.max_file_size_bytes)

            logger.info("[%s] 2/7 metadata (access tags)", path.name)
            record = record or resolve_record(path.name, self.inventory)
            result.document_id = record.document_id
            sha = file_sha256(path)
            self._ensure_ready()

            if self.store is not None and not force and self.store.fingerprint(record.document_id) == sha:
                result.status, result.message = "unchanged", "already indexed with identical content (use --force)"
                result.chunks = self.store.count(record.document_id)
                return result

            logger.info("[%s] 3-5/7 extract, preprocess, chunk", path.name)
            chunks, pages, _ = build_chunks(path, record, self.settings)
            result.pages = pages
            result.chunks = len(chunks)
            sizes = [c.word_count for c in chunks]
            result.max_words, result.avg_words = max(sizes), sum(sizes) / len(sizes)
            ingested_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
            result.payloads = [build_payload(c, record, sha, ingested_at) for c in chunks]

            if self.embedder is None or self.store is None:
                result.status = "dry_run"
                return result

            from app.retrieval.embeddings import chunk_embedding_text
            from app.retrieval.vector_store import ChunkPoint

            logger.info("[%s] 6/7 embed %d chunks", path.name, len(chunks))
            texts = [chunk_embedding_text(c) for c in chunks]
            stats = self.embedder.token_stats(texts) if hasattr(self.embedder, "token_stats") else {}
            if stats.get("over_limit"):
                logger.warning("[%s] %d chunk(s) exceed %d tokens (longest %d) and will be truncated by the model; "
                               "lower CHUNK_MAX_WORDS", path.name, stats["over_limit"], stats["limit"], stats["max_tokens"])
            vectors = self.embedder.embed_documents(texts)
            for payload in result.payloads:       # never mix models in one collection: remember which one made the vectors
                payload["embedding_model"] = getattr(self.embedder, "name", "unknown")
                payload["embedding_dim"] = int(vectors.shape[1])

            logger.info("[%s] 7/7 store in '%s'", path.name, self.store.collection)
            points = [ChunkPoint(c.chunk_id, vectors[i], result.payloads[i]) for i, c in enumerate(chunks)]
            self.store.delete_document(record.document_id)       # replace: remove the old version first
            self.store.upsert(points)
            result.status = "indexed"
        except IngestionError as exc:
            result.status = "rejected"
            result.message = str(exc)
            logger.error("[%s] rejected: %s", path.name, exc)
        except Exception as exc:                                  # unexpected: report, keep going
            result.status = "failed"
            result.message = f"{type(exc).__name__}: {exc}"
            logger.exception("[%s] failed", path.name)
        finally:
            result.seconds = time.perf_counter() - started
        return result

    # --------------------------------------------------------------- a directory
    def ingest_directory(self, directory: Path, force: bool = False) -> list[IngestResult]:
        files = sorted(p for p in Path(directory).iterdir()
                       if p.is_file() and not p.name.startswith(("~$", ".")) and p.suffix.lower() in SUPPORTED_EXTENSIONS)
        if not files:
            logger.warning("No supported files found in %s", directory)
        return [self.ingest_file(p, force) for p in files]


# ------------------------------------------------------------------------- CLI
def _print_summary(results: list[IngestResult]) -> None:
    print("\n" + "=" * 100)
    print(f"{'file':46s} {'status':10s} {'pages':>5s} {'chunks':>6s} {'max w':>6s} {'avg w':>6s} {'sec':>6s}")
    print("-" * 100)
    for r in results:
        pages = str(r.pages) if r.pages else "-"
        print(f"{r.file_name[:46]:46s} {r.status:10s} {pages:>5s} {r.chunks:6d} {r.max_words:6d} {r.avg_words:6.0f} {r.seconds:6.1f}")
        if r.message:
            print(f"    -> {r.message}")
    print("-" * 100)
    ok = [r for r in results if r.status in ("indexed", "unchanged", "dry_run")]
    print(f"{len(ok)} of {len(results)} files ok, {sum(r.chunks for r in ok)} chunks")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest documents: chunk, add metadata, embed, store in Qdrant")
    parser.add_argument("--path", type=Path, help="a file or a folder (default: data/raw)")
    parser.add_argument("--inventory", type=Path, help="inventory CSV with access tags")
    parser.add_argument("--dry-run", action="store_true", help="extract and chunk only; no embeddings, nothing stored")
    parser.add_argument("--force", action="store_true", help="re-embed files that are already indexed unchanged")
    parser.add_argument("--export-chunks", type=Path, help="write the chunks (with metadata) to a JSONL file")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")

    settings = get_settings()
    if args.inventory:
        settings = settings.model_copy(update={"inventory_path": args.inventory})
    target = args.path or settings.raw_data_dir

    embedder = store = None
    if not args.dry_run:
        from app.retrieval.embeddings import SentenceTransformerEmbedder
        from app.retrieval.vector_store import QdrantStore, VectorStoreError
        try:
            store = QdrantStore(settings)
            embedder = SentenceTransformerEmbedder(settings)
        except VectorStoreError as exc:
            print(f"ERROR: {exc}")
            return 2

    indexer = Indexer(settings, embedder, store)
    try:
        if Path(target).is_dir():
            results = indexer.ingest_directory(target, args.force)
        else:
            results = [indexer.ingest_file(target, args.force)]
    except IngestionError as exc:
        print(f"ERROR: {exc}")
        return 2

    if args.export_chunks:
        args.export_chunks.parent.mkdir(parents=True, exist_ok=True)
        with args.export_chunks.open("w", encoding="utf-8") as fh:
            for r in results:
                for payload in r.payloads:
                    fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
        print(f"Chunks written to {args.export_chunks}")

    _print_summary(results)
    if store is not None:
        print(f"Collection '{store.collection}' now holds {store.count()} chunks.")
        print("Open the Qdrant dashboard -> Collections -> " + store.collection + " to look at the points.")
    return 0 if all(r.status in ("indexed", "unchanged", "dry_run") for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
