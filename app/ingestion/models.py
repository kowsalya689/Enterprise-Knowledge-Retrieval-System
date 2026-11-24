"""Data structures shared by the ingestion pipeline, plus its exceptions."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

BlockKind = Literal["heading", "paragraph", "list", "table"]


# --------------------------------------------------------------------------- errors
class IngestionError(Exception):
    """Base class for every expected, reportable ingestion failure."""


class UnsupportedFileError(IngestionError):
    """File type is not on the allow-list."""


class EmptyFileError(IngestionError):
    """File has no content."""


class FileTooLargeError(IngestionError):
    """File is larger than the configured limit."""


class ExtractionError(IngestionError):
    """File could not be read (corrupt, locked, wrong format)."""


class MissingAccessTagsError(IngestionError):
    """Document has no access tags; ingestion fails closed."""


# --------------------------------------------------------------------------- extraction
@dataclass
class Block:
    """One structural element produced by an extractor, in reading order."""

    kind: BlockKind
    text: str = ""                    # heading / paragraph text; list items joined by newline
    level: int = 0                    # heading level, 1 = highest
    page: Optional[int] = None        # page where the block starts (None = no fixed pages)
    page_end: Optional[int] = None    # page where the block ends
    headers: list[str] = field(default_factory=list)       # tables
    rows: list[list[str]] = field(default_factory=list)    # tables
    sheet: Optional[str] = None       # worksheet name
    row_start: Optional[int] = None   # file or sheet row number of the first table data row


@dataclass
class ExtractedDocument:
    file_name: str
    file_type: str
    title: str
    blocks: list[Block]
    page_count: Optional[int] = None


# --------------------------------------------------------------------------- chunking
@dataclass
class Chunk:
    chunk_id: str
    chunk_index: int
    text: str
    title_path: str                   # "Document title > Section > Sub-section"
    section_path: str                 # "Section > Sub-section"
    chunk_type: Literal["text", "table"]
    word_count: int
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    sheet_name: Optional[str] = None
    row_start: Optional[int] = None
    row_end: Optional[int] = None

    @property
    def page_number(self) -> Optional[int]:
        """The page where the chunk starts: this is the page cited to the user."""
        return self.page_start
