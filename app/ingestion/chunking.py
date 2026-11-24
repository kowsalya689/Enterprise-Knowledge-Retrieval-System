"""Structure-aware chunking: one method per document type.

======================  ===================  ====================================
Document type           Chunker              Rule
======================  ===================  ====================================
PDF, DOCX, HTML, MD,    ``SectionChunker``   one chunk per heading / sub-heading;
TXT                                          tables are kept whole
XLSX, XLS, CSV          ``TableChunker``     groups of rows, column names repeated
                                             in every row
======================  ===================  ====================================

Common rules
------------
* A chunk never has more than ``max_words`` words **including its title path**
  (``Document title > Section > Sub-section``).
* A section longer than the limit is split at paragraph, list-item or table-row
  boundaries; a long paragraph is split at sentence boundaries.  Only split text
  receives a small overlap (``overlap_words``); table rows never overlap.
* Every chunk gets a stable id ``<document_id>-C001``, ``-C002``, ...
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from app.config import Settings, get_settings
from app.ingestion.models import Block, Chunk, ExtractedDocument

logger = logging.getLogger(__name__)

_SENTENCE_END = re.compile(r"(?<=[.!?;])\s+")
_MIN_BUDGET = 40


def words(text: str) -> int:
    return len(text.split())


def serialize_row(headers: list[str], row: list[str]) -> str:
    """``Header: value; Header: value`` (empty cells are left out)."""
    return "; ".join(f"{h}: {v}" for h, v in zip(headers, row) if v != "")


@dataclass
class _Unit:
    """Smallest piece that is never split across chunks (unless it is too long)."""
    text: str
    kind: str = "text"                      # "text" | "table"
    page: Optional[int] = None
    page_end: Optional[int] = None
    sheet: Optional[str] = None
    row_start: Optional[int] = None
    row_end: Optional[int] = None
    overlap: bool = False                   # copied from the previous chunk

    @property
    def words(self) -> int:
        return words(self.text)


@dataclass
class _Draft:
    units: list[_Unit]
    section_path: str
    label: str = ""                         # used as section path when there is no heading
    heading_page: Optional[int] = None      # page where the section heading starts


def _split_long_text(text: str, budget: int) -> list[str]:
    """Split text that is longer than ``budget`` words, preferring sentence ends."""
    pieces, current, count = [], [], 0
    for sentence in _SENTENCE_END.split(text):
        n = words(sentence)
        if n > budget:                       # no sentence boundary: cut by words
            if current:
                pieces.append(" ".join(current)); current, count = [], 0
            tokens = sentence.split()
            pieces += [" ".join(tokens[i:i + budget]) for i in range(0, len(tokens), budget)]
            continue
        if count + n > budget and current:
            pieces.append(" ".join(current)); current, count = [], 0
        current.append(sentence); count += n
    if current:
        pieces.append(" ".join(current))
    return pieces


def _overlap_tail(text: str, overlap_words: int) -> str:
    """Last ``overlap_words`` words of ``text``, starting at a sentence start if possible."""
    tokens = text.split()
    if len(tokens) <= overlap_words:
        return text
    tail = " ".join(tokens[-overlap_words:])
    m = _SENTENCE_END.search(tail)
    return tail[m.end():] if m and words(tail[m.end():]) >= overlap_words // 2 else tail


class SectionChunker:
    """Heading-based chunking for documents that have a heading structure."""

    def __init__(self, max_words: int = 380, overlap_words: int = 30):
        self.max_words = max_words
        self.overlap_words = overlap_words

    # ------------------------------------------------------------------ public
    def chunk(self, doc: ExtractedDocument, document_id: str) -> list[Chunk]:
        drafts: list[_Draft] = []
        for section_path, blocks, heading_page in self._sections(doc):
            title_path = f"{doc.title} > {section_path}" if section_path else doc.title
            budget = max(self.max_words - words(title_path), _MIN_BUDGET)
            section_drafts = self._section_drafts(blocks, section_path, budget)
            if section_drafts:
                section_drafts[0].heading_page = heading_page   # the cited page is where the heading starts
            drafts += section_drafts
        chunks = [self._finalise(d, doc.title, document_id, i) for i, d in enumerate(drafts)]
        logger.debug("%s: %d chunks", doc.file_name, len(chunks))
        return chunks

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def _sections(doc: ExtractedDocument):
        """Yield (section_path, blocks, heading_page) for every heading and for any text before the first one."""
        stack: list[tuple[int, str]] = []
        current: list[Block] = []
        path, heading_page = "", None
        for block in doc.blocks:
            if block.kind == "heading":
                if current:
                    yield path, current, heading_page
                current = []
                while stack and stack[-1][0] >= block.level:
                    stack.pop()
                stack.append((block.level, block.text))
                path = " > ".join(t for _, t in stack)
                heading_page = block.page
            else:
                current.append(block)
        if current:
            yield path, current, heading_page

    def _units(self, blocks: list[Block], budget: int) -> list[_Unit]:
        units: list[_Unit] = []
        for b in blocks:
            if b.kind == "paragraph":
                units.append(_Unit(b.text, page=b.page, page_end=b.page_end or b.page, sheet=b.sheet))
            elif b.kind == "list":
                units += [_Unit(line, page=b.page, page_end=b.page_end or b.page, sheet=b.sheet) for line in b.text.split("\n")]
            elif b.kind == "table":
                units += self._table_units(b, budget)
        return units

    def _table_units(self, table: Block, budget: int) -> list[_Unit]:
        """A table stays whole when it fits; otherwise it is split by rows."""
        lines = [serialize_row(table.headers, r) for r in table.rows]
        lines = [l for l in lines if l]
        whole = "\n".join(lines)
        if words(whole) <= budget:
            return [_Unit(whole, "table", table.page, table.page_end or table.page, table.sheet)] if whole else []
        return [_Unit(t, "table", table.page, table.page_end or table.page, table.sheet)
                for t in self._group_lines(lines, budget)]

    @staticmethod
    def _group_lines(lines: list[str], budget: int, max_rows: Optional[int] = None) -> list[str]:
        groups, current, count = [], [], 0
        for line in lines:
            n = words(line)
            if n > budget:                                   # a single very long row
                if current:
                    groups.append("\n".join(current)); current, count = [], 0
                groups += _split_long_text(line, budget)
                continue
            if current and (count + n > budget or (max_rows and len(current) >= max_rows)):
                groups.append("\n".join(current)); current, count = [], 0
            current.append(line); count += n
        if current:
            groups.append("\n".join(current))
        return groups

    def _section_drafts(self, blocks: list[Block], section_path: str, budget: int) -> list[_Draft]:
        return self._pack(self._units(blocks, budget), section_path, budget)

    def _pack(self, units: list[_Unit], section_path: str, budget: int) -> list[_Draft]:
        """Fill chunks greedily with units, in order, up to ``budget`` words."""
        # a unit longer than the budget is split into pieces first
        piece_budget = budget - self.overlap_words if self.overlap_words < budget // 2 else budget
        expanded: list[_Unit] = []
        for u in units:
            if u.words > budget:
                expanded += [_Unit(t, u.kind, u.page, u.page_end, u.sheet) for t in _split_long_text(u.text, piece_budget)]
            else:
                expanded.append(u)

        drafts: list[_Draft] = []
        current: list[_Unit] = []
        count = 0
        for u in expanded:
            if current and count + u.words > budget:
                drafts.append(_Draft(current, section_path))
                tail_unit = self._overlap_unit(current[-1], u, budget)
                current, count = ([tail_unit], tail_unit.words) if tail_unit else ([], 0)
            current.append(u)
            count += u.words
        if current and any(not x.overlap for x in current):
            drafts.append(_Draft(current, section_path))
        return drafts

    def _overlap_unit(self, previous: _Unit, nxt: _Unit, budget: int) -> Optional[_Unit]:
        if not self.overlap_words or previous.kind != "text" or nxt.kind != "text":
            return None
        tail = _overlap_tail(previous.text, self.overlap_words)
        if words(tail) + nxt.words > budget:
            return None
        return _Unit(tail, "text", previous.page_end or previous.page, previous.page_end, previous.sheet, overlap=True)

    def _finalise(self, draft: _Draft, doc_title: str, document_id: str, index: int) -> Chunk:
        real = [u for u in draft.units if not u.overlap] or draft.units
        text = "\n".join(u.text for u in draft.units)
        section = draft.section_path or draft.label or "Front matter"
        title_path = f"{doc_title} > {section}" if section else doc_title
        pages = [p for u in real for p in (u.page, u.page_end) if p]
        if draft.heading_page:
            pages.append(draft.heading_page)
        rows = [r for u in real for r in (u.row_start, u.row_end) if r]
        return Chunk(
            chunk_id=f"{document_id}-C{index + 1:03d}",
            chunk_index=index,
            text=text,
            title_path=title_path,
            section_path=section,
            chunk_type="table" if any(u.kind == "table" for u in real) else "text",
            word_count=words(text),
            page_start=min(pages) if pages else None,
            page_end=max(pages) if pages else None,
            sheet_name=next((u.sheet for u in real if u.sheet), None),
            row_start=min(rows) if rows else None,
            row_end=max(rows) if rows else None,
        )


class TableChunker(SectionChunker):
    """Row-group chunking for spreadsheets and CSV files.

    Every chunk holds up to ``rows_per_chunk`` rows; each row is written as
    ``Column: value; Column: value`` so it keeps its column names.  Text that
    comes before a table in the same section (title, notes) is added to the
    first row group of that table.
    """

    def __init__(self, max_words: int = 380, overlap_words: int = 30, rows_per_chunk: int = 12):
        super().__init__(max_words, overlap_words)
        self.rows_per_chunk = rows_per_chunk

    def _section_drafts(self, blocks: list[Block], section_path: str, budget: int) -> list[_Draft]:
        drafts: list[_Draft] = []
        pending: list[Block] = []
        first_table_done = False

        for block in blocks:
            if block.kind != "table":
                pending.append(block)
                continue
            lead = self._units(pending, budget) if (pending and not first_table_done) else []
            if pending and first_table_done:
                drafts += self._pack(self._units(pending, budget), section_path, budget)
            pending = []
            drafts += self._table_drafts(block, section_path, budget, lead)
            first_table_done = True
        if pending:
            drafts += self._pack(self._units(pending, budget), section_path, budget)
        return drafts

    def _table_drafts(self, table: Block, section_path: str, budget: int, lead: list[_Unit]) -> list[_Draft]:
        entries = [(serialize_row(table.headers, r), i) for i, r in enumerate(table.rows)]
        entries = [(line, i) for line, i in entries if line]
        drafts: list[_Draft] = []
        group: list[tuple[str, int]] = []
        count = 0

        def emit():
            nonlocal group, count, lead
            if not group:
                return
            first, last = group[0][1], group[-1][1]
            units = list(lead)
            lead = []                                    # lead-in goes only into the first group
            units.append(_Unit("\n".join(l for l, _ in group), "table", table.page, table.page_end or table.page, table.sheet,
                               table.row_start + first if table.row_start else None,
                               table.row_start + last if table.row_start else None))
            label = "" if section_path else f"Rows {first + 1}-{last + 1}"
            drafts.append(_Draft(units, section_path, label))
            group, count = [], 0

        lead_words = sum(u.words for u in lead)
        if lead_words > budget // 2:                      # too long to share a chunk
            drafts += self._pack(lead, section_path, budget)
            lead, lead_words = [], 0
        for line, idx in entries:
            n = words(line)
            allowed = budget - lead_words if lead else budget
            if n > allowed:                               # very long single row: cut by words
                emit()
                for piece in _split_long_text(line, budget):
                    group, count = [(piece, idx)], words(piece)
                    emit()
                continue
            if group and (len(group) >= self.rows_per_chunk or count + n > allowed):
                emit()
            group.append((line, idx)); count += n
        emit()
        return drafts


_DOCUMENT_TYPES = {"pdf", "docx", "html", "md", "txt"}
_TABULAR_TYPES = {"xlsx", "xls", "csv"}


def get_chunker(file_type: str, settings: Optional[Settings] = None) -> SectionChunker:
    """Return the chunker for a document type."""
    settings = settings or get_settings()
    if file_type in _TABULAR_TYPES:
        return TableChunker(settings.chunk_max_words, settings.chunk_overlap_words, settings.table_rows_per_chunk)
    if file_type in _DOCUMENT_TYPES:
        return SectionChunker(settings.chunk_max_words, settings.chunk_overlap_words)
    raise ValueError(f"No chunker for file type '{file_type}'")


def chunk_document(doc: ExtractedDocument, document_id: str, settings: Optional[Settings] = None) -> list[Chunk]:
    """Chunk a cleaned document with the method that belongs to its type."""
    return get_chunker(doc.file_type, settings).chunk(doc, document_id)
