"""Context blocks and citations, built from stored chunk metadata (never from model text)."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from app.retrieval.vector_store import Hit

_MARKER = re.compile(r"\[(\d{1,2})\]")


@dataclass
class ContextBlock:
    number: int                       # the [n] the model must cite
    hit: Hit
    text_block: str                   # what is sent to the model


@dataclass
class Citation:
    number: int
    document_id: str
    document_name: str
    document_title: str
    section: str
    page: Optional[int]
    sheet: Optional[str]
    chunk_id: str
    score: float
    snippet: str


def build_context(hits: list[Hit], max_words: int) -> list[ContextBlock]:
    """Number the hits 1..n and keep as many as fit in ``max_words`` (best first)."""
    blocks: list[ContextBlock] = []
    used = 0
    for hit in hits:
        pl = hit.payload
        words = len(pl.get("text", "").split())
        if blocks and used + words > max_words:
            break
        n = len(blocks) + 1
        where = [f"Document: {pl.get('file_name')}", f"Section: {pl.get('section_path') or 'Front matter'}"]
        if pl.get("page_number"):
            where.append(f"Page: {pl['page_number']}")
        if pl.get("sheet_name"):
            where.append(f"Sheet: {pl['sheet_name']}")
        blocks.append(ContextBlock(n, hit, f"[{n}] ({' | '.join(where)})\n{pl.get('text', '')}"))
        used += words
    return blocks


def extract_markers(answer: str, max_number: int) -> list[int]:
    """Block numbers the answer cites, in order, ignoring numbers that do not exist."""
    seen: list[int] = []
    for m in _MARKER.finditer(answer):
        n = int(m.group(1))
        if 1 <= n <= max_number and n not in seen:
            seen.append(n)
    return seen


def to_citation(block: ContextBlock) -> Citation:
    pl = block.hit.payload
    text = " ".join(pl.get("text", "").split())
    return Citation(
        number=block.number,
        document_id=pl.get("document_id", ""),
        document_name=pl.get("file_name", ""),
        document_title=pl.get("document_title", ""),
        section=pl.get("section_path", ""),
        page=pl.get("page_number"),
        sheet=pl.get("sheet_name"),
        chunk_id=pl.get("chunk_id", ""),
        score=round(block.hit.score, 4),
        snippet=text[:240] + ("..." if len(text) > 240 else ""),
    )


def select_citations(blocks: list[ContextBlock], answer: str, strict: bool) -> tuple[list[Citation], str]:
    """Citations for an answer.

    Returns ``(citations, source)``: ``model_markers`` when the model cited blocks with [n],
    ``retrieved_context`` when it did not (all blocks that were given to the model are cited),
    and an empty list when ``strict`` is on and the model cited nothing.
    """
    markers = extract_markers(answer, len(blocks))
    if markers:
        return [to_citation(blocks[n - 1]) for n in markers], "model_markers"
    if strict:
        return [], "none"
    return [to_citation(b) for b in blocks], "retrieved_context"
