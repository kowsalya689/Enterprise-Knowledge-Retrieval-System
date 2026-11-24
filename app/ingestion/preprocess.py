"""Cleaning and normalising of extracted blocks.

Runs between extraction and chunking:

* Unicode normalisation (NFKC), removal of zero-width and soft-hyphen characters;
* whitespace clean-up;
* removal of page numbers and of headers / footers repeated on most pages;
* removal of empty blocks and empty table rows.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from collections import defaultdict

from app.ingestion.models import Block, ExtractedDocument

logger = logging.getLogger(__name__)

_INVISIBLE = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff\u00ad"), None)
_SPACES = re.compile(r"[ \t\f\v]+")
_PAGE_NUMBER = re.compile(r"^(page\s*)?\d{1,4}(\s*(of|/)\s*\d{1,4})?$", re.I)
_DIGITS = re.compile(r"\d+")


def normalise_text(text: str) -> str:
    """NFKC, invisible characters removed, runs of spaces collapsed."""
    text = unicodedata.normalize("NFKC", text or "").translate(_INVISIBLE)
    return _SPACES.sub(" ", text).strip()


def _normalise_lines(text: str) -> str:
    return "\n".join(l for l in (normalise_text(x) for x in text.split("\n")) if l)


def _boilerplate_keys(blocks: list[Block], page_count: int | None) -> set[str]:
    """Texts that appear on most pages (running headers and footers)."""
    if not page_count or page_count < 3:
        return set()
    pages_by_key: dict[str, set[int]] = defaultdict(set)
    for b in blocks:
        if b.kind == "paragraph" and b.page:
            pages_by_key[_DIGITS.sub("#", b.text.lower())].add(b.page)
    threshold = max(3, page_count * 0.5)
    return {k for k, pages in pages_by_key.items() if len(pages) >= threshold}


def preprocess(doc: ExtractedDocument) -> ExtractedDocument:
    """Return a cleaned copy of ``doc``."""
    repeated = _boilerplate_keys(doc.blocks, doc.page_count)
    cleaned: list[Block] = []
    dropped = 0
    for b in doc.blocks:
        if b.kind in ("heading", "paragraph"):
            b.text = normalise_text(b.text)
            if not b.text or (b.kind == "paragraph" and (
                    _PAGE_NUMBER.match(b.text) or _DIGITS.sub("#", b.text.lower()) in repeated)):
                dropped += 1
                continue
        elif b.kind == "list":
            b.text = _normalise_lines(b.text)
            if not b.text:
                dropped += 1
                continue
        elif b.kind == "table":
            b.headers = [normalise_text(h) for h in b.headers]
            b.rows = [[normalise_text(c) for c in row] for row in b.rows]
            b.rows = [row for row in b.rows if any(row)]
            if not b.rows and not any(b.headers):
                dropped += 1
                continue
        cleaned.append(b)
    if dropped:
        logger.debug("%s: removed %d empty or repeated blocks", doc.file_name, dropped)
    doc.blocks = cleaned
    doc.title = normalise_text(doc.title)
    return doc
