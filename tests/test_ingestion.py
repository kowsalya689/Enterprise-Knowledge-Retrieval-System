"""Extraction, preprocessing, chunking and metadata, tested on the real dataset."""
from __future__ import annotations

import csv
from collections import Counter

import pytest

from app.config import PROJECT_ROOT
from app.ingestion.chunking import SectionChunker, TableChunker, chunk_document, words
from app.ingestion.extractors import extract, validate_source_file
from app.ingestion.metadata import build_payload, file_sha256, parse_access_tags, resolve_record
from app.ingestion.models import (
    Block, EmptyFileError, ExtractedDocument, FileTooLargeError, MissingAccessTagsError, UnsupportedFileError,
)
from app.ingestion.preprocess import preprocess
from tests.conftest import HAS_SOFFICE, RAW_DIR

ALL_FILES = sorted(RAW_DIR.iterdir())
EXPECTED = list(csv.DictReader((PROJECT_ROOT / "tests" / "data" / "expected_chunks.csv").open(encoding="utf-8")))


def _file(prefix: str):
    return next(p for p in ALL_FILES if p.name.startswith(prefix))


def _chunks(prefix: str, settings):
    doc = preprocess(extract(_file(prefix), settings))
    return doc, chunk_document(doc, prefix, settings)


# ------------------------------------------------------------------ validation
def test_validation_rules(tmp_path):
    ok = tmp_path / "a.txt"; ok.write_text("hello")
    assert validate_source_file(ok, 1000) == "txt"
    empty = tmp_path / "e.pdf"; empty.write_bytes(b"")
    with pytest.raises(EmptyFileError):
        validate_source_file(empty, 1000)
    bad = tmp_path / "x.exe"; bad.write_bytes(b"MZ")
    with pytest.raises(UnsupportedFileError):
        validate_source_file(bad, 1000)
    big = tmp_path / "big.csv"; big.write_bytes(b"a,b\n" * 1000)
    with pytest.raises(FileTooLargeError):
        validate_source_file(big, 100)


# ------------------------------------------------------------------ extractors, one per format
def test_pdf_headings_tables_and_pages(settings):
    doc = preprocess(extract(_file("DOC-002"), settings))
    assert doc.title == "Information Security Policy" and doc.page_count == 5
    heads = [b for b in doc.blocks if b.kind == "heading"]
    assert [h.text for h in heads][:3] == ["1. Purpose and Objectives", "2. Information Classification", "3. Access Control"]
    assert next(h for h in heads if h.text.startswith("3.1")).level == 2
    tables = [b for b in doc.blocks if b.kind == "table"]
    assert tables and all(len(t.headers) >= 2 for t in tables)
    # running footer ("... | Page 3") must be gone
    assert not any("Page " in b.text and "DOC-002" in b.text for b in doc.blocks if b.kind == "paragraph")
    assert max(b.page for b in doc.blocks) == 5


def test_pdf_repeated_table_header_rows_are_not_data(settings):
    doc = preprocess(extract(_file("DOC-003"), settings))
    for t in (b for b in doc.blocks if b.kind == "table"):
        assert tuple(t.rows[0]) != tuple(t.headers)


@pytest.mark.skipif(not HAS_SOFFICE, reason="LibreOffice is needed to find DOCX page numbers")
def test_docx_structure_and_pages(settings):
    doc = preprocess(extract(_file("DOC-001"), settings))
    assert doc.title == "Leave and Attendance Policy"
    assert doc.page_count == 7
    h = next(b for b in doc.blocks if b.kind == "heading" and b.text.startswith("5.2"))
    assert h.level == 2 and h.page == 4


def test_docx_without_page_rendering(settings):
    s = settings.model_copy(update={"docx_page_mode": "off"})
    doc = extract(_file("DOC-004"), s)
    assert doc.page_count is None and any(b.kind == "table" for b in doc.blocks)
    assert all(b.page is None for b in doc.blocks)


def test_markdown_reader(settings):
    doc = extract(_file("DOC-005"), settings)
    assert doc.title == "Incident Response Procedure (SOP)"
    levels = {b.text: b.level for b in doc.blocks if b.kind == "heading"}
    assert levels["5. Procedure"] == 2 and levels["5.1 Step 1: Detect and report"] == 3
    table = next(b for b in doc.blocks if b.kind == "table" and "Severity" in b.headers)
    assert len(table.rows) == 4 and table.rows[0][0] == "P1 Critical"
    assert any(b.kind == "list" and b.text.startswith("- ") for b in doc.blocks)


def test_markdown_front_matter_code_and_nested_list(tmp_path, settings):
    f = tmp_path / "n.md"
    f.write_text("---\ntitle: x\n---\n# Doc\n\n## A\n\ntext **bold** and [link](http://x)\n\n- one\n  - nested\n- two\n\n```\n# not a heading\n```\n")
    doc = extract(f, settings)
    assert doc.title == "Doc"
    assert [b.text for b in doc.blocks if b.kind == "heading"] == ["A"]
    assert next(b for b in doc.blocks if b.kind == "paragraph").text == "text bold and link"
    assert next(b for b in doc.blocks if b.kind == "list").text == "- one\n- nested\n- two"


def test_html_reader(settings):
    doc = extract(_file("DOC-006"), settings)
    assert doc.title == "Employee Onboarding Guide"
    assert not any(b.kind == "heading" and b.level == 1 for b in doc.blocks)     # <h1> repeats the title
    assert any(b.kind == "heading" and b.text == "3. Your First Day" for b in doc.blocks)
    assert not any("font-family" in b.text for b in doc.blocks)                  # <style> removed


def test_txt_reader(settings):
    doc = extract(_file("DOC-007"), settings)
    assert doc.title == "CUSTOMER SUPPORT ESCALATION SOP"
    heads = [b.text for b in doc.blocks if b.kind == "heading"]
    assert "3. TICKET PRIORITY AND SERVICE LEVELS" in heads
    assert any(b.kind == "list" and "P1 Urgent" in b.text for b in doc.blocks)


def test_xlsx_reader(settings):
    doc = extract(_file("DOC-008"), settings)
    sheets = [b.text for b in doc.blocks if b.kind == "heading" and b.level == 1]
    assert sheets == ["Summary", "Regional", "Products", "Detail", "Pipeline"]
    key = next(b for b in doc.blocks if b.kind == "table" and b.headers == ["Metric", "Value"])
    assert key.sheet == "Summary" and ["Total Q3 revenue (INR lakh)", "4244.9"] in key.rows


def test_xls_reader_reads_old_format(settings):
    doc = extract(_file("DOC-009"), settings)
    assert [b.text for b in doc.blocks if b.kind == "heading" and b.level == 1] == ["Summary", "Quarterly Phasing", "Capex Projects", "Assumptions"]
    summary = next(b for b in doc.blocks if b.kind == "table" and b.sheet == "Summary")
    assert summary.rows[0][0] == "Manufacturing" and summary.rows[-1][0] == "Total"
    assert "2990" in summary.rows[0]                      # 2990.0 is written as 2990


def test_csv_reader(settings):
    doc = extract(_file("DOC-010"), settings)
    table = doc.blocks[0]
    assert table.headers[0] == "ticket_id" and len(table.rows) == 160 and table.row_start == 2


def test_csv_semicolon_and_utf8_bom(tmp_path, settings):
    f = tmp_path / "d.csv"
    f.write_bytes("\ufeffa;b\n1;2\n3;4\n".encode("utf-8"))
    t = extract(f, settings).blocks[0]
    assert t.headers == ["a", "b"] and t.rows == [["1", "2"], ["3", "4"]]


# ------------------------------------------------------------------ preprocessing
def test_preprocess_normalises_and_drops_noise():
    blocks = [Block("paragraph", "Caf\u00e9\u200b  text\u00a0here", page=1),
              Block("paragraph", "Page 3", page=1), Block("paragraph", "  ", page=1),
              Block("table", headers=["A", " B "], rows=[["1", ""], ["", ""]], page=1)]
    doc = preprocess(ExtractedDocument("x.pdf", "pdf", " T ", blocks, 1))
    assert [b.text for b in doc.blocks if b.kind == "paragraph"] == ["Caf\u00e9 text here"]
    table = doc.blocks[-1]
    assert table.headers == ["A", "B"] and table.rows == [["1", ""]]


# ------------------------------------------------------------------ chunking
@pytest.mark.parametrize("path", ALL_FILES, ids=lambda p: p.name[:7])
def test_every_chunk_respects_word_limit_and_has_title_path(path, settings):
    doc = preprocess(extract(path, settings))
    chunks = chunk_document(doc, path.name[:7], settings)
    assert chunks
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids)) and ids[0] == f"{path.name[:7]}-C001"
    for c in chunks:
        assert c.title_path.startswith(doc.title)
        assert words(c.title_path) + c.word_count <= settings.chunk_max_words, c.chunk_id
        assert c.text.strip()


def test_long_section_is_split_with_overlap_and_title_path():
    para = " ".join(f"Sentence number {i} says something useful about leave." for i in range(120))
    doc = ExtractedDocument("l.txt", "txt", "Long Doc", [Block("heading", "1. Big", level=1), Block("paragraph", para)])
    chunks = SectionChunker(max_words=100, overlap_words=15).chunk(doc, "D")
    assert len(chunks) > 3
    for c in chunks:
        assert words(c.title_path) + c.word_count <= 100
        assert c.section_path == "1. Big"
    # overlap: the start of a chunk repeats the end of the previous one
    first_words = chunks[1].text.split()[:5]
    assert " ".join(first_words) in chunks[0].text


def test_large_table_is_split_by_rows_with_column_names():
    rows = [[f"item{i}", "x " * 20] for i in range(40)]
    doc = ExtractedDocument("t.pdf", "pdf", "T", [Block("heading", "1. Tab", level=1), Block("table", headers=["Name", "Note"], rows=rows)])
    chunks = SectionChunker(max_words=120, overlap_words=10).chunk(doc, "D")
    assert len(chunks) > 3 and all(c.chunk_type == "table" for c in chunks)
    assert all(line.startswith("Name: item") for c in chunks for line in c.text.splitlines())
    assert sum(len(c.text.splitlines()) for c in chunks) == 40            # no overlap, no lost rows


def test_table_chunker_groups_of_rows(settings):
    doc = preprocess(extract(_file("DOC-010"), settings))
    chunks = TableChunker(380, 30, rows_per_chunk=12).chunk(doc, "DOC-010")
    assert [c.section_path for c in chunks][:2] == ["Rows 1-12", "Rows 13-24"]
    assert chunks[0].row_start == 2 and chunks[0].row_end == 13
    assert all(len(c.text.splitlines()) <= 12 for c in chunks)


def test_different_chunker_per_document_type(settings):
    from app.ingestion.chunking import get_chunker
    for t in ("pdf", "docx", "html", "md", "txt"):
        assert type(get_chunker(t, settings)) is SectionChunker
    for t in ("xlsx", "xls", "csv"):
        assert type(get_chunker(t, settings)) is TableChunker


# ------------------------------------------------------------------ extraction quality against the labelled dataset
@pytest.mark.parametrize("row", EXPECTED, ids=lambda r: f"{r['question_id']}-{r['document_id']}")
def test_expected_section_page_and_text(row, settings):
    doc_id = row["document_id"]
    path = _file(doc_id)
    if path.suffix == ".docx" and not HAS_SOFFICE:
        pytest.skip("LibreOffice needed for DOCX pages")
    doc = preprocess(extract(path, settings))
    chunks = chunk_document(doc, doc_id, settings)
    want_section = row["section_path"].casefold()                        # TXT headings are upper case
    hits = [c for c in chunks if c.section_path.casefold() == want_section and row["evidence_phrase"] in c.text]
    assert hits, f"no chunk with section '{row['section_path']}' containing '{row['evidence_phrase']}'"
    if row["page_number"]:
        assert any(str(c.page_number) == row["page_number"] for c in hits), \
            f"expected page {row['page_number']}, got {[c.page_number for c in hits]}"
    else:
        assert all(c.page_number is None for c in hits)


# ------------------------------------------------------------------ metadata
def test_access_tags_are_parsed_and_validated():
    assert parse_access_tags("HR | management") == ("hr", "management")
    with pytest.raises(MissingAccessTagsError):
        parse_access_tags("")
    with pytest.raises(MissingAccessTagsError):
        parse_access_tags("bad tag!")


def test_document_not_in_inventory_is_rejected(inventory):
    with pytest.raises(MissingAccessTagsError):
        resolve_record("unknown.pdf", inventory)


def test_payload_has_required_metadata(settings, inventory):
    path = _file("DOC-011")
    record = resolve_record(path.name, inventory)
    chunks = chunk_document(preprocess(extract(path, settings)), record.document_id, settings)
    payload = build_payload(chunks[3], record, file_sha256(path))
    assert payload["file_name"] == path.name
    assert payload["access_tags"] == ["hr", "management"]
    assert payload["section_path"] and payload["page_number"] and payload["text"]
    assert payload["document_id"] == "DOC-011" and payload["document_version"] == record.version


def test_chunk_counts_by_type_are_stable(settings):
    counts = Counter()
    for p in ALL_FILES:
        counts[p.suffix] += len(chunk_document(preprocess(extract(p, settings)), p.name[:7], settings))
    assert sum(counts.values()) == 163
