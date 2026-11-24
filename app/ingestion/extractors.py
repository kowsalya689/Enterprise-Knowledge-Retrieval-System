"""Structure readers: one extractor per file type.

Each extractor returns an :class:`ExtractedDocument`, a list of ordered
:class:`Block` objects (headings, paragraphs, lists, tables) that keep the
document structure.  Extractors only *read*; cleaning is done in
``preprocess.py`` and splitting in ``chunking.py``.

Supported types: PDF, DOCX, XLSX, XLS, HTML/HTM, CSV, TXT and Markdown.
"""
from __future__ import annotations

import csv
import datetime as dt
import io
import logging
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Callable, Iterable, Optional

from app.config import Settings, get_settings
from app.ingestion.models import (
    Block,
    EmptyFileError,
    ExtractedDocument,
    ExtractionError,
    FileTooLargeError,
    UnsupportedFileError,
)

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS: dict[str, str] = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".xlsx": "xlsx",
    ".xls": "xls",
    ".html": "html",
    ".htm": "html",
    ".csv": "csv",
    ".txt": "txt",
    ".md": "md",
}

_WS = re.compile(r"\s+")


def _clean(text: object) -> str:
    return _WS.sub(" ", str(text or "")).strip()


# =============================================================================
# Validation
# =============================================================================
def validate_source_file(path: Path, max_size_bytes: int) -> str:
    """Check type, size and content of a file and return its normalised type.

    Rules: allowed extension, not empty, not larger than the limit.
    """
    path = Path(path)
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileError(
            f"'{path.name}': unsupported type '{ext or 'none'}'. "
            f"Allowed: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    if not path.is_file():
        raise ExtractionError(f"'{path.name}' does not exist or is not a file")
    size = path.stat().st_size
    if size == 0:
        raise EmptyFileError(f"'{path.name}' is empty")
    if size > max_size_bytes:
        raise FileTooLargeError(
            f"'{path.name}' is {size / 1_048_576:.1f} MB; limit is {max_size_bytes / 1_048_576:.0f} MB"
        )
    return SUPPORTED_EXTENSIONS[ext]


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-16", "cp1252"):
        try:
            return raw.decode(encoding)
        except UnicodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _fmt_number(value: float) -> str:
    """4244.9 -> '4244.9', 2990.0 -> '2990' (spreadsheets store ints as floats)."""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.10g}"


def _fmt_cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        return _fmt_number(value)
    if isinstance(value, dt.datetime):
        return value.date().isoformat() if value.time() == dt.time(0) else value.isoformat(sep=" ")
    if isinstance(value, dt.date):
        return value.isoformat()
    return _clean(value)


def _table_block(headers: list[str], rows: list[list[str]], **kw) -> Block:
    headers = [h if h else f"Column {i + 1}" for i, h in enumerate(headers)]
    width = len(headers)
    fixed = [(r + [""] * width)[:width] for r in rows]
    return Block(kind="table", headers=headers, rows=fixed, **kw)


# =============================================================================
# PDF
# =============================================================================
_NUMBERED = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+\S")
_BOLD_FONT = re.compile(r"bold|black|heavy|semibold|demi", re.I)
_BULLET = re.compile(r"^[\u2022\u25cf\u25aa\u25e6\u00b7\-\*]\s+")


def _inside(obj: dict, bbox: tuple) -> bool:
    x0, top, x1, bottom = bbox
    return (obj["x0"] >= x0 - 1 and obj["x1"] <= x1 + 1 and obj["top"] >= top - 1 and obj["bottom"] <= bottom + 1)


def _line_info(line: dict) -> dict:
    chars = [c for c in line.get("chars", []) if c.get("text", "").strip()]
    sizes = [round(c["size"] * 2) / 2 for c in chars] or [0.0]
    bold = sum(1 for c in chars if _BOLD_FONT.search(c.get("fontname", ""))) > len(chars) / 2 if chars else False
    return {"text": _clean(line["text"]), "size": median(sizes), "bold": bold, "top": line["top"], "bottom": line["bottom"]}


def extract_pdf(path: Path) -> ExtractedDocument:
    import pdfplumber

    try:
        pdf = pdfplumber.open(str(path))
    except Exception as exc:  # corrupt, encrypted, not a PDF
        raise ExtractionError(f"'{path.name}' could not be opened as PDF: {exc}") from exc

    with pdf:
        meta_title = _clean((pdf.metadata or {}).get("Title"))
        page_count = len(pdf.pages)
        pages: list[list[tuple[float, str, object]]] = []   # per page: (top, "line"|"table", payload)
        size_counter: Counter = Counter()

        for page in pdf.pages:
            items: list[tuple[float, str, object]] = []
            tables = page.find_tables()
            bboxes = [t.bbox for t in tables]
            text_page = page.filter(lambda o: o.get("object_type") != "char" or not any(_inside(o, b) for b in bboxes)) if bboxes else page
            for line in text_page.extract_text_lines(return_chars=True, strip=True):
                info = _line_info(line)
                if not info["text"]:
                    continue
                size_counter[info["size"]] += len(info["text"])
                items.append((info["top"], "line", info))
            for t in tables:
                grid = [[_clean(c) for c in row] for row in (t.extract() or [])]
                grid = [r for r in grid if any(r)]
                if grid:
                    items.append((t.bbox[1], "table", grid))
            items.sort(key=lambda x: x[0])
            pages.append(items)

    if not size_counter:
        raise ExtractionError(f"'{path.name}' has no extractable text (scanned PDF? OCR is not supported)")

    body_size = size_counter.most_common(1)[0][0]

    # --- classify lines: title / heading / body
    title_lines: list[str] = []
    heading_sizes: set[float] = set()
    first_page_max = max((i[2]["size"] for i in pages[0] if i[1] == "line"), default=0.0) if pages else 0.0
    for items in pages:
        for _, kind, info in items:
            if kind == "line" and _is_heading_line(info, body_size):
                heading_sizes.add(info["size"])
    title_size = first_page_max if first_page_max >= body_size * 1.7 and first_page_max > max(heading_sizes - {first_page_max}, default=0) else None
    ranked = sorted((s for s in heading_sizes if s != title_size), reverse=True)

    blocks: list[Block] = []
    for pno, items in enumerate(pages, start=1):
        para: list[str] = []
        list_items: list[str] = []
        last_bottom: Optional[float] = None
        last_kind: Optional[str] = None

        def flush_para():
            nonlocal para
            if para:
                blocks.append(Block("paragraph", _clean(" ".join(para)), page=pno))
            para = []

        def flush_list():
            nonlocal list_items
            if list_items:
                blocks.append(Block("list", "\n".join(list_items), page=pno))
            list_items = []

        for _, kind, payload in items:
            if kind == "table":
                flush_para(); flush_list()
                grid = payload  # type: ignore[assignment]
                blocks.append(_table_block(grid[0], grid[1:], page=pno, page_end=pno))
                last_bottom, last_kind = None, "table"
                continue
            info = payload  # type: ignore[assignment]
            if title_size and pno == 1 and info["size"] == title_size and info["bold"]:
                title_lines.append(info["text"])
                last_bottom, last_kind = info["bottom"], "title"
                continue
            if _is_heading_line(info, body_size):
                flush_para(); flush_list()
                m = _NUMBERED.match(info["text"])
                level = m.group(1).count(".") + 1 if m else ranked.index(info["size"]) + 1
                blocks.append(Block("heading", info["text"], level=level, page=pno))
                last_bottom, last_kind = info["bottom"], "heading"
                continue
            gap = (info["top"] - last_bottom) if last_bottom is not None else None
            tight = gap is not None and gap < info["size"] * 0.5
            if _BULLET.match(info["text"]):
                flush_para()
                list_items.append("- " + _BULLET.sub("", info["text"]))
            elif tight and list_items:
                list_items[-1] += " " + info["text"]
            elif tight and para:
                para.append(info["text"])
            else:
                flush_para(); flush_list()
                para.append(info["text"])
            last_bottom, last_kind = info["bottom"], "body"
        flush_para(); flush_list()

    blocks = _merge_split_tables(blocks)
    title = meta_title or _clean(" ".join(title_lines)) or Path(path).stem
    return ExtractedDocument(path.name, "pdf", title, blocks, page_count)


def _is_heading_line(info: dict, body_size: float) -> bool:
    text = info["text"]
    if len(text) > 150 or text.endswith((".", ",", ";")) and not _NUMBERED.match(text):
        return False
    if info["bold"] and info["size"] >= body_size * 1.12:
        return True
    return bool(info["bold"] and _NUMBERED.match(text) and len(text) <= 100 and info["size"] >= body_size)


def _merge_split_tables(blocks: list[Block]) -> list[Block]:
    """A table continued on the next page repeats its header row: join the parts."""
    merged: list[Block] = []
    for b in blocks:
        prev = merged[-1] if merged else None
        if (b.kind == "table" and prev is not None and prev.kind == "table" and prev.headers == b.headers
                and prev.page_end is not None and b.page is not None and b.page == prev.page_end + 1):
            prev.rows.extend(b.rows)
            prev.page_end = b.page_end
            continue
        merged.append(b)
    return merged


# =============================================================================
# DOCX
# =============================================================================
def _find_soffice(settings: Settings) -> Optional[str]:
    if settings.soffice_path and Path(settings.soffice_path).exists():
        return settings.soffice_path
    return shutil.which("soffice") or shutil.which("libreoffice")


def _norm_key(text: str) -> str:
    return _WS.sub(" ", text).strip().lower()


def _docx_page_texts(path: Path, settings: Settings) -> Optional[list[str]]:
    """Render the DOCX to PDF with LibreOffice and return the text of every page.

    Word files do not store page numbers; the page depends on the renderer.
    Returns None when LibreOffice is unavailable or conversion fails.
    """
    if settings.docx_page_mode == "off":
        return None
    soffice = _find_soffice(settings)
    if not soffice:
        logger.warning("LibreOffice not found: page numbers for '%s' will be empty", path.name)
        return None
    import pdfplumber

    with tempfile.TemporaryDirectory() as tmp:
        try:
            subprocess.run([soffice, "--headless", "--convert-to", "pdf", "--outdir", tmp, str(path)],
                           capture_output=True, timeout=180, check=False)
            pdf_path = Path(tmp) / (path.stem + ".pdf")
            if not pdf_path.exists():
                return None
            with pdfplumber.open(str(pdf_path)) as pdf:
                return [_norm_key(p.extract_text() or "") for p in pdf.pages]
        except (subprocess.SubprocessError, OSError) as exc:
            logger.warning("DOCX page rendering failed for '%s': %s", path.name, exc)
            return None


def _assign_pages(blocks: list[Block], page_texts: list[str]) -> None:
    """Find the page of each block by searching its first words in the rendered pages."""
    cursor = 0
    for b in blocks:
        key = _norm_key(b.text)[:60] if b.kind != "table" else _norm_key(" ".join(b.headers))[:60]
        if not key:
            b.page = b.page_end = cursor + 1
            continue
        for p in range(cursor, len(page_texts)):
            if key in page_texts[p]:
                cursor = p
                break
        b.page = cursor + 1
        b.page_end = cursor + 1


def extract_docx(path: Path, settings: Optional[Settings] = None) -> ExtractedDocument:
    from docx import Document
    from docx.oxml.ns import qn
    from docx.shared import Pt
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    settings = settings or get_settings()
    try:
        document = Document(str(path))
    except Exception as exc:
        raise ExtractionError(f"'{path.name}' could not be opened as DOCX: {exc}") from exc

    blocks: list[Block] = []
    list_items: list[str] = []
    title = _clean(document.core_properties.title)
    big_title: list[str] = []

    def flush_list():
        nonlocal list_items
        if list_items:
            blocks.append(Block("list", "\n".join(list_items)))
        list_items = []

    for child in document.element.body.iterchildren():
        if child.tag == qn("w:p"):
            p = Paragraph(child, document)
            text = _clean(p.text)
            if not text:
                continue
            style = (p.style.name if p.style is not None else "") or ""
            if style.lower().startswith("heading"):
                flush_list()
                digits = re.findall(r"\d+", style)
                blocks.append(Block("heading", text, level=int(digits[0]) if digits else 1))
            elif style.lower() == "title" or any(r.font.size and r.font.size >= Pt(20) for r in p.runs):
                big_title.append(text)       # document title line, not a section
            elif "list" in style.lower():
                list_items.append("- " + text)
            else:
                flush_list()
                blocks.append(Block("paragraph", text))
        elif child.tag == qn("w:tbl"):
            flush_list()
            table = Table(child, document)
            grid: list[list[str]] = []
            for row in table.rows:
                cells, seen = [], set()
                for cell in row.cells:
                    if id(cell._tc) in seen:     # merged cell repeated by python-docx
                        continue
                    seen.add(id(cell._tc))
                    cells.append(_clean(cell.text))
                if any(cells):
                    grid.append(cells)
            if grid:
                blocks.append(_table_block(grid[0], grid[1:]))
    flush_list()

    page_texts = _docx_page_texts(path, settings)
    page_count = None
    if page_texts:
        _assign_pages(blocks, page_texts)
        page_count = len(page_texts)
    title = title or _clean(" ".join(big_title)) or path.stem
    return ExtractedDocument(path.name, "docx", title, blocks, page_count)


# =============================================================================
# Spreadsheets (XLSX / XLS) and CSV
# =============================================================================
class _Row:
    __slots__ = ("number", "cells", "bold", "size")

    def __init__(self, number: int, cells: list[str], bold: bool = False, size: Optional[float] = None):
        self.number, self.cells, self.bold, self.size = number, cells, bold, size


def _sheet_to_blocks(sheet_name: str, rows: Iterable[_Row]) -> list[Block]:
    """Turn spreadsheet rows into blocks.

    * a blank row ends the current table;
    * a row with a single filled cell is a label (if bold) or a text line;
    * the first multi-cell row of a table is its header row.
    """
    blocks = [Block("heading", sheet_name, level=1, sheet=sheet_name)]
    headers: Optional[list[str]] = None
    body: list[list[str]] = []
    start: Optional[int] = None

    def flush():
        nonlocal headers, body, start
        if headers is not None:
            blocks.append(_table_block(headers, body, sheet=sheet_name, row_start=start))
        headers, body, start = None, [], None

    for row in rows:
        filled = [c for c in row.cells if c != ""]
        if not filled:
            flush()
            continue
        if len(filled) == 1:
            flush()
            text = filled[0]
            if row.bold and (row.size or 0) < 13:
                blocks.append(Block("heading", text, level=2, sheet=sheet_name))
            else:
                blocks.append(Block("paragraph", text, sheet=sheet_name))
            continue
        cells = list(row.cells)
        while cells and cells[-1] == "":
            cells.pop()
        if headers is None:
            headers = cells
        else:
            if start is None:
                start = row.number
            body.append(cells)
    flush()
    return blocks


def extract_xlsx(path: Path) -> ExtractedDocument:
    import openpyxl

    try:
        wb = openpyxl.load_workbook(str(path), data_only=True)
    except Exception as exc:
        raise ExtractionError(f"'{path.name}' could not be opened as XLSX: {exc}") from exc
    blocks: list[Block] = []
    for ws in wb.worksheets:
        if ws.sheet_state != "visible":
            continue
        rows = []
        for r_idx, row in enumerate(ws.iter_rows(), start=1):
            cells = [_fmt_cell(c.value) for c in row]
            first = next((c for c in row if c.value not in (None, "")), None)
            font = first.font if first is not None else None
            rows.append(_Row(r_idx, cells, bool(font and font.bold), float(font.sz) if font and font.sz else None))
        blocks.extend(_sheet_to_blocks(ws.title, rows))
    wb.close()
    return ExtractedDocument(path.name, "xlsx", path.stem, blocks, None)


def extract_xls(path: Path) -> ExtractedDocument:
    import xlrd

    try:
        try:
            book = xlrd.open_workbook(str(path), formatting_info=True)
        except NotImplementedError:      # some files do not support formatting info
            book = xlrd.open_workbook(str(path))
    except Exception as exc:
        raise ExtractionError(
            f"'{path.name}' could not be opened as XLS ({exc}). If it is really an .xlsx/.html file, rename it."
        ) from exc
    has_fmt = bool(getattr(book, "xf_list", None))
    blocks: list[Block] = []
    for sheet in book.sheets():
        if sheet.visibility != 0:
            continue
        rows = []
        for r in range(sheet.nrows):
            cells, bold, size = [], False, None
            for c in range(sheet.ncols):
                cell = sheet.cell(r, c)
                if cell.ctype == xlrd.XL_CELL_DATE:
                    value = _fmt_cell(xlrd.xldate_as_datetime(cell.value, book.datemode))
                elif cell.ctype in (xlrd.XL_CELL_EMPTY, xlrd.XL_CELL_BLANK):
                    value = ""
                elif cell.ctype == xlrd.XL_CELL_NUMBER:
                    value = _fmt_number(cell.value)
                elif cell.ctype == xlrd.XL_CELL_BOOLEAN:
                    value = "TRUE" if cell.value else "FALSE"
                elif cell.ctype == xlrd.XL_CELL_ERROR:
                    value = ""
                else:
                    value = _clean(cell.value)
                cells.append(value)
                if value and not size and has_fmt:
                    font = book.font_list[book.xf_list[cell.xf_index].font_index]
                    bold, size = bool(font.bold), font.height / 20.0
            rows.append(_Row(r + 1, cells, bold, size))
        blocks.extend(_sheet_to_blocks(sheet.name, rows))
    return ExtractedDocument(path.name, "xls", path.stem, blocks, None)


def extract_csv(path: Path) -> ExtractedDocument:
    text = _read_text(path)
    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel
    rows = [[_clean(c) for c in r] for r in csv.reader(io.StringIO(text), dialect)]
    rows = [r for r in rows if any(r)]
    if not rows:
        raise EmptyFileError(f"'{path.name}' has no rows")
    blocks = [_table_block(rows[0], rows[1:], row_start=2)]
    return ExtractedDocument(path.name, "csv", path.stem, blocks, None)


# =============================================================================
# HTML
# =============================================================================
_HEADING_TAGS = {f"h{i}": i for i in range(1, 7)}
_CONTAINERS = {"div", "section", "article", "main", "body", "header", "footer", "aside", "details", "form", "figure", "center"}
_PARAGRAPH_TAGS = {"p", "pre", "blockquote", "dd", "dt", "figcaption", "summary", "address"}
_SKIP_TAGS = {"script", "style", "noscript", "template", "svg", "nav", "iframe", "head"}


def extract_html(path: Path) -> ExtractedDocument:
    from bs4 import BeautifulSoup, NavigableString, Tag

    soup = BeautifulSoup(_read_text(path), "lxml")
    title = _clean(soup.title.get_text()) if soup.title else ""
    for tag in soup.find_all(list(_SKIP_TAGS - {"head"})):
        tag.decompose()
    blocks: list[Block] = []
    inline: list[str] = []

    def flush_inline():
        nonlocal inline
        text = _clean(" ".join(inline))
        if text:
            blocks.append(Block("paragraph", text))
        inline = []

    def table_block(table: Tag) -> Optional[Block]:
        grid = []
        for tr in table.find_all("tr"):
            cells = [_clean(c.get_text(" ")) for c in tr.find_all(["th", "td"])]
            if any(cells):
                grid.append(cells)
        return _table_block(grid[0], grid[1:]) if grid else None

    def walk(node: Tag):
        for child in node.children:
            if isinstance(child, NavigableString):
                if child.strip():
                    inline.append(str(child))
                continue
            if not isinstance(child, Tag):
                continue
            name = child.name
            if name in _HEADING_TAGS:
                flush_inline()
                text = _clean(child.get_text(" "))
                if text:
                    blocks.append(Block("heading", text, level=_HEADING_TAGS[name]))
            elif name == "table":
                flush_inline()
                tb = table_block(child)
                if tb:
                    blocks.append(tb)
            elif name in ("ul", "ol"):
                flush_inline()
                items = ["- " + _clean(li.get_text(" ")) for li in child.find_all("li") if _clean(li.get_text(" "))]
                if items:
                    blocks.append(Block("list", "\n".join(items)))
            elif name in _PARAGRAPH_TAGS:
                flush_inline()
                text = _clean(child.get_text(" "))
                if text:
                    blocks.append(Block("paragraph", text))
            elif name in _CONTAINERS:
                flush_inline()
                walk(child)
                flush_inline()
            elif name == "br":
                continue
            else:
                inline.append(child.get_text(" "))

    walk(soup.body or soup)
    flush_inline()
    if not title:
        first_h1 = next((b.text for b in blocks if b.kind == "heading" and b.level == 1), "")
        title = first_h1 or path.stem
    # the document's own <h1> repeats the title; it is not a section
    blocks = [b for b in blocks if not (b.kind == "heading" and b.level == 1 and _norm_key(b.text) == _norm_key(title))]
    return ExtractedDocument(path.name, "html", title, blocks, None)


# =============================================================================
# TXT
# =============================================================================
_UNDERLINE = re.compile(r"^(={3,}|-{3,}|~{3,})\s*$")
_TXT_NUMBERED = re.compile(r"^(\d+(?:\.\d+)*)[.)]?\s+[A-Za-z].{0,90}$")
_TXT_CAPS = re.compile(r"^[A-Z0-9][A-Z0-9 ,&/()\-:'.]{3,80}$")
_TXT_BULLET = re.compile(r"^\s*[-*\u2022+]\s+(.*\S)\s*$")


def extract_txt(path: Path) -> ExtractedDocument:
    lines = _read_text(path).replace("\r\n", "\n").replace("\r", "\n").split("\n")
    blocks: list[Block] = []
    title = ""
    para: list[str] = []
    items: list[str] = []

    def flush():
        nonlocal para, items
        if para:
            blocks.append(Block("paragraph", _clean(" ".join(para))))
        if items:
            blocks.append(Block("list", "\n".join("- " + _clean(i) for i in items)))
        para, items = [], []

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
        prev_blank = i == 0 or not lines[i - 1].strip()
        if not stripped:
            flush(); i += 1; continue
        if nxt and _UNDERLINE.match(nxt) and not _UNDERLINE.match(stripped):
            flush()
            char = nxt[0]
            if char == "=" and not title:
                title = _clean(stripped)
            else:
                blocks.append(Block("heading", _clean(stripped), level={"=": 1, "-": 2, "~": 3}[char]))
            i += 2; continue
        next_blank = not nxt
        if prev_blank and next_blank and not _TXT_BULLET.match(line):
            m = _TXT_NUMBERED.match(stripped)
            if m and not stripped.endswith((".", ":")):
                flush(); blocks.append(Block("heading", _clean(stripped), level=m.group(1).count(".") + 2)); i += 1; continue
            if _TXT_CAPS.match(stripped) and sum(ch.isalpha() for ch in stripped) >= 4:
                flush(); blocks.append(Block("heading", _clean(stripped), level=2)); i += 1; continue
        bullet = _TXT_BULLET.match(line)
        if bullet:
            if para:
                blocks.append(Block("paragraph", _clean(" ".join(para)))); para = []
            items.append(bullet.group(1))
        elif items and (line.startswith((" ", "\t"))):
            items[-1] += " " + stripped
        else:
            if items:
                blocks.append(Block("list", "\n".join("- " + _clean(x) for x in items))); items = []
            para.append(stripped)
        i += 1
    flush()
    return ExtractedDocument(path.name, "txt", title or path.stem, blocks, None)


# =============================================================================
# Markdown
# =============================================================================
_FRONT_MATTER = re.compile(r"\A---\s*\n.*?\n---\s*\n", re.S)


def _inline_text(token) -> str:
    if not token.children:
        return _clean(token.content)
    parts = []
    for ch in token.children:
        if ch.type in ("text", "code_inline"):
            parts.append(ch.content)
        elif ch.type in ("softbreak", "hardbreak"):
            parts.append(" ")
    return _clean("".join(parts))


def extract_md(path: Path) -> ExtractedDocument:
    from markdown_it import MarkdownIt

    text = _FRONT_MATTER.sub("", _read_text(path), count=1)
    tokens = MarkdownIt("commonmark").enable("table").parse(text)
    blocks: list[Block] = []
    title = ""
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t.type == "heading_open":
            level, content = int(t.tag[1]), _inline_text(tokens[i + 1])
            if level == 1 and not title:
                title = content
            elif content:
                blocks.append(Block("heading", content, level=level))
            i += 3
        elif t.type == "paragraph_open":
            content = _inline_text(tokens[i + 1])
            if content:
                blocks.append(Block("paragraph", content))
            i += 3
        elif t.type in ("bullet_list_open", "ordered_list_open"):
            j, items = i + 1, []
            while not (tokens[j].type == t.type.replace("_open", "_close") and tokens[j].level == t.level):
                if tokens[j].type == "inline":
                    items.append("- " + _inline_text(tokens[j]))
                j += 1
            if items:
                blocks.append(Block("list", "\n".join(items)))
            i = j + 1
        elif t.type == "table_open":
            j, grid, row = i + 1, [], None
            while tokens[j].type != "table_close":
                if tokens[j].type == "tr_open":
                    row = []
                elif tokens[j].type == "tr_close" and row is not None:
                    grid.append(row)
                elif tokens[j].type == "inline" and row is not None:
                    row.append(_inline_text(tokens[j]))
                j += 1
            if grid:
                blocks.append(_table_block(grid[0], grid[1:]))
            i = j + 1
        elif t.type in ("fence", "code_block"):
            content = t.content.strip()
            if content:
                blocks.append(Block("paragraph", content))
            i += 1
        else:
            i += 1
    return ExtractedDocument(path.name, "md", title or path.stem, blocks, None)


# =============================================================================
# Dispatch
# =============================================================================
EXTRACTORS: dict[str, Callable[..., ExtractedDocument]] = {
    "pdf": extract_pdf,
    "docx": extract_docx,
    "xlsx": extract_xlsx,
    "xls": extract_xls,
    "html": extract_html,
    "csv": extract_csv,
    "txt": extract_txt,
    "md": extract_md,
}


def extract(path: Path, settings: Optional[Settings] = None) -> ExtractedDocument:
    """Validate ``path`` and read it with the extractor for its type."""
    settings = settings or get_settings()
    path = Path(path)
    file_type = validate_source_file(path, settings.max_file_size_bytes)
    func = EXTRACTORS[file_type]
    try:
        doc = func(path, settings) if file_type == "docx" else func(path)
    except Exception as exc:
        if isinstance(exc, (ExtractionError, EmptyFileError)):
            raise
        raise ExtractionError(f"'{path.name}' could not be read as {file_type}: {exc}") from exc
    if not doc.blocks:
        raise EmptyFileError(f"'{path.name}' contains no readable content")
    return doc
