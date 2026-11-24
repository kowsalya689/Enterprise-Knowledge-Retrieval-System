# Design: a real file-format check at upload

**Limit chosen** (from the "Not built yet" list): *a real file-format check.* Today a file is accepted by its **name** only.

## 1. The problem

`validate_source_file` (`app/ingestion/extractors.py`) and `DocumentService.upload` (`app/services/documents.py`) check the extension, the size and "not empty". Nothing looks at the file's **content**. I tested the running API with six files:

| Upload | Result today | Why it is a problem |
|---|---|---|
| `invoice.pdf` (a Windows program) | 422, after the file was saved and opened | Late rejection; the message shows the internal temp name `.2d29…__invoice.pdf` |
| `report.docx` (plain text) | 422 | The message shows the **server path** `/tmp/…/uploads/` |
| `data.xls` (really an .xlsx) | 422, "rename it" | Works, but only after parsing started |
| `blob.txt` (256 binary byte values repeated) | **201 indexed, 1 chunk** | Binary garbage becomes searchable text and is sent to the LLM |
| `page.txt` (HTML with `<script>`) | **201 indexed** | Wrong type accepted silently |

Text types (`.txt .md .csv .html`) have **no** content check at all.

## 2. Proposed change

Add `sniff_file_type(path) -> str | None` that reads the first bytes and decides the real type:

| Declared | Accept only if |
|---|---|
| `.pdf` | starts with `%PDF-` (first 1 KB) |
| `.docx` / `.xlsx` | ZIP signature `PK\x03\x04` **and** contains `word/document.xml` / `xl/workbook.xml` |
| `.xls` | OLE2 signature `D0 CF 11 E0 A1 B1 1A E1` |
| `.txt .md .csv .html` | first 8 KB has **no NUL byte**, decodes as UTF-8 (or cp1252), under 2 % control characters; and does not start with a binary signature (`%PDF`, `PK`, `MZ`, OLE2) |

If the real type differs from the extension, reject with **415** `unsupported_file_type` and a safe message: *"The content of 'invoice.pdf' is not a valid PDF."* Never include temp names or paths. An HTML-looking `.txt` is accepted but logged as a warning (no false rejects of legitimate text).

## 3. Files to change

| File | Change |
|---|---|
| `app/ingestion/extractors.py` | add `sniff_file_type`; call it at the end of `validate_source_file` |
| `app/services/documents.py` | run the check right after `_save`, **before** `ingest_file`; map errors to the original file name |
| `app/ingestion/indexer.py` | use the original file name in `rejected:` messages, not the temp file |
| `tests/test_ingestion.py`, `tests/test_api.py` | new tests (below) |
| `README.md` | one line in the upload rules |

No change to chunking, embeddings, Qdrant or the cache.

## 4. How to test it

| Test | Expected |
|---|---|
| Each of the 11 files in `data/raw/` | still accepted (no false rejects) |
| The six probe files above | the first five get 415 or 422 with a clean message; `blob.txt` is rejected; `page.txt` accepted with a warning |
| Extension renamed (`DOC-002.pdf` → `.docx`) | 415 |
| Message check | response text contains neither `/tmp` nor `.` + 32 hex characters |
| Empty file / 51 MB file | unchanged (422 / 413) |
| UTF-8 file with Tamil text; cp1252 file with `€` | accepted |
| After a rejection | nothing in Qdrant, nothing left in `data/documents/`, cache version unchanged |

**Done when** all tests pass, `python -m pytest -q` is green, and re-uploading the 11 dataset files gives the same chunk counts as before.

**Out of scope:** virus scanning, OCR of scanned PDFs, macro detection in Office files.
