# Workflow, step by step

Two workflows share one Qdrant collection and one embedding model:

* **Ingestion**: runs when a file is uploaded (or `python -m app.ingestion.indexer`). It prepares the knowledge.
* **Query**: runs for **every** question. It finds evidence, answers and cites.

> This file is the current version. `docs/workflow.pdf` was written before the semantic cache and the 0.30 cutoff; where they differ, this file is right.

## 1. The 12 steps from the project addendum, mapped to this code

| # | Step | What happens | Technology | Code | State |
|---|---|---|---|---|---|
| 1 | Upload and validate | Check file type, size (max 50 MB) and empty file; clean the file name | FastAPI `UploadFile` | `api/upload.py`, `services/documents.py`, `extractors.validate_source_file` | Built. Content check is **not** built (design: `docs/improvement_design.md`) |
| 2 | Extract | Read text, headings and tables; keep page, sheet and row | pdfplumber, python-docx, openpyxl, xlrd, BeautifulSoup, markdown-it | `ingestion/extractors.py` | Built for 8 formats. No OCR |
| 3 | Clean and chunk | Unicode and whitespace clean-up, remove repeated headers and footers, then **structure-aware chunking** (section 3 below) | own code | `ingestion/preprocess.py`, `chunking.py` | Built (basic cleaning) |
| 4 | Add metadata | document id, section path, page or sheet, version, department, access tags, file hash, embedding model | own code | `ingestion/metadata.py` | Built. Department is stored; only **access tags** filter results |
| 5 | Generate embeddings | Chunks to vectors with **Nomic Embed Text v2 MoE** (section 2 below); BGE-small is the baseline to compare | sentence-transformers | `retrieval/embeddings.py` | Built. Model is a `.env` setting |
| 6 | Store in Qdrant | Vectors and metadata, one **collection per embedding model**; vector size and model name are checked | qdrant-client | `retrieval/vector_store.py` | Built |
| 7 | Authenticate | `X-API-Key` → role → access tags. The role comes from the server, never from the request | FastAPI security | `security/permissions.py`, `data/roles.json` | Built |
| 8 | Check Redis cache | **Exact** match, then **semantic** match, only inside the caller's access scope and the current index, prompt and model version | Redis, numpy | `cache/redis_cache.py` | Built |
| 9 | Retrieve evidence | On a miss: embed the question, search Qdrant **with the access-tag filter inside the query**, keep chunks with score ≥ **0.30** | Qdrant | `rag/pipeline.py`, `vector_store.search` | Built |
| 10 | Evidence check | No chunk above the cutoff → **"Not found in the available documents"**, the LLM is not called | own code | `rag/pipeline.py` | Built |
| 11 | Answer and cite | Local LLM answers only from the numbered context; citations are built from stored metadata, not from model text | Ollama (`llama3.2:3b`) | `rag/llm.py`, `prompts.py`, `citations.py` | Built |
| 12 | Cache and display | Cache only grounded answers (TTL 24 h, **never** "not found"); show answer and sources | FastAPI + 2 pages | `api/query.py`, `web/chat.html`, `web/upload.html` | Built |

**Security rule (from the addendum):** the permission filter applies to **both** vector retrieval and cache reuse. Retrieval: the filter is part of the Qdrant query. Cache: the access scope (a hash of the tags) is part of every cache key, exact and semantic, so a cached HR answer can never be served to an employee.

## 2. Embeddings with Nomic Embed Text v2 ("V2")

An embedding model turns text into a list of numbers (a vector) so that texts with similar **meaning** have vectors that point in similar directions. Search then means: embed the question, find the chunk vectors closest to it (cosine similarity, 0 to 1 for normal text).

How it is used here:

| Point | Value |
|---|---|
| Model | `nomic-ai/nomic-embed-text-v2-moe` (mixture-of-experts, multilingual, runs locally, about 1.9 GB download) |
| Vector size | 768 numbers (can be cut to 256 with `EMBEDDING_DIM`; the model supports this) |
| Max input | 512 tokens, so a chunk is limited to 380 words (title path included) |
| Document prefix | `search_document: ` + title path + chunk text |
| Query prefix | `search_query: ` + question |
| Normalisation | every vector has length 1, so the dot product equals cosine similarity |

The two prefixes are **required** by Nomic: they tell the model which side of the search a text is on. Forgetting one lowers the scores. The **0.30 cutoff** is a cosine score on these vectors; a different model needs a different cutoff (calibrate with `python -m app.evaluation.retrieval_eval`).

## 3. Structure-aware chunking

Instead of cutting every N characters, the chunker follows the document's own structure, so a chunk is a meaningful unit and its citation (section, page) is exact.

| Document type | Rule |
|---|---|
| PDF, DOCX, HTML, MD, TXT | one chunk per heading or sub-heading; a table stays whole |
| XLSX, XLS, CSV | groups of 12 rows; the column names are repeated in every row (`Header: value; Header: value`) |
| Too long (> 380 words) | split at paragraph, list-item or table-row boundaries, then at sentence boundaries; only split text gets a 30-word overlap |
| Every chunk | title path in front, stable id `DOC-003-C004`, page or sheet kept |

Real example (DOC-003, travel policy, chunk `DOC-003-C004`, page 2):

```
title path : Travel and Expense Reimbursement Policy > 3. Eligibility by Grade
chunk text : Entitlements depend on the employee's grade. Grades G1 to G3 are junior staff ...
             Item: Hotel limit per night (Tier 1 cities); G1 to G3: INR 4,000; G4 to G6: INR 6,000; ...
embedded as: search_document: Travel and Expense Reimbursement Policy > 3. Eligibility by Grade
             Entitlements depend on the employee's grade ...
```

Why this matters for answers: the hotel limit question finds **this** chunk because the table row, its column names and its section title are together. With fixed-size cutting, the row could be separated from "G4 to G6" and the answer would be wrong.

## 4. The query workflow in order (what happens to one question)

```
question ─► 401 if key wrong ─► exact cache? ─yes─► answer (cached: exact)
                                    │no
                                    ▼
                         embed question (search_query:)
                                    │
                  semantic cache? ──yes (similarity ≥ 0.90, same numbers, same scope)──► answer (cached: semantic)
                                    │no
                                    ▼
              Qdrant search, access-tag filter inside the query, top 8
                                    │
                  keep chunks with score ≥ 0.30
                                    │
              none left? ──yes──► "Not found in the available documents"   (not cached, no LLM call)
                                    │no
                                    ▼
          numbered context blocks ─► Ollama ─► NOT_FOUND? ──yes──► "Not found" (not cached)
                                    │no
                                    ▼
                 citations from metadata (document, section, page, score)
                                    │
                 store in cache (TTL 24 h) ─► return answer + sources
```

## 5. Fair comparison: BGE-small-en-v1.5 (baseline) vs Nomic Embed v2

The addendum asks for both models on **identical chunks**. The chunker does not depend on the embedding model, so the chunks are identical by construction. Run the project twice, once per model, with a **separate collection** each (the code refuses to mix models in one collection).

| Setting | Run A: Nomic v2 | Run B: BGE-small baseline |
|---|---|---|
| `QDRANT_COLLECTION` | `enterprise_kb` | `enterprise_kb_bge` |
| `EMBEDDING_MODEL` | `nomic-ai/nomic-embed-text-v2-moe` | `BAAI/bge-small-en-v1.5` |
| `EMBEDDING_DIM` | `768` | *(empty = native 384)* |
| `EMBEDDING_DOC_PREFIX` | `"search_document: "` | `""` |
| `EMBEDDING_QUERY_PREFIX` | `"search_query: "` | `"Represent this sentence for searching relevant passages: "` |

For each run:

```
python -m app.ingestion.indexer --force
python -m app.evaluation.retrieval_eval        # writes data/processed/retrieval_eval_<collection>.json
```

Copy the numbers into this table for the final submission (the project could not run either model in the build environment, so these cells are empty on purpose):

| Metric (66 labelled questions) | BGE-small | Nomic v2 |
|---|---|---|
| Recall@5 | | |
| Recall@10 | | |
| MRR | | |
| Access-control leaks (must be 0) | | |
| Suggested cutoff | | |
| Answerable p10 / unanswerable p90 top score | | |

Keep one variable changing at a time: same documents, same chunks, same questions, same `top_k`. Use each model's own suggested cutoff for its own run, otherwise the comparison is unfair (BGE scores sit in a different range than Nomic scores).
