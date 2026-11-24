# Enterprise Knowledge Retrieval - RAG with FastAPI, Qdrant, Redis and Ollama

This version adds, on top of the existing ingestion code (extraction, chunking, metadata, embeddings, Qdrant storage):

1. **FastAPI REST API** with Swagger UI at `/docs`;
2. **a demonstration of how documents and embeddings are stored in Qdrant**;
3. **the complete retrieval process** from user query to cited answer (search, access control, evidence check, Ollama, citations, Redis cache).

Final task (this version): the **0.30 cutoff**, an **exact + semantic Redis cache** that never stores "not found", **two web pages** (`/chat`, `/upload`), the **whole system in Docker**, a **10-question test** and a **one-page design**. Full documentation: `docs/Enterprise_Knowledge_Retrieval_Documentation.docx` / `.pdf`; workflow: `docs/Enterprise_Knowledge_Retrieval_Workflow.pdf`; one-page design: `docs/One_Page_Design_File_Format_Check.pdf`. Short notes: `docs/final_task_notes.md`; the step-by-step workflow is in `docs/workflow_step_by_step.md` (`docs/workflow.pdf` is the earlier version).

---

## 1. Analysis of the existing project

| Area | State before this change | What was done |
|---|---|---|
| Ingestion (8 formats), preprocessing, chunking, metadata | Existing and tested (111 tests) | **Reused unchanged.** `Indexer.ingest_file` got one optional argument (`record`) so the API can pass upload metadata |
| Embeddings (Nomic v2 MoE) | Existing | Reused. Added `ThreadSafeEmbedder` (one request at a time uses the model) |
| Qdrant storage | Existing: collection, indexes, upsert, inspection CLI | Extended: filtered **search**, collection info, per-document listing with access filter, **model guard** (no mixing of embedding models) |
| Redis cache | **Missing** | Added `app/cache/redis_cache.py` |
| Ollama integration | **Missing** | Added `app/rag/llm.py` |
| Retrieval, evidence check, citations | **Missing** | Added `app/rag/pipeline.py`, `citations.py`, `prompts.py` |
| Access control | Access tags stored, no users | Added `app/security/permissions.py` (API key -> role -> tags) |
| FastAPI | **Missing** | Added `app/main.py`, `app/api/*`, `app/services/documents.py`, `app/dependencies.py` |
| Docker | **Missing** | Added `Dockerfile`, `docker-compose.yml` |

The existing folder structure was kept. Mapping to the structure your mentor's prompt suggests: `routes/` = `app/api/`, `schemas/` = `app/api/schemas.py`, `services/` = `app/services/`, `models/` = `app/ingestion/models.py`, `database/` = `app/retrieval/vector_store.py`, `core/` = `app/config.py` + `app/dependencies.py`.

## 2. Project structure

```
enterprise-knowledge-retrieval/
|-- app/
|   |-- main.py                     FastAPI app, error handlers, Swagger
|   |-- config.py                   all settings (.env)
|   |-- dependencies.py             service wiring, API-key dependency
|   |-- api/
|   |   |-- schemas.py              Pydantic request/response models
|   |   |-- status.py               GET /health, GET /collection/info
|   |   |-- upload.py               POST /documents/upload, GET /documents
|   |   |-- query.py                POST /query
|   |   `-- pages.py                GET /chat, /upload (HTML pages)
|   |-- services/documents.py       upload rules: safe name, size, tags, versions, cache invalidation
|   |-- security/permissions.py     API keys, roles, access tags
|   |-- ingestion/                  (existing) extractors, preprocess, chunking, metadata, indexer
|   |-- retrieval/
|   |   |-- embeddings.py           (existing) Nomic v2 MoE
|   |   `-- vector_store.py         Qdrant: storage + search + inspection
|   |-- rag/
|   |   |-- pipeline.py             question -> answer
|   |   |-- llm.py                  Ollama client
|   |   |-- prompts.py              grounded-answer prompt
|   |   `-- citations.py            context blocks and citations
|   |-- cache/redis_cache.py        exact + semantic cache, TTL, invalidation
|   |-- web/                        chat.html, upload.html, static/ (the two pages)
|   `-- evaluation/retrieval_eval.py  Recall@k, MRR, leaks, threshold suggestion
|-- data/  raw/ (11 documents)  processed/document_inventory.csv  roles.json  evaluation_questions.csv
|-- tests/                          184 tests (see section 9)
|-- docs/workflow.pdf
|-- requirements.txt  .env.example  Dockerfile  docker-compose.yml  pytest.ini
```

## 3. Setup (once)

```
python -m venv .venv                      # Windows: .venv\Scripts\Activate.ps1   macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env                    # macOS/Linux: cp .env.example .env
```

Edit `.env`:

| Setting | What to do |
|---|---|
| `API_KEYS` | Replace the placeholders with random keys: `python -c "import secrets; print(secrets.token_urlsafe(24))"`. Format `key:role,key:role` (roles in `data/roles.json`). **No key is built in.** |
| `QDRANT_URL`, `QDRANT_API_KEY` | Free Qdrant Cloud cluster (cloud.qdrant.io -> create free cluster -> copy URL with `:6333` and the API key). Or `QDRANT_LOCAL_PATH=./qdrant_data` for a local folder. |
| `OLLAMA_MODEL` | Install Ollama (ollama.com), then `ollama pull llama3.2:3b` (or another model you prefer). |
| `REDIS_URL` | Redis on `localhost:6379` (on Windows run `docker run -p 6379:6379 redis:7-alpine`), or leave empty / `CACHE_ENABLED=false` to run without a cache. |

## 4. Run it

```
# 1. put the 11 documents into Qdrant (first run downloads the embedding model, about 1.9 GB)
python -m app.ingestion.indexer

# 2. check the stored points (section 6)
python -m app.retrieval.vector_store --stats

# 3. calibrate the "not found" cutoff, then put the suggested value in .env as RELEVANCE_THRESHOLD
python -m app.evaluation.retrieval_eval

# 4. start the API
uvicorn app.main:app --reload
```

Open **http://localhost:8000/docs** (Swagger UI). The model is loaded at start-up, so the first start takes longer.

**Swagger demonstration, step by step**

1. `GET /health` -> **Try it out** -> **Execute**. Every component shows `ok` (Redis shows `disabled` without a cache).
2. Click **Authorize**, paste one of your keys (for example the `employee` key), **Authorize**.
3. `GET /documents` lists only the documents that role may see (5 of 11 for `employee`).
4. `POST /query` with `{"question": "What is the hotel limit per night in Tier 1 cities for grade G4 to G6?"}`. Read `answer`, `citations` (document name, section, page), `cached`.
5. Execute the same question again: `cached` is `true` and `timings_ms` is about a millisecond.
6. Ask `What are the minimum, midpoint and maximum salary for grade G5?` with the `employee` key (no citation from the compensation document), then Authorize with the `hr` key and ask again: the answer cites `DOC-011`, page 2.
7. Ask something that is not in the documents: the response is `status: not_found`, `answer: "Not found in the available documents"`.
8. `POST /documents/upload`: choose a file, `access_tags` = `all_employees`, department. Then ask a question about its content.
9. `POST /query` with `"debug": true` shows every retrieved chunk with its score.

## 5. API reference

All protected endpoints need the header `X-API-Key: <your key>`. The two web pages (`/chat`, `/upload`) have no key box: they send `X-Web-UI: 1` and the server applies the role `WEB_UI_ROLE` (default `employee`; empty switches the pages off). `GET /pipeline/info` shows the embedding model and chunking settings, and every upload response says which embedding model and chunk settings were used. Errors always look like `{"error": {"code": "...", "message": "..."}}`.

| Endpoint | Auth | Purpose | Responses |
|---|---|---|---|
| `GET /health` | no | Status of API, embedding model, Qdrant, Redis, Ollama | 200 `ok`/`degraded`; 503 if Qdrant or the embedding model is unavailable |
| `POST /documents/upload` | yes | multipart: `file`, `access_tags` (comma list), `department`. Validates, extracts, chunks, embeds, stores | 201 `indexed` or `unchanged`; 401, 403 (tag not held), 413 (> 50 MB), 415 (type), 422 (empty file, bad tags, unreadable) |
| `GET /documents` | yes | Documents the caller may see | 200 |
| `POST /query` | yes | `{question, top_k?, debug?}` | 200 `answered`/`not_found`; 401, 422, 503 (Ollama or Qdrant down) |
| `GET /collection/info?sample=N` | yes | Collection facts, indexed fields, visible documents, sample points | 200; 401, 503 |

Accepted upload types: PDF, DOCX, TXT (as requested) and also XLSX, XLS, HTML, CSV, Markdown, because the ingestion pipeline supports them.

Example `POST /query` response (shape). *This sample came from the offline demo run (section 9) with stand-ins for the models, so the answer wording is not real model output:*

```json
{
  "status": "answered",
  "answer": "<answer written by the model, with [1] markers>",
  "citations": [{
    "number": 1, "document_id": "DOC-003",
    "document_name": "DOC-003_Finance_Travel_Reimbursement_Policy.pdf",
    "document_title": "Travel and Expense Reimbursement Policy",
    "section": "3. Eligibility by Grade", "page": 2, "sheet": null,
    "chunk_id": "DOC-003-C004", "score": 0.4544, "snippet": "Entitlements depend on the employee's grade ..."
  }],
  "citation_source": "model_markers", "reason": null, "cached": false,
  "access_scope": "06d1ce227f9f79d5",
  "timings_ms": {"embed": 0.1, "search": 68.1, "llm": 0.1, "total": 79.4}
}
```

Not found: `{"status": "not_found", "answer": "Not found in the available documents", "citations": [], "reason": "insufficient_evidence", ...}`. `reason` is `insufficient_evidence` (no chunk above the cutoff, no LLM call), `model_not_found` (the model replied NOT_FOUND) or `no_citation` (only with `STRICT_CITATIONS=true`).

## 6. Qdrant storage demonstration

After `python -m app.ingestion.indexer` (or an upload) every chunk is one **point**:

```json
{
  "id": "<UUID from document id + version + chunk number>",
  "vector": [768 numbers],
  "payload": {
    "document_id": "DOC-003",
    "file_name": "DOC-003_Finance_Travel_Reimbursement_Policy.pdf",
    "document_title": "Travel and Expense Reimbursement Policy",
    "text": "Item: Hotel limit per night (Tier 1 cities); G1 to G3: INR 4,000; ...",
    "section_path": "3. Eligibility by Grade",
    "page_number": 2,
    "department": "Finance",
    "access_tags": ["all_employees"],
    "document_version": "1.4",
    "embedding_model": "nomic-ai/nomic-embed-text-v2-moe",
    "embedding_dim": 768,
    "chunk_id": "DOC-003-C004", "file_sha256": "...", "ingested_at": "..."
  }
}
```

Look at them in three ways:

| Way | How |
|---|---|
| Dashboard | cloud.qdrant.io -> your cluster -> dashboard -> Collections -> `enterprise_kb` -> points and payload |
| API | `GET /collection/info?sample=3` (collection name, point count, vector size, distance, indexed payload fields, your visible documents, sample points) |
| Command line | `python -m app.retrieval.vector_store --stats` and `python -m app.retrieval.vector_store --document DOC-003 --sample 2` |

Safety rules in the code (`vector_store.py`): the collection is created automatically; the vector size is checked against the model; every point stores `embedding_model`, and adding vectors from another model to the same collection raises `EmbeddingModelMismatchError`; point ids are deterministic so re-uploading never duplicates; payload indexes exist for `access_tags`, `document_id`, `file_name`, `file_type`, `department`.

## 7. Retrieval demonstration (query to answer)

| # | Step in the mentor's list | Code |
|---|---|---|
| 1 | Receive the question through FastAPI | `app/api/query.py` |
| 2 | Generate the query embedding | `RagPipeline.ask` -> `embedder.embed_query` (`search_query:` prefix) |
| 3 | Vector similarity search | `QdrantStore.search` |
| 4 | Retrieve top relevant chunks | `top_k` (default 8) |
| 5 | Access-control filtering | `access_tags` filter **inside** the Qdrant query; search without tags is refused |
| 6 | Check for sufficient evidence | `RELEVANCE_THRESHOLD` and `MIN_EVIDENCE_CHUNKS` |
| 7 | Pass context to Ollama | `build_context` + `OllamaClient.generate` |
| 8 | Grounded answer | prompt: answer only from numbered blocks, cite `[n]`, otherwise `NOT_FOUND` |
| 9 | Answer with document name, page, citations | `select_citations` (from stored metadata, not from model text) |
| 10 | "Not found in the available documents" | returned when evidence is missing or the model says `NOT_FOUND` |

Redis sits in front of steps 3-9: an **exact** match (normalised text, no model call) and then a **semantic** match (question embeddings at or above `SEMANTIC_CACHE_THRESHOLD`, same numbers in both questions). Both are inside the caller's access scope, so users with different permissions never share an answer. Only grounded answers are cached (TTL `CACHE_TTL_ANSWER_SECONDS`); **"not found" is never cached**. Entries are invalidated when a document is uploaded or replaced. The response shows `cached`, `cache_match` (`exact`/`semantic`), `cache_similarity` and `top_score`.

**The cutoff must be calibrated.** `RELEVANCE_THRESHOLD=0.30` (the value set for the final task) is only a starting value. Run `python -m app.evaluation.retrieval_eval` after ingestion: it prints Recall@5/10, MRR, access-control leaks (must be 0) and a suggested threshold from your 66 labelled questions.

## 8. Docker (Qdrant, Redis, Ollama and the app)

```
cp .env.example .env            # set API_KEYS; the service addresses are set by docker-compose.yml
docker compose up --build       # qdrant + redis + ollama + app; ollama-pull downloads the model, then exits
docker compose exec app python -m app.ingestion.indexer     # load the 11 documents, once
```

Open **http://localhost:8000/chat** (ask), **/upload** (upload), **/docs** (Swagger); Qdrant dashboard at http://localhost:6333/dashboard. The first start downloads the embedding model into a volume (about 1.9 GB) and the health check allows 10 minutes for it. To use Qdrant Cloud instead, remove the `qdrant` service and the `QDRANT_*` lines under `app.environment`. DOCX page numbers need LibreOffice: build with `INSTALL_LIBREOFFICE: "true"`.

**Only the services in Docker, the app on your computer:** `docker compose up qdrant redis ollama ollama-pull`, then in `.env` set `QDRANT_URL=http://localhost:6333`, `REDIS_URL=redis://localhost:6379/0`, `OLLAMA_BASE_URL=http://localhost:11434` and run `uvicorn app.main:app --reload`.

**10-question test** (after the documents are loaded): `python scripts/run_ten_questions.py --api-key <employee key>` writes `docs/ten_question_results.md`.

## 9. Testing

```
python -m pytest -q                      # 206 passed, 5 skipped (integration tests need real servers)

# with a real Redis running (checked: 208 passed, 3 skipped):
REDIS_TEST_URL=redis://localhost:6379/1 python -m pytest -q
# with a real Qdrant server as well, the 3 remaining integration tests also run:
QDRANT_TEST_URL=http://localhost:6333 REDIS_TEST_URL=redis://localhost:6379/1 python -m pytest -q
python -m tests.demo_server              # OFFLINE demo of the API at http://localhost:8001/docs
                                         # (keys demo-employee, demo-hr, demo-manager, demo-finance;
                                         #  stand-ins for the embedding model and Ollama)
```

| Mentor's test item | Test file |
|---|---|
| FastAPI health endpoint | `test_api.py` (`test_health_*`, Swagger/OpenAPI test) |
| Document upload | `test_api.py` (PDF, DOCX, TXT, rejections, size limit, path traversal, versions) |
| Document processing and chunking | `test_ingestion.py` (95 tests, every format, 62 labelled sections/pages/texts) |
| Embedding generation | `test_retrieval.py` (prefixes, normalisation, dimension) |
| Qdrant vector storage | `test_retrieval.py`, `test_search.py`, `test_integration.py` (real server) |
| Qdrant similarity search | `test_search.py` |
| Retrieval with metadata and citations | `test_rag_pipeline.py`, `test_api.py` |
| Redis cache behaviour (exact, semantic, TTL, never cache not-found) | `test_cache.py`, `test_rag_pipeline.py`, `test_integration.py` (real Redis) |
| Guardrails | `test_guardrails.py` |
| Web pages | `test_api.py` (`test_both_pages_*`) |
| Access-control filtering | `test_permissions.py`, `test_rag_pipeline.py`, `test_api.py` |
| Not-found response | `test_rag_pipeline.py`, `test_api.py` |
| Ollama client | `test_ollama_client.py` (HTTP simulated) |

**What was and was not tested.** Tested with real software: a real Qdrant 1.20 server (collection, payload indexes, filtered search) and a real Redis 7 server (TTL, sets, invalidation). The embedding model and Ollama were replaced by test doubles (`tests/doubles.py`): a bag-of-words vectoriser and an extractive answerer. This proves the code paths, filtering, caching and API behaviour, **not** the quality of Nomic embeddings or of your Ollama model. Not tested here: the real Nomic model, a running Ollama, Qdrant Cloud, Docker (`docker compose up` was not run), and a Windows machine.

## 10. How this answers the mentor's feedback

| Feedback | Where it is satisfied |
|---|---|
| Integrate FastAPI to expose the functionality | Section 5: five endpoints, Pydantic schemas, error handling, Swagger UI at `/docs` with an Authorize button |
| Demonstrate how documents and embeddings are stored in Qdrant | Section 6: point structure, `GET /collection/info`, CLI, dashboard, model and size guards |
| Demonstrate how the retrieval process works | Section 7: the 10 steps mapped to code, `debug: true` shows scores and dropped chunks, `docs/workflow.pdf` has the diagrams |

## 11. Troubleshooting

| Problem | Fix |
|---|---|
| Start-up error `No vector database configured` | Set `QDRANT_URL` + `QDRANT_API_KEY` or `QDRANT_LOCAL_PATH` |
| `EmbeddingModelMismatchError` / `VectorSizeMismatchError` | The collection holds vectors of another model/size: use a new `QDRANT_COLLECTION` |
| Every protected call returns 401 | `API_KEYS` is empty or the header `X-API-Key` is missing |
| `/query` returns 503 `llm_unavailable` | Start Ollama and run `ollama pull <OLLAMA_MODEL>`; `/health` shows the reason |
| Many `not_found` answers although the answer exists | `RELEVANCE_THRESHOLD` is too high for your model: run `python -m app.evaluation.retrieval_eval` |
| Answers always cite all blocks | The model does not write `[n]` markers; use a stronger model or set `STRICT_CITATIONS=true` |
| Cache never hits | Redis is down (see `/health`) or `REDIS_URL` is empty. "Not found" answers are never cached, by design |
| Two different questions get the same cached answer | Raise `SEMANTIC_CACHE_THRESHOLD` (0.95) or set `SEMANTIC_CACHE_ENABLED=false` |
| Warning "chunk(s) exceed 512 tokens" | Lower `CHUNK_MAX_WORDS` (for example 250) and `TABLE_ROWS_PER_CHUNK` (6), re-ingest with `--force` |
| DOCX page numbers empty | Install LibreOffice and set `SOFFICE_PATH`, or accept empty DOCX pages |
