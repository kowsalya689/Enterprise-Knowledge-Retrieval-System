# Final task: what was done, how to run it, and the questions you asked

## 1. The five parts of the final task

| # | Task | Where it is |
|---|---|---|
| 1 | Query workflow: search, permission filter, **0.30 cutoff**, prompt, local LLM, citations | `app/rag/pipeline.py` (cutoff default in `app/config.py` and `.env.example`) |
| 2 | Cache: exact + semantic match, TTL, **never cache "not found"** | `app/cache/redis_cache.py`, `RagPipeline._store_in_cache` |
| 3 | Run the whole system: Qdrant and Redis in Docker, FastAPI, both pages | `docker-compose.yml`, pages `/chat` and `/upload` (`app/web/`) |
| 4 | Test with 10 questions (7 about documents, 3 off-topic) | `data/final_test_questions.csv`, `scripts/run_ten_questions.py` |
| 5 | One-page design of one improvement | `docs/improvement_design.md` |

## 2. Run it (Docker)

```
cp .env.example .env            # set API_KEYS (random keys, format key:role,key:role)
docker compose up --build       # Qdrant, Redis, Ollama, the app; downloads ~1.9 GB embedding model + the Ollama model
docker compose exec app python -m app.ingestion.indexer       # load the 11 documents (once)
```

Open `http://localhost:8000/upload` (upload page), `http://localhost:8000/chat` (ask page), `http://localhost:8000/docs` (Swagger). Qdrant dashboard: `http://localhost:6333/dashboard`. The two pages have **no key box**: they use the role `WEB_UI_ROLE` from `.env` (default `employee`, least privilege). API keys are for scripts and the Swagger page (`/docs`, button **Authorize**) only. The Docker ports are bound to `127.0.0.1`, so only your own computer can reach the app.

Then the 10-question test, with the **employee** key (the questions were chosen for that role):

```
python scripts/run_ten_questions.py --api-key <employee key>
```

It writes `docs/ten_question_results.md` and `.csv` (answer, correct?, source cited, top score, cached) and ends with a cache check. "Correct?" is an automatic first check; read the answers yourself before you submit. If an off-topic question is answered, or a document question is "not found", the top scores in the table tell you whether to move `RELEVANCE_THRESHOLD`.

## 3. How the cache behaves (and why)

| Rule | Reason |
|---|---|
| Key contains the **access scope** | an HR answer is never served to an employee, exact or semantic |
| Exact match first (normalised text) | no model call at all, about 1 ms |
| Then semantic match: similarity ≥ `SEMANTIC_CACHE_THRESHOLD` (0.90) **and** the same numbers/codes | "G4" and "G5" have almost the same embedding but different answers; the number guard stops that |
| TTL 24 h (`CACHE_TTL_ANSWER_SECONDS`) | old answers expire; also deleted when a cited document is replaced, and an index version makes older entries unreachable |
| "Not found" and errors are **not** cached | a document uploaded a minute later must be able to answer the same question |

Known limit: a semantic cache can still confuse two questions that differ in a *word* ("minimum" vs "maximum") at similarity ≥ 0.90. If you see this, raise the threshold to 0.95 or set `SEMANTIC_CACHE_ENABLED=false`.

## 4. Guardrails: which one, and are they allowed in tests?

A *guardrail* is a rule that limits what goes into the system or comes out of it. There are three places:

| Where | Guardrail | In this project |
|---|---|---|
| Input | question length, prompt-injection | length is validated (3 to 1000 characters). Injection cannot widen access because the role comes from the API key |
| Retrieval | who may see what | access-tag filter inside the Qdrant query; cache scope |
| Output | answer only from documents, cite sources, say "not found" | 0.30 cutoff, grounded prompt (`NOT_FOUND`), citations built from stored metadata |

**Which library?** Guardrails AI, NeMo Guardrails and LLM Guard exist, but they add checks that usually need another model or service. **For this final task use none**: the guardrails above are plain code, deterministic and already tested. Adding a library would add weight without being asked for.

**Are guardrails OK in test cases? Yes, and recommended.** `tests/test_guardrails.py` (6 tests) checks, without any real model: an injection attack as an employee never retrieves the salary document; `role`/`access_tags` in the request body are ignored; a wrong key gets 401; an over-long or empty question gets 422; an off-topic question is refused **without calling the LLM**; the prompt contains the grounding rules. What a test with a fake model cannot prove is that the **real** model obeys the prompt; that is what the 10-question run shows (off-topic question T10 is the hard one).

## 5. LangChain or LangSmith for testing?

They are **not alternatives**: LangChain is a framework for building LLM applications; LangSmith is a hosted service for tracing and evaluating them (it can be used without LangChain).

| | LangChain | LangSmith |
|---|---|---|
| What it is | library: chains, retrievers, agents | platform: traces, datasets, evaluations |
| Fits this project? | **No.** The project is plain Python (FastAPI, qdrant-client, redis, Ollama over HTTP). Adopting LangChain would mean rewriting working, tested code | **Optional, later.** Useful to *see* each step of a question. By default traces go to a cloud service, which conflicts with a local-first project holding private documents unless you self-host |

**For the final task use `pytest` + `scripts/run_ten_questions.py` + `app/evaluation/retrieval_eval.py`.** They test the real API, need no account, and keep data local. If you later want automated answer-quality scores, add a library such as RAGAS or DeepEval, but a 3B local model is a weak judge.

## 6. pip or uv?

Both install the same packages from `requirements.txt`. `uv` is a faster installer with the same commands; I checked that `uv` resolves this project's `requirements.txt` without errors.

| | pip | uv |
|---|---|---|
| Install | `python -m venv .venv` then `pip install -r requirements.txt` | `uv venv` then `uv pip install -r requirements.txt` |
| Speed | slower | much faster |
| In this project | used by `README.md` and the `Dockerfile` | works, but is not used by the Dockerfile |

**Recommendation: keep pip for the submission**: README, Dockerfile and the mentor's instructions all say pip, and the Docker image installs a small CPU-only PyTorch first (`pip install torch --index-url https://download.pytorch.org/whl/cpu`). Use uv on your own laptop if you want speed; install the CPU torch the same way first.

## 7. What was tested, and what was not

| Tested | How |
|---|---|
| Query workflow, cutoff, permission filter, citations, not-found | `pytest`, with a word-matching stand-in for the embedding model and an extractive stand-in for Ollama |
| Cache: exact, semantic, TTL, scope isolation, invalidation, never-cache-not-found | `pytest` with fakeredis **and** a real Redis 7 server (real TTL expiry) |
| Both pages | opened in a real browser (Chromium) against the offline demo: ask, repeat (cache hit), off-topic (not found), upload, document list |
| `docker-compose.yml` | syntax checked with `docker compose config` |

**Not tested** (not possible in the build environment): the real Nomic model, a real Ollama model, a real Qdrant server, and `docker compose up`. So the answers' quality, the real score ranges (is 0.30 a good cutoff for Nomic?) and the first Docker start are **yours to run**. If the first start fails while loading the embedding model, the likely cause is a `transformers` version that does not match the model's custom code; pin an older `transformers` in `requirements.txt`.

## API keys: why this format
`API_KEYS=key:role,key:role` — the text before the colon is the secret sent in the `X-API-Key` header; the role after it is chosen by the server and mapped to access tags in `data/roles.json`. Nobody can claim a bigger role in the request. Example keys are for local practice only: replace them with random ones (`python -c "import secrets; print(secrets.token_urlsafe(24))"`).

## LangChain / LangSmith / guardrails
Neither LangChain nor LangSmith is used (not in `requirements.txt` or `app/`). Guardrails are plain code (permission filter, 0.30 cutoff, grounded prompt, citations from metadata, length limits).
