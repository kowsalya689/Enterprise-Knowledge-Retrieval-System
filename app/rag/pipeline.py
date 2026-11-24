"""The question-answering flow, from question to cited answer.

 1. exact cache lookup  key = access scope + versions + normalised question (no model call)
 2. embed the question  same model and prefix rules as the chunks
 3. semantic cache      a similar earlier question of the same access scope (same numbers)
 4. similarity search   in Qdrant, restricted to the caller's access tags (permission filter)
 5. relevance cutoff    chunks below ``relevance_threshold`` (0.30) are dropped
 6. evidence check      fewer than ``min_evidence_chunks`` left -> "Not found", no LLM call
 7. context             numbered blocks with document, section and page
 8. generation          Ollama answers from the context only (or says NOT_FOUND)
 9. citations           built from stored metadata
10. cache + audit log   only grounded answers are cached, never "not found"
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

from app.cache.redis_cache import Cache
from app.config import Settings
from app.rag.citations import Citation, build_context, select_citations
from app.rag.llm import LLMClient
from app.rag.prompts import NOT_FOUND_TOKEN, SYSTEM_PROMPT, build_user_prompt
from app.retrieval.vector_store import QdrantStore
from app.security.permissions import Principal

logger = logging.getLogger(__name__)
audit = logging.getLogger("audit")

NOT_FOUND_MESSAGE = "Not found in the available documents"


@dataclass
class RetrievedChunk:
    rank: int
    score: float
    kept: bool                         # False = below the relevance cutoff
    chunk_id: str
    document_id: str
    document_name: str
    section: str
    page: Optional[int]


@dataclass
class RagResult:
    status: str                        # "answered" | "not_found"
    answer: str
    citations: list[Citation] = field(default_factory=list)
    citation_source: Optional[str] = None
    reason: Optional[str] = None       # why it is "not_found"
    cached: bool = False
    cache_match: Optional[str] = None  # "exact" | "semantic" when cached
    cache_similarity: Optional[float] = None
    top_score: Optional[float] = None  # best similarity score found by the search (even if below the cutoff)
    access_scope: str = ""
    timings_ms: dict[str, float] = field(default_factory=dict)
    retrieved: list[RetrievedChunk] = field(default_factory=list)


def is_not_found(text: str) -> bool:
    return re.sub(r"[^A-Za-z_]", "", text.strip().split("\n")[0]).upper().startswith(NOT_FOUND_TOKEN)


class RagPipeline:
    def __init__(self, settings: Settings, embedder, store: QdrantStore, cache: Cache, llm: LLMClient):
        self.settings, self.embedder, self.store, self.cache, self.llm = settings, embedder, store, cache, llm

    # ------------------------------------------------------------------ public
    def ask(self, question: str, principal: Principal, top_k: Optional[int] = None, debug: bool = False) -> RagResult:
        started = time.perf_counter()
        question = question.strip()
        timings: dict[str, float] = {}

        if not debug:                                           # debug runs always show the full retrieval
            hit = self.cache.get(principal.scope, question)
            if hit:
                result = self._from_cache(hit, principal, "exact")
                result.timings_ms = {"total": _ms(started)}
                self._audit(principal, question, result)
                return result

        t = time.perf_counter()
        query_vector = self.embedder.embed_query(question)
        timings["embed"] = _ms(t)

        if not debug:                                           # same meaning, different words
            t = time.perf_counter()
            similar = self.cache.get_semantic(principal.scope, question, query_vector)
            timings["cache"] = _ms(t)
            if similar:
                result = self._from_cache(similar.value, principal, "semantic", similar.similarity)
                timings["total"] = _ms(started)
                result.timings_ms = timings
                self._audit(principal, question, result)
                return result

        t = time.perf_counter()
        hits = self.store.search(query_vector, sorted(principal.access_tags), limit=top_k or self.settings.retrieval_top_k)
        timings["search"] = _ms(t)

        top_score = round(hits[0].score, 4) if hits else None
        threshold = self.settings.relevance_threshold
        kept = [h for h in hits if h.score >= threshold]
        retrieved = [RetrievedChunk(i + 1, round(h.score, 4), h.score >= threshold, h.payload.get("chunk_id", ""),
                                    h.payload.get("document_id", ""), h.payload.get("file_name", ""),
                                    h.payload.get("section_path", ""), h.payload.get("page_number"))
                     for i, h in enumerate(hits)]

        if len(kept) < self.settings.min_evidence_chunks:
            result = self._not_found("insufficient_evidence")
        else:
            blocks = build_context(kept, self.settings.max_context_words)
            t = time.perf_counter()
            raw = self.llm.generate(SYSTEM_PROMPT, build_user_prompt(question, [b.text_block for b in blocks]))
            timings["llm"] = _ms(t)
            if not raw or is_not_found(raw):
                result = self._not_found("model_not_found")
            else:
                citations, source = select_citations(blocks, raw, self.settings.strict_citations)
                result = (RagResult("answered", raw, citations, source) if citations else self._not_found("no_citation"))

        result.access_scope = principal.scope
        result.top_score = top_score
        result.retrieved = retrieved
        timings["total"] = _ms(started)
        result.timings_ms = timings
        self._store_in_cache(principal, question, result, query_vector)
        self._audit(principal, question, result)
        return result

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def _not_found(reason: str) -> RagResult:
        return RagResult("not_found", NOT_FOUND_MESSAGE, [], None, reason)

    def _store_in_cache(self, principal: Principal, question: str, result: RagResult, query_vector) -> None:
        """Only grounded answers are cached. "Not found" is never cached: a document uploaded a minute
        later (or a better-calibrated cutoff) must be able to answer the same question."""
        if result.status != "answered":
            return
        value = {"status": result.status, "answer": result.answer, "citations": [asdict(c) for c in result.citations],
                 "citation_source": result.citation_source, "reason": result.reason, "top_score": result.top_score}
        self.cache.set(principal.scope, question, value, self.settings.cache_ttl_answer_seconds,
                       {c.document_id for c in result.citations}, vector=query_vector)

    @staticmethod
    def _from_cache(value: dict, principal: Principal, match: str, similarity: Optional[float] = None) -> RagResult:
        return RagResult(value["status"], value["answer"], [Citation(**c) for c in value.get("citations", [])],
                         value.get("citation_source"), value.get("reason"), cached=True, cache_match=match,
                         cache_similarity=similarity, top_score=value.get("top_score"), access_scope=principal.scope)

    @staticmethod
    def _audit(principal: Principal, question: str, result: RagResult) -> None:
        import hashlib
        audit.info(json.dumps({
            "user": principal.user, "role": principal.role, "scope": principal.scope,
            "question_sha": hashlib.sha256(question.lower().encode()).hexdigest()[:12],
            "status": result.status, "reason": result.reason, "cached": result.cached, "cache_match": result.cache_match,
            "documents": sorted({c.document_id for c in result.citations}),
            "ms": result.timings_ms.get("total")}))


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)
