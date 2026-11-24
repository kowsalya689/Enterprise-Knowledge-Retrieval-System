"""Question answering."""
from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, Depends

from app.api.schemas import CitationOut, ErrorResponse, QueryRequest, QueryResponse, RetrievedChunkOut
from app.dependencies import Services, get_principal, get_services
from app.security.permissions import Principal

router = APIRouter(tags=["Query"])


@router.post("/query", response_model=QueryResponse, summary="Ask a question and get a cited answer",
             responses={401: {"model": ErrorResponse}, 503: {"model": ErrorResponse}})
def query(body: QueryRequest, principal: Principal = Depends(get_principal), services: Services = Depends(get_services)):
    """Retrieves the most relevant chunks the caller may see, checks that there is enough evidence,
    asks the local language model to answer from them, and returns the answer with document, section and page.

    When the documents do not contain the answer the response is `status: not_found` with the message
    *Not found in the available documents*.
    """
    r = services.pipeline.ask(body.question, principal, top_k=body.top_k, debug=body.debug)
    return QueryResponse(
        status=r.status, answer=r.answer, citations=[CitationOut(**asdict(c)) for c in r.citations],
        citation_source=r.citation_source, reason=r.reason, cached=r.cached, cache_match=r.cache_match,
        cache_similarity=r.cache_similarity, top_score=r.top_score, access_scope=r.access_scope,
        timings_ms=r.timings_ms, retrieved=[RetrievedChunkOut(**asdict(x)) for x in r.retrieved] if body.debug else None)
