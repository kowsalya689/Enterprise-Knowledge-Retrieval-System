"""Retrieval with metadata and citations, evidence check, Not found, caching and access control."""
import pytest

from app.rag.citations import extract_markers
from app.rag.llm import LLMUnavailableError
from app.rag.pipeline import NOT_FOUND_MESSAGE, RagPipeline, is_not_found
from app.security.permissions import Principal
from tests.conftest import build_services
from tests.doubles import FakeLLM

EMP = Principal("employee-x", "employee", frozenset({"all_employees"}))
HR = Principal("hr-x", "hr", frozenset({"all_employees", "hr"}))
SALARY_Q = "What are the minimum midpoint and maximum salary for grade G5 manager?"
LEAVE_Q = "What is the maximum number of consecutive days of casual leave?"


def test_answer_carries_document_section_page_and_citations(services):
    r = services.pipeline.ask("What is the hotel limit per night in Tier 1 cities for grade G4 to G6?", EMP)
    assert r.status == "answered" and not r.cached
    c = r.citations[0]
    assert (c.document_id, c.document_name, c.section, c.page) == (
        "DOC-003", "DOC-003_Finance_Travel_Reimbursement_Policy.pdf", "3. Eligibility by Grade", 2)
    assert c.chunk_id.startswith("DOC-003-C") and c.snippet and r.citation_source == "model_markers"
    assert set(r.timings_ms) >= {"embed", "search", "llm", "total"}


def test_retrieved_context_given_to_the_model_is_numbered_with_location(services):
    services.pipeline.ask(LEAVE_Q, EMP)
    _, user_prompt = services.llm.calls[-1]
    assert "[1] (Document: DOC-001_HR_Leave_and_Attendance_Policy.docx | Section:" in user_prompt
    assert "Question: " + LEAVE_Q in user_prompt


def test_no_relevant_chunk_returns_not_found_without_calling_the_model(services):
    r = services.pipeline.ask("zebra quantum trampoline xylophone", EMP)
    assert r.status == "not_found" and r.answer == NOT_FOUND_MESSAGE == "Not found in the available documents"
    assert r.reason == "insufficient_evidence" and r.citations == [] and services.llm.calls == []


def test_threshold_decides_what_counts_as_evidence(tmp_path, kb):
    strict = build_services(tmp_path, kb, relevance_threshold=0.99)
    assert strict.pipeline.ask(LEAVE_Q, EMP).reason == "insufficient_evidence"


def test_model_saying_not_found_is_respected(tmp_path, kb):
    s = build_services(tmp_path, kb, llm=FakeLLM("not_found"))
    r = s.pipeline.ask(LEAVE_Q, EMP)
    assert r.status == "not_found" and r.reason == "model_not_found" and r.answer == NOT_FOUND_MESSAGE


def test_answer_without_markers_cites_the_context_unless_strict(tmp_path, kb):
    lenient = build_services(tmp_path, kb, llm=FakeLLM("no_markers")).pipeline.ask(LEAVE_Q, EMP)
    assert lenient.status == "answered" and lenient.citation_source == "retrieved_context" and lenient.citations
    strict = build_services(tmp_path, kb, llm=FakeLLM("no_markers"), strict_citations=True).pipeline.ask(LEAVE_Q, EMP)
    assert strict.status == "not_found" and strict.reason == "no_citation"


def test_marker_parsing_ignores_invented_block_numbers():
    assert extract_markers("a [2] b [2] c [9] d [1]", 3) == [2, 1]
    assert is_not_found("NOT_FOUND") and is_not_found("not_found.") and not is_not_found("The answer is 3")


# ------------------------------------------------------------------ access control
def test_restricted_document_is_invisible_to_an_unauthorised_user(services):
    r = services.pipeline.ask(SALARY_Q, EMP, debug=True)
    assert all(x.document_id != "DOC-011" for x in r.retrieved)
    sent = " ".join(u for _, u in services.llm.calls)
    assert "Compensation" not in sent and "11,00,000" not in sent          # the text never reached the model


def test_authorised_user_gets_the_restricted_answer_with_citation(services):
    r = services.pipeline.ask(SALARY_Q, HR)
    assert r.status == "answered" and r.citations[0].document_id == "DOC-011" and "G5" in services.llm.calls[-1][1]


def test_retrieval_trace_shows_scores_and_dropped_chunks(services):
    r = services.pipeline.ask(LEAVE_Q, EMP, top_k=20, debug=True)
    assert len(r.retrieved) == 20 and [x.rank for x in r.retrieved][:3] == [1, 2, 3]
    assert r.retrieved[0].kept and [x.score for x in r.retrieved] == sorted((x.score for x in r.retrieved), reverse=True)
    assert any(not x.kept for x in r.retrieved)                            # something fell below the cutoff


# ------------------------------------------------------------------ cache behaviour
def test_second_identical_question_is_served_from_cache_without_the_model(services):
    first = services.pipeline.ask(LEAVE_Q, EMP)
    calls = len(services.llm.calls)
    second = services.pipeline.ask("  " + LEAVE_Q.lower() + "  ", EMP)
    assert not first.cached and second.cached and len(services.llm.calls) == calls
    assert second.answer == first.answer and [c.chunk_id for c in second.citations] == [c.chunk_id for c in first.citations]


def test_cached_answer_of_one_role_is_not_served_to_another(services):
    services.pipeline.ask(SALARY_Q, HR)
    other = services.pipeline.ask(SALARY_Q, EMP)
    assert not other.cached and all(c.document_id != "DOC-011" for c in other.citations)


def test_not_found_is_never_cached(services):
    q = "zebra quantum trampoline xylophone"
    first = services.pipeline.ask(q, EMP)
    again = services.pipeline.ask(q, EMP)
    assert first.status == again.status == "not_found" and not again.cached
    assert services.cache.client.get(services.cache.make_key(EMP.scope, q)) is None


def test_model_not_found_is_never_cached_either(tmp_path, kb):
    s = build_services(tmp_path, kb, llm=FakeLLM("not_found"))
    s.pipeline.ask(LEAVE_Q, EMP)
    assert not s.pipeline.ask(LEAVE_Q, EMP).cached
    s.llm.mode = "extractive"                                    # the model now knows the answer ...
    r = s.pipeline.ask(LEAVE_Q, EMP)
    assert r.status == "answered" and not r.cached               # ... and it is not blocked by an old "not found"


def test_grounded_answer_is_cached_with_the_answer_ttl(services):
    services.pipeline.ask(LEAVE_Q, EMP)
    ttl = services.cache.client.ttl(services.cache.make_key(EMP.scope, LEAVE_Q))
    assert 0 < ttl <= services.settings.cache_ttl_answer_seconds


def test_default_relevance_cutoff_is_0_30():
    from app.config import Settings
    assert Settings(_env_file=None).relevance_threshold == 0.30


def test_top_score_is_reported_for_answers_and_not_found(services):
    ok = services.pipeline.ask(LEAVE_Q, EMP)
    no = services.pipeline.ask("zebra quantum trampoline xylophone", EMP)
    assert ok.top_score and ok.top_score >= services.settings.relevance_threshold
    assert no.top_score is not None and no.top_score < services.settings.relevance_threshold
    assert services.pipeline.ask(LEAVE_Q, EMP).top_score == ok.top_score          # kept in the cache entry


def test_changing_the_document_set_invalidates_cached_answers(services):
    services.pipeline.ask(LEAVE_Q, EMP)
    services.cache.bump_index_version()
    assert not services.pipeline.ask(LEAVE_Q, EMP).cached


def test_llm_failure_is_not_cached_and_is_raised(tmp_path, kb):
    s = build_services(tmp_path, kb, llm=FakeLLM("error"))
    with pytest.raises(LLMUnavailableError):
        s.pipeline.ask(LEAVE_Q, EMP)
    s.llm.mode = "extractive"
    assert not s.pipeline.ask(LEAVE_Q, EMP).cached


# ------------------------------------------------------------------ semantic cache (pipeline level)
class _Paraphrasing:
    """Embedder wrapper: questions in ``same`` get the vector of the first one (a perfect paraphrase)."""
    def __init__(self, inner, same):
        self.inner, self.same, self.name, self.dimension = inner, same, inner.name, inner.dimension

    def embed_query(self, text):
        return self.inner.embed_query(self.same[0] if text in self.same else text)

    def embed_documents(self, texts):
        return self.inner.embed_documents(texts)


def test_reworded_question_is_answered_from_the_semantic_cache(services):
    reworded = "Casual leave: how many days in a row am I allowed?"
    services.pipeline.embedder = _Paraphrasing(services.embedder, [LEAVE_Q, reworded])
    first = services.pipeline.ask(LEAVE_Q, EMP)
    calls = len(services.llm.calls)
    second = services.pipeline.ask(reworded, EMP)
    assert first.cache_match is None and second.cached and second.cache_match == "semantic"
    assert second.cache_similarity >= services.settings.semantic_cache_threshold
    assert second.answer == first.answer and len(services.llm.calls) == calls          # no second LLM call
    assert services.pipeline.ask(LEAVE_Q, EMP).cache_match == "exact"


def test_semantic_cache_does_not_mix_up_different_numbers(services):
    a, b = "Hotel limit per night for grade G4 in Tier 1 cities?", "Hotel limit per night for grade G7 in Tier 1 cities?"
    services.pipeline.embedder = _Paraphrasing(services.embedder, [a, b])             # embeddings are identical
    services.pipeline.ask(a, EMP)
    assert not services.pipeline.ask(b, EMP).cached


def test_semantic_cache_never_crosses_access_scopes(services):
    other = "Pay range for the G5 manager grade?"
    services.pipeline.embedder = _Paraphrasing(services.embedder, [SALARY_Q, other])
    services.pipeline.ask(SALARY_Q, HR)
    leaked = services.pipeline.ask(other, EMP)
    assert not leaked.cached and all(c.document_id != "DOC-011" for c in leaked.citations)


def test_api_reports_how_the_answer_was_cached(client):
    from tests.conftest import H
    body = {"question": LEAVE_Q}
    first = client.post("/query", json=body, headers=H["employee"]).json()
    second = client.post("/query", json=body, headers=H["employee"]).json()
    assert first["cached"] is False and first["cache_match"] is None and first["top_score"] > 0
    assert second["cached"] is True and second["cache_match"] == "exact" and second["top_score"] == first["top_score"]
