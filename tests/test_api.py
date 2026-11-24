"""FastAPI endpoints: health, upload, listing, query, collection info, errors and Swagger."""
import io

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from tests.conftest import H, RAW_DIR

LEAVE_Q = {"question": "What is the maximum number of consecutive days of casual leave?"}


# ------------------------------------------------------------------ health and docs
def test_health_reports_every_component(client):
    r = client.get("/health")
    body = r.json()
    assert r.status_code == 200 and body["status"] == "ok"
    assert set(body["components"]) == {"api", "embedding_model", "qdrant", "redis_cache", "ollama"}
    assert "163 points" in body["components"]["qdrant"]["detail"]


def test_health_is_degraded_when_the_collection_is_empty(empty_client):
    client, _ = empty_client
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "degraded"


def test_swagger_ui_and_openapi_are_available(client):
    assert client.get("/docs").status_code == 200
    spec = client.get("/openapi.json").json()
    assert {"/health", "/documents/upload", "/documents", "/query", "/collection/info"} <= set(spec["paths"])
    assert "APIKeyHeader" in spec["components"]["securitySchemes"]
    assert spec["paths"]["/query"]["post"]["responses"]["200"]["content"]["application/json"]["schema"]["$ref"].endswith("QueryResponse")


# ------------------------------------------------------------------ authentication
@pytest.mark.parametrize("method,path", [("get", "/documents"), ("get", "/collection/info"), ("post", "/query")])
def test_protected_endpoints_need_a_valid_key(client, method, path):
    kw = {"json": LEAVE_Q} if method == "post" else {}
    r = getattr(client, method)(path, **kw)
    assert r.status_code == 401 and r.json()["error"]["code"] == "authentication_failed"
    assert getattr(client, method)(path, headers={"X-API-Key": "wrong"}, **kw).status_code == 401


# ------------------------------------------------------------------ upload
def _upload(client, role, name, data, tags="all_employees", dept="Training"):
    return client.post("/documents/upload", headers=H[role], data={"access_tags": tags, "department": dept},
                       files={"file": (name, io.BytesIO(data), "application/octet-stream")})


def test_upload_txt_indexes_chunks_and_makes_them_searchable(empty_client):
    client, svc = empty_client
    text = "1. Parking\n----------\n\nEmployees may park in the east car park between 07:00 and 19:00.\n"
    r = _upload(client, "employee", "Parking Rules.txt", text.encode())
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["status"] == "indexed" and body["chunks"] >= 1 and body["file_name"] == "Parking_Rules.txt"
    assert body["access_tags"] == ["all_employees"] and body["version"] == "1.0" and body["document_id"].startswith("UPL-")
    assert svc.store.count() == body["chunks"]
    pl = svc.store.sample(limit=1)[0]["payload"]
    assert pl["document_id"] == body["document_id"] and pl["department"] == "Training" and pl["embedding_model"] == "bow-test-embedder"
    ans = client.post("/query", headers=H["employee"], json={"question": "When may employees park in the east car park?"}).json()
    assert ans["status"] == "answered" and ans["citations"][0]["document_name"] == "Parking_Rules.txt"


@pytest.mark.parametrize("name", ["DOC-002_IT_Information_Security_Policy.pdf", "DOC-001_HR_Leave_and_Attendance_Policy.docx",
                                  "DOC-007_Support_Escalation_SOP.txt"])
def test_upload_pdf_docx_txt_from_the_dataset(empty_client, name):
    client, _ = empty_client
    tags = "all_employees"
    r = _upload(client, "employee", name, (RAW_DIR / name).read_bytes(), tags)
    assert r.status_code == 201 and r.json()["chunks"] > 5


def test_upload_rejections(empty_client):
    client, _ = empty_client
    assert _upload(client, "employee", "x.exe", b"MZ").status_code == 415
    assert _upload(client, "employee", "empty.txt", b"").json()["error"]["code"] == "empty_file"
    assert _upload(client, "employee", "a.txt", b"text", tags="hr").status_code == 403      # a tag the user does not hold
    assert _upload(client, "employee", "a.txt", b"text", tags="  ").status_code in (422,)      # no tags: fail closed
    r = client.post("/documents/upload", headers=H["employee"], data={"access_tags": "all_employees"})
    assert r.status_code == 422 and r.json()["error"]["code"] == "validation_error"           # no file


def test_upload_size_limit_is_enforced_while_streaming(tmp_path, empty_client):
    client, svc = empty_client
    svc.settings = svc.settings.model_copy(update={"max_file_size_mb": 1})
    svc.documents.settings = svc.settings
    r = _upload(client, "employee", "big.txt", b"a" * (1_048_576 + 10))
    assert r.status_code == 413 and r.json()["error"]["code"] == "file_too_large"


def test_path_traversal_in_the_file_name_is_neutralised(empty_client, tmp_path):
    client, svc = empty_client
    r = _upload(client, "employee", "../../etc/pass wd.txt", b"1. A\n---\n\nhello world text here\n")
    assert r.status_code == 201 and r.json()["file_name"] == "pass_wd.txt"
    assert (svc.settings.upload_dir / "pass_wd.txt").exists() and not (tmp_path / "etc").exists()


def test_same_file_twice_is_unchanged_and_changed_content_gets_a_new_version(empty_client):
    client, svc = empty_client
    one = b"1. Rule\n------\n\nVisitors sign in at the gate.\n"
    first = _upload(client, "employee", "rules.txt", one).json()
    again = _upload(client, "employee", "rules.txt", one)
    assert again.status_code == 201 and again.json()["status"] == "unchanged" and again.json()["version"] == "1.0"
    changed = _upload(client, "employee", "rules.txt", one + b"\n2. More\n------\n\nBadges are mandatory.\n").json()
    assert changed["document_id"] == first["document_id"] and changed["version"] == "2.0" and changed["chunks"] > first["chunks"]
    assert svc.store.count(first["document_id"]) == changed["chunks"]


def test_uploading_a_document_invalidates_cached_answers(empty_client):
    client, svc = empty_client
    _upload(client, "employee", "rules.txt", b"1. Rule\n------\n\nVisitors sign in at the gate.\n")
    q = {"question": "Where do visitors sign in?"}
    assert not client.post("/query", headers=H["employee"], json=q).json()["cached"]
    assert client.post("/query", headers=H["employee"], json=q).json()["cached"]
    _upload(client, "employee", "extra.txt", b"1. More\n------\n\nBadges are mandatory for visitors.\n")
    assert not client.post("/query", headers=H["employee"], json=q).json()["cached"]


# ------------------------------------------------------------------ listing and collection info
def test_document_list_only_shows_what_the_role_may_see(client):
    emp = client.get("/documents", headers=H["employee"]).json()
    assert emp["count"] == 5 and {d["document_id"] for d in emp["documents"]} == {"DOC-001", "DOC-002", "DOC-003", "DOC-006", "DOC-007"}
    hr = client.get("/documents", headers=H["hr"]).json()
    assert "DOC-011" in {d["document_id"] for d in hr["documents"]} and hr["count"] == 6
    d = next(x for x in hr["documents"] if x["document_id"] == "DOC-011")
    assert d["access_tags"] == ["hr", "management"] and d["department"] == "Human Resources" and d["chunks"] == 10


def test_collection_info_shows_how_points_are_stored(client):
    r = client.get("/collection/info", params={"sample": 2}, headers=H["employee"])
    body = r.json()
    assert r.status_code == 200 and body["points_count"] == 163 and body["vector_size"] == 4096 and body["distance"] == "Cosine"
    assert body["embedding_model"] == "bow-test-embedder" and len(body["sample_points"]) == 2
    for p in body["sample_points"]:
        assert p["vector_size"] == 4096 and set(p["payload"]["access_tags"]) <= {"all_employees"}     # only permitted points
        assert {"document_id", "file_name", "text", "page_number", "department", "access_tags"} <= set(p["payload"])
    assert len(body["documents_visible_to_you"]) == 5


# ------------------------------------------------------------------ query
def test_query_returns_answer_with_document_name_page_and_citations(client):
    r = client.post("/query", headers=H["employee"],
                    json={"question": "What is the hotel limit per night in Tier 1 cities for grade G4 to G6?"})
    b = r.json()
    assert r.status_code == 200 and b["status"] == "answered" and b["answer"].endswith("[1]")
    c = b["citations"][0]
    assert c["document_name"] == "DOC-003_Finance_Travel_Reimbursement_Policy.pdf" and c["page"] == 2 and c["section"]
    assert b["cached"] is False and b["retrieved"] is None and b["timings_ms"]["total"] >= 0


def test_query_not_found_message(client):
    b = client.post("/query", headers=H["employee"], json={"question": "zebra quantum trampoline xylophone"}).json()
    assert b["status"] == "not_found" and b["answer"] == "Not found in the available documents" and b["citations"] == []


def test_query_access_control_end_to_end(client):
    q = {"question": "What are the minimum midpoint and maximum salary for grade G5 manager?"}
    emp = client.post("/query", headers=H["employee"], json=q).json()
    hr = client.post("/query", headers=H["hr"], json=q).json()
    assert all(c["document_id"] != "DOC-011" for c in emp["citations"])
    assert hr["status"] == "answered" and hr["citations"][0]["document_id"] == "DOC-011" and emp["access_scope"] != hr["access_scope"]


def test_query_debug_returns_the_retrieval_trace(client):
    b = client.post("/query", headers=H["employee"], json={**LEAVE_Q, "debug": True, "top_k": 5}).json()
    assert len(b["retrieved"]) == 5 and {"rank", "score", "kept", "chunk_id", "document_name", "section", "page"} <= set(b["retrieved"][0])


def test_query_cache_flag(client):
    assert client.post("/query", headers=H["employee"], json=LEAVE_Q).json()["cached"] is False
    assert client.post("/query", headers=H["employee"], json=LEAVE_Q).json()["cached"] is True


def test_query_validation_errors_use_the_error_format(client):
    r = client.post("/query", headers=H["employee"], json={"question": "x"})
    assert r.status_code == 422 and r.json()["error"]["code"] == "validation_error" and r.json()["error"]["details"]
    assert client.post("/query", headers=H["employee"], json={"question": "valid question here", "top_k": 99}).status_code == 422


def test_llm_outage_returns_503_with_a_clear_message(tmp_path, kb):
    from tests.conftest import build_services
    from tests.doubles import FakeLLM
    svc = build_services(tmp_path, kb, llm=FakeLLM("error"))
    c = TestClient(create_app(svc.settings, svc), raise_server_exceptions=False)
    r = c.post("/query", headers=H["employee"], json=LEAVE_Q)
    assert r.status_code == 503 and r.json()["error"]["code"] == "llm_unavailable"


def test_unexpected_errors_return_500_without_internal_details(tmp_path, kb):
    from tests.conftest import build_services
    svc = build_services(tmp_path, kb)
    svc.pipeline.ask = lambda *a, **k: 1 / 0
    r = TestClient(create_app(svc.settings, svc), raise_server_exceptions=False).post("/query", headers=H["employee"], json=LEAVE_Q)
    assert r.status_code == 500 and "division" not in r.text and r.json()["error"]["code"] == "internal_error"


def test_application_start_up_builds_the_real_wiring_from_settings(tmp_path, monkeypatch):
    """The production start-up path (settings -> services) with only the heavy model replaced."""
    import app.dependencies as deps
    from tests.conftest import make_settings
    from tests.doubles import BagOfWordsEmbedder
    monkeypatch.setattr(deps, "SentenceTransformerEmbedder", lambda settings: BagOfWordsEmbedder())
    cfg = make_settings(tmp_path, qdrant_local_path=tmp_path / "qdrant", ollama_base_url="http://127.0.0.1:9", ollama_timeout=1)
    with TestClient(create_app(cfg)) as c:                       # runs the lifespan = build_services(settings)
        body = c.get("/health").json()
        assert body["components"]["qdrant"]["status"] == "degraded" or body["components"]["qdrant"]["status"] == "ok"
        assert body["components"]["embedding_model"]["status"] == "ok"
        assert body["components"]["ollama"]["status"] == "degraded"            # no Ollama on port 9
        assert body["components"]["redis_cache"]["status"] == "disabled"       # REDIS_URL not set
        assert c.get("/documents").status_code == 401
        up = c.post("/documents/upload", headers=H["employee"], data={"access_tags": "all_employees"},
                    files={"file": ("n.txt", io.BytesIO(b"1. A\n---\n\nhello world policy text\n"), "text/plain")})
        assert up.status_code == 201 and c.get("/collection/info", headers=H["employee"]).json()["points_count"] == 1
        q = c.post("/query", headers=H["employee"], json={"question": "hello world policy text"})
        assert q.status_code == 503 and q.json()["error"]["code"] == "llm_unavailable"       # clear error, not a crash


# ------------------------------------------------------------------ web pages
def test_both_pages_and_their_assets_are_served(client):
    for path, marker in (("/chat", "Ask a question"), ("/upload", "Upload a document")):
        r = client.get(path)
        assert r.status_code == 200 and "text/html" in r.headers["content-type"] and marker in r.text
    assert client.get("/query/ui").status_code == 200 and client.get("/ingest/ui").status_code == 200   # course-material names
    assert client.get("/static/app.js").status_code == 200 and client.get("/static/app.css").status_code == 200
    home = client.get("/", follow_redirects=False)
    assert home.status_code in (302, 307) and home.headers["location"] == "/chat"


def test_pages_are_not_listed_in_the_api_documentation(client):
    paths = client.get("/openapi.json").json()["paths"]
    assert "/chat" not in paths and "/upload" not in paths and "/query" in paths


def test_pages_never_insert_server_text_as_html():
    from app.api.pages import WEB_DIR
    for f in (WEB_DIR / "chat.html", WEB_DIR / "upload.html", WEB_DIR / "static" / "app.js"):
        assert "innerHTML" not in f.read_text(encoding="utf-8"), f.name


def test_slide_addresses_serve_the_same_pages(client):
    assert client.get("/query/ui").text == client.get("/chat").text
    assert client.get("/ingest/ui").text == client.get("/upload").text


# ------------------------------------------------------------------ web pages without a key
WEB = {"X-Web-UI": "1"}


def test_web_pages_have_no_api_key_box():
    from app.api.pages import WEB_DIR
    for f in ("chat.html", "upload.html"):
        text = (WEB_DIR / f).read_text(encoding="utf-8")
        assert 'id="api-key"' not in text and "X-API-Key" not in text, f
    js = (WEB_DIR / "static" / "app.js").read_text(encoding="utf-8")
    assert "X-API-Key" not in js and "sessionStorage" not in js and '"X-Web-UI"' in js


def test_web_header_without_key_uses_the_web_role_only(client):
    r = client.get("/documents", headers=WEB)
    assert r.status_code == 200
    assert {d["file_name"] for d in r.json()["documents"]}                       # the employee can see some documents
    seen_by_hr = client.get("/documents", headers=H["hr"]).json()["count"]
    assert r.json()["count"] < seen_by_hr                                       # and fewer than HR


def test_calls_without_key_and_without_web_header_are_refused(client):
    assert client.get("/documents").status_code == 401
    assert client.post("/query", json=LEAVE_Q).status_code == 401


def test_web_role_can_be_switched_off(tmp_path, kb):
    from tests.conftest import build_services
    svc = build_services(tmp_path, kb, web_ui_role="")
    c = TestClient(create_app(svc.settings, svc), raise_server_exceptions=False)
    assert c.get("/documents", headers=WEB).status_code == 401
    assert c.get("/documents", headers=H["employee"]).status_code == 200          # keys still work


def test_a_key_always_beats_the_web_header(client):
    r = client.get("/documents", headers={**WEB, **H["hr"]})
    assert r.json()["count"] == client.get("/documents", headers=H["hr"]).json()["count"]


def test_web_role_cannot_assign_tags_it_does_not_hold(empty_client):
    client, _ = empty_client
    r = client.post("/documents/upload", headers=WEB, data={"access_tags": "hr"},
                    files={"file": ("a.txt", io.BytesIO(b"text"), "text/plain")})
    assert r.status_code == 403


def test_unknown_web_role_is_a_configuration_error(tmp_path):
    from app.security.permissions import Authenticator, ConfigurationError
    from tests.conftest import make_settings
    with pytest.raises(ConfigurationError):
        Authenticator.from_settings(make_settings(tmp_path, web_ui_role="nobody"))


# ------------------------------------------------------------------ how the file was processed
def test_upload_response_explains_embedding_and_chunking(empty_client):
    client, svc = empty_client
    text = "1. Parking\n----------\n\nEmployees may park in the east car park between 07:00 and 19:00.\n"
    body = _upload(client, "employee", "Parking Rules.txt", text.encode()).json()
    assert body["embedding_model"] == "bow-test-embedder"
    assert body["chunking"]["max_words_per_chunk"] == svc.settings.chunk_max_words
    assert body["chunking"]["overlap_words"] == svc.settings.chunk_overlap_words
    assert body["chunk_words_max"] >= body["chunk_words_avg"] > 0


def test_pipeline_info_endpoint(client, services):
    r = client.get("/pipeline/info")
    assert r.status_code == 200
    b = r.json()
    assert b["embedding"]["model"] and b["chunking"]["max_words_per_chunk"] == 380
    assert b["relevance_threshold"] == services.settings.relevance_threshold and b["web_ui_role"] == "employee"
