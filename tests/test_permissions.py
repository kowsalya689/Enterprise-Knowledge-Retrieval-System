"""Roles, API keys and access-control filtering."""
import numpy as np
import pytest

from app.config import PROJECT_ROOT
from app.retrieval.vector_store import VectorStoreError
from app.security.permissions import (
    AuthenticationError, Authenticator, ConfigurationError, load_roles, parse_api_keys,
)
from tests.doubles import BagOfWordsEmbedder

ROLES = load_roles(PROJECT_ROOT / "data" / "roles.json")


def test_api_keys_are_parsed():
    assert parse_api_keys("a:hr, b:employee") == {"a": "hr", "b": "employee"}
    assert parse_api_keys("") == {}
    with pytest.raises(ConfigurationError):
        parse_api_keys("broken")


def test_authenticate_returns_role_and_tags():
    auth = Authenticator(ROLES, {"secret1": "hr"})
    p = auth.authenticate("secret1")
    assert p.role == "hr" and p.access_tags == frozenset({"all_employees", "hr"})
    with pytest.raises(AuthenticationError):
        auth.authenticate("wrong")
    with pytest.raises(AuthenticationError):
        auth.authenticate(None)


def test_key_for_unknown_role_is_a_configuration_error():
    with pytest.raises(ConfigurationError):
        Authenticator(ROLES, {"k": "ceo"})


def test_scope_differs_between_permission_sets_and_is_stable():
    auth = Authenticator(ROLES, {"a": "hr", "b": "employee", "c": "hr"})
    assert auth.authenticate("a").scope == auth.authenticate("c").scope != auth.authenticate("b").scope


def test_search_without_access_tags_is_refused(kb):
    with pytest.raises(VectorStoreError):
        kb.search(BagOfWordsEmbedder().embed_query("salary"), [], 5)


@pytest.mark.parametrize("role,tags,forbidden", [
    ("employee", ["all_employees"], {"DOC-004", "DOC-005", "DOC-008", "DOC-009", "DOC-010", "DOC-011"}),
    ("hr", ["all_employees", "hr"], {"DOC-004", "DOC-005", "DOC-008", "DOC-009", "DOC-010"}),
    ("sales", ["all_employees", "sales"], {"DOC-004", "DOC-005", "DOC-009", "DOC-010", "DOC-011"}),
])
def test_search_never_returns_chunks_outside_the_access_tags(kb, role, tags, forbidden):
    emb = BagOfWordsEmbedder()
    for q in ("salary band G5", "budget total", "ticket escalation", "procurement approval", "security incident P1", "sales revenue"):
        hits = kb.search(emb.embed_query(q), tags, limit=20)
        assert hits
        for h in hits:
            assert set(h.payload["access_tags"]) & set(tags), (role, q, h.payload["chunk_id"])
            assert h.payload["document_id"] not in forbidden
