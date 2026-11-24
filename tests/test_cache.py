"""Redis cache behaviour: hit/miss, TTL, isolation, invalidation, Redis failure."""
import os

import fakeredis
import pytest

from app.cache.redis_cache import NullCache, RedisAnswerCache, build_cache, normalise_question
from app.config import Settings


@pytest.fixture()
def cache():
    return RedisAnswerCache(fakeredis.FakeRedis(decode_responses=True), "t", "p1", "model-x")


def test_question_normalisation_makes_exact_match_forgiving():
    assert normalise_question("  How MANY days?? ") == normalise_question("how many   days")


def test_miss_then_hit(cache):
    assert cache.get("scopeA", "How many days?") is None
    cache.set("scopeA", "How many days?", {"answer": "18"}, 60, ["DOC-1"])
    assert cache.get("scopeA", "how many days") == {"answer": "18"}


def test_ttl_is_applied(cache):
    cache.set("s", "q", {"a": 1}, 90, [])
    assert 0 < cache.client.ttl(cache.make_key("s", "q")) <= 90


def test_users_with_different_access_scope_never_share_entries(cache):
    cache.set("scope-hr", "salary band", {"answer": "secret"}, 60, [])
    assert cache.get("scope-employee", "salary band") is None
    assert cache.get("scope-hr", "salary band") == {"answer": "secret"}


def test_document_invalidation_removes_only_answers_that_cited_it(cache):
    cache.set("s", "q1", {"a": 1}, 60, ["DOC-1"])
    cache.set("s", "q2", {"a": 2}, 60, ["DOC-2"])
    assert cache.invalidate_document("DOC-1") == 1
    assert cache.get("s", "q1") is None and cache.get("s", "q2") == {"a": 2}


def test_index_version_bump_makes_old_answers_unreachable(cache):
    cache.set("s", "q", {"a": 1}, 60, [])
    cache.bump_index_version()
    assert cache.get("s", "q") is None


def test_prompt_or_model_change_uses_new_keys():
    client = fakeredis.FakeRedis(decode_responses=True)
    old = RedisAnswerCache(client, "t", "p1", "m1"); old.set("s", "q", {"a": 1}, 60, [])
    assert RedisAnswerCache(client, "t", "p2", "m1").get("s", "q") is None
    assert RedisAnswerCache(client, "t", "p1", "m2").get("s", "q") is None
    assert RedisAnswerCache(client, "t", "p1", "m1").get("s", "q") == {"a": 1}


def test_redis_failure_is_survivable(monkeypatch):
    class Broken:
        def get(self, *a, **k): raise ConnectionError("down")
        def pipeline(self): raise ConnectionError("down")
        def ping(self): raise ConnectionError("down")
        def incr(self, *a): raise ConnectionError("down")
        def smembers(self, *a): raise ConnectionError("down")
    c = RedisAnswerCache(Broken())
    assert c.get("s", "q") is None and c.set("s", "q", {}, 10, []) is False
    assert c.invalidate_document("D") == 0 and c.bump_index_version() == 0 and c.ping() is False


def test_cache_is_off_without_redis_url_or_when_disabled():
    assert isinstance(build_cache(Settings(_env_file=None), "p", "m"), NullCache)
    assert isinstance(build_cache(Settings(_env_file=None, cache_enabled=False, redis_url="redis://x"), "p", "m"), NullCache)


@pytest.mark.skipif(not os.getenv("REDIS_TEST_URL"), reason="set REDIS_TEST_URL to test against a real Redis")
def test_real_redis_ttl_and_invalidation():
    import redis
    client = redis.Redis.from_url(os.environ["REDIS_TEST_URL"], decode_responses=True)
    client.flushdb()
    c = RedisAnswerCache(client, "real", "p", "m")
    c.set("s", "q", {"a": 1}, 30, ["D1"])
    assert c.get("s", "q") == {"a": 1} and 0 < client.ttl(c.make_key("s", "q")) <= 30
    assert c.invalidate_document("D1") == 1 and c.get("s", "q") is None
    client.flushdb()


# ============================================================ semantic match
import numpy as np  # noqa: E402

from app.cache.redis_cache import number_signature  # noqa: E402


def _vec(*values):
    v = np.array(values, dtype=np.float32)
    return v / np.linalg.norm(v)


A = _vec(1, 0, 0)
CLOSE = _vec(1, 0.1, 0)        # cosine with A about 0.995
FAR = _vec(1, 1, 0)            # cosine with A about 0.707


@pytest.fixture()
def sem():
    return RedisAnswerCache(fakeredis.FakeRedis(decode_responses=True), "t", "p1", "model-x",
                            semantic_threshold=0.92, semantic_max_entries=3)


def test_semantic_hit_for_a_similar_question(sem):
    sem.set("s", "How many days of annual leave do I get?", {"answer": "18"}, 60, ["D1"], vector=A)
    hit = sem.get_semantic("s", "What is my yearly leave entitlement?", CLOSE)
    assert hit and hit.value == {"answer": "18"} and hit.similarity >= 0.92
    assert hit.matched_question == "how many days of annual leave do i get"


def test_semantic_miss_below_the_threshold(sem):
    sem.set("s", "q1", {"answer": "18"}, 60, [], vector=A)
    assert sem.get_semantic("s", "something else", FAR) is None


def test_questions_with_different_numbers_never_match_semantically(sem):
    sem.set("s", "Hotel limit for grade G4 in Tier 1 cities?", {"answer": "6000"}, 60, [], vector=A)
    assert sem.get_semantic("s", "Hotel limit for grade G7 in Tier 1 cities?", A) is None      # identical vectors, other grade
    assert sem.get_semantic("s", "Hotel limit for grade G4 in Tier 1 cities please", A) is not None
    assert number_signature("G5, 10 days") == {"g5", "10"} and number_signature("no numbers here") == frozenset()


def test_semantic_match_respects_the_access_scope(sem):
    sem.set("scope-hr", "salary band G5", {"answer": "secret"}, 60, [], vector=A)
    assert sem.get_semantic("scope-employee", "pay band for G5", A) is None
    assert sem.get_semantic("scope-hr", "pay band for G5", A) is not None


def test_semantic_entry_follows_the_ttl_of_the_answer(sem):
    sem.set("s", "q", {"answer": "x"}, 60, [], vector=A)
    sem.client.delete(sem.make_key("s", "q"))                      # what Redis does when the TTL ends
    assert sem.get_semantic("s", "similar q", CLOSE) is None
    assert sem.client.hlen(sem._vectors_key("s", "0")) == 0         # the stale vector was cleaned up


def test_semantic_entries_are_removed_by_document_invalidation_and_index_version(sem):
    sem.set("s", "q1", {"a": 1}, 60, ["DOC-1"], vector=A)
    assert sem.invalidate_document("DOC-1") == 1
    assert sem.get_semantic("s", "q1 reworded", CLOSE) is None
    sem.set("s", "q2", {"a": 2}, 60, [], vector=A)
    sem.bump_index_version()
    assert sem.get_semantic("s", "q2 reworded", CLOSE) is None


def test_semantic_store_is_limited_and_drops_the_oldest(sem):
    for i in range(5):
        sem.set("s", f"question number {i}", {"a": i}, 60, [], vector=_vec(1, i * 0.01, 0))
    assert sem.client.hlen(sem._vectors_key("s", "0")) == 3
    assert sem.get("s", "question number 4") == {"a": 4}            # the newest is kept


def test_semantic_can_be_switched_off_and_needs_a_vector():
    off = RedisAnswerCache(fakeredis.FakeRedis(decode_responses=True), "t", semantic_enabled=False)
    off.set("s", "q", {"a": 1}, 60, [], vector=A)
    assert off.get_semantic("s", "q", A) is None and off.get("s", "q") == {"a": 1}
    on = RedisAnswerCache(fakeredis.FakeRedis(decode_responses=True), "t")
    on.set("s", "q", {"a": 1}, 60, [])                              # no vector: exact only
    assert on.get_semantic("s", "q", A) is None and on.get_semantic("s", "q", None) is None


def test_semantic_redis_failure_is_survivable():
    class Broken:
        def get(self, *a, **k): raise ConnectionError("down")
        def hgetall(self, *a, **k): raise ConnectionError("down")
    assert RedisAnswerCache(Broken()).get_semantic("s", "q", A) is None


@pytest.mark.skipif(not os.getenv("REDIS_TEST_URL"), reason="set REDIS_TEST_URL to test against a real Redis")
def test_real_redis_semantic_ttl_and_isolation():
    import time
    import redis
    client = redis.Redis.from_url(os.environ["REDIS_TEST_URL"], decode_responses=True)
    client.flushdb()
    c = RedisAnswerCache(client, "real", "p", "m", semantic_max_entries=3)
    c.set("s", "How many leave days?", {"a": 18}, 2, ["D1"], vector=A)
    hit = c.get_semantic("s", "yearly leave days?", CLOSE)
    assert hit and hit.value == {"a": 18}
    assert c.get_semantic("other-scope", "yearly leave days?", CLOSE) is None
    time.sleep(2.2)                                                # real TTL expiry
    assert c.get("s", "How many leave days?") is None and c.get_semantic("s", "yearly leave days?", CLOSE) is None
    client.flushdb()
