"""Redis answer cache: exact match + semantic match + TTL + invalidation.

Two ways to find a cached answer, always inside the caller's own access scope:

1. **Exact match**  the normalised question text is hashed (cheap, no model call).
2. **Semantic match**  the question embedding is compared with the embeddings of earlier
   questions of the same scope; the best one at or above ``semantic_threshold`` is reused.
   Safety guard: two questions that contain different numbers or codes ("G4" vs "G5",
   "10 days" vs "30 days") never match, because embeddings barely notice such differences.

Rules
-----
* Only grounded answers are cached.  "Not found" is **never** cached (a document uploaded a
  minute later must be able to answer it) and neither is an error.
* Every entry has a TTL (``cache_ttl_answer_seconds``).

Key layout::

    answer   rag:v1:<access scope>:<index version>:<prompt version>:<model version>:<question hash>
    vectors  rag:v1:sem:<access scope>:<index version>:<prompt version>:<model version>   (Redis hash)

* ``access scope``    hash of the caller's access tags: users with different permissions
                      can never read each other's cached answers (exact or semantic);
* ``index version``   a counter increased whenever a document is added, replaced or
                      deleted: older keys become unreachable at once;
* ``prompt/model``    a new prompt or model never reuses old answers.

Every answer key is also added to a Redis set per cited document, so a document update
deletes exactly the answers that used it.  If Redis is down the cache is simply
skipped (a warning is logged); permissions are never skipped.
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Protocol

import numpy as np

from app.config import Settings

logger = logging.getLogger(__name__)
SCHEMA = "v1"
_NUMBER_TOKEN = re.compile(r"[a-z]*\d[a-z0-9.,]*")


def normalise_question(question: str) -> str:
    return re.sub(r"\s+", " ", question.strip().lower()).rstrip("?! .")


def number_signature(question: str) -> frozenset[str]:
    """Numbers and codes in a question ('g5', '10', '30'); both questions must have the same set."""
    return frozenset(t.strip(".,") for t in _NUMBER_TOKEN.findall(question.lower()))


@dataclass
class SemanticHit:
    value: dict
    similarity: float
    matched_question: str


class Cache(Protocol):
    enabled: bool
    def get(self, scope: str, question: str) -> Optional[dict]: ...
    def get_semantic(self, scope: str, question: str, vector) -> Optional[SemanticHit]: ...
    def set(self, scope: str, question: str, value: dict, ttl: int, document_ids: Iterable[str],
            vector=None) -> bool: ...
    def invalidate_document(self, document_id: str) -> int: ...
    def bump_index_version(self) -> int: ...
    def ping(self) -> bool: ...


class NullCache:
    """Used when the cache is switched off or not configured."""
    enabled = False

    def get(self, scope, question): return None
    def get_semantic(self, scope, question, vector): return None
    def set(self, scope, question, value, ttl, document_ids, vector=None): return False
    def invalidate_document(self, document_id): return 0
    def bump_index_version(self): return 0
    def ping(self): return False


class RedisAnswerCache:
    enabled = True

    def __init__(self, client, prefix: str = "rag", prompt_version: str = "p0", model_version: str = "m0",
                 semantic_enabled: bool = True, semantic_threshold: float = 0.90, semantic_max_entries: int = 300):
        self.client = client
        self.prefix = prefix
        self.prompt_version = prompt_version
        self.model_version = _short(model_version)
        self.semantic_enabled = semantic_enabled
        self.semantic_threshold = semantic_threshold
        self.semantic_max_entries = semantic_max_entries

    # ------------------------------------------------------------------ keys
    @property
    def _version_key(self) -> str:
        return f"{self.prefix}:index_version"

    def _index_version(self) -> str:
        return self.client.get(self._version_key) or "0"

    @staticmethod
    def _qhash(question: str) -> str:
        return hashlib.sha256(normalise_question(question).encode()).hexdigest()[:24]

    def _key(self, scope: str, version: str, qhash: str) -> str:
        return f"{self.prefix}:{SCHEMA}:{scope}:{version}:{self.prompt_version}:{self.model_version}:{qhash}"

    def make_key(self, scope: str, question: str) -> str:
        return self._key(scope, self._index_version(), self._qhash(question))

    def _vectors_key(self, scope: str, version: str) -> str:
        return f"{self.prefix}:{SCHEMA}:sem:{scope}:{version}:{self.prompt_version}:{self.model_version}"

    def _doc_set(self, document_id: str) -> str:
        return f"{self.prefix}:doc:{document_id}"

    # ------------------------------------------------------------ operations
    def get(self, scope: str, question: str) -> Optional[dict]:
        """Exact match on the normalised question."""
        try:
            raw = self.client.get(self.make_key(scope, question))
            return json.loads(raw) if raw else None
        except Exception as exc:
            logger.warning("Cache read skipped (Redis problem): %s", exc)
            return None

    def get_semantic(self, scope: str, question: str, vector) -> Optional[SemanticHit]:
        """Best earlier question of this scope with similarity >= threshold (and the same numbers)."""
        if not self.semantic_enabled or vector is None:
            return None
        try:
            version = self._index_version()
            entries = self.client.hgetall(self._vectors_key(scope, version))
            if not entries:
                return None
            query = _unit(np.asarray(vector, dtype=np.float32))
            signature = number_signature(question)
            candidates = []
            for qhash, raw in entries.items():
                try:
                    entry = json.loads(raw)
                    stored = np.frombuffer(base64.b64decode(entry["v"]), dtype=np.float32)
                except (ValueError, KeyError, TypeError):
                    continue
                if stored.shape != query.shape:
                    continue
                similarity = float(np.dot(query, stored))
                if similarity >= self.semantic_threshold and number_signature(entry["q"]) == signature:
                    candidates.append((similarity, qhash, entry["q"]))
            for similarity, qhash, matched in sorted(candidates, reverse=True)[:5]:
                raw = self.client.get(self._key(scope, version, qhash))
                if raw:
                    return SemanticHit(json.loads(raw), round(similarity, 4), matched)
                self.client.hdel(self._vectors_key(scope, version), qhash)      # expired or invalidated
            return None
        except Exception as exc:
            logger.warning("Semantic cache read skipped (Redis problem): %s", exc)
            return None

    def set(self, scope: str, question: str, value: dict[str, Any], ttl: int, document_ids: Iterable[str],
            vector=None) -> bool:
        try:
            version = self._index_version()
            qhash = self._qhash(question)
            key = self._key(scope, version, qhash)
            pipe = self.client.pipeline()
            pipe.set(key, json.dumps(value, ensure_ascii=False), ex=ttl)
            for doc_id in set(document_ids):
                pipe.sadd(self._doc_set(doc_id), key)
                pipe.expire(self._doc_set(doc_id), max(ttl, 60) + 3600)
            if self.semantic_enabled and vector is not None:
                vkey = self._vectors_key(scope, version)
                entry = {"q": normalise_question(question), "t": time.time(),
                         "v": base64.b64encode(_unit(np.asarray(vector, dtype=np.float32)).tobytes()).decode()}
                pipe.hset(vkey, qhash, json.dumps(entry))
                pipe.expire(vkey, max(ttl, 60) + 3600)
            pipe.execute()
            if self.semantic_enabled and vector is not None:
                self._prune(self._vectors_key(scope, version))
            return True
        except Exception as exc:
            logger.warning("Cache write skipped (Redis problem): %s", exc)
            return False

    def _prune(self, vectors_key: str) -> None:
        """Keep at most ``semantic_max_entries`` question vectors per scope (drop the oldest)."""
        extra = self.client.hlen(vectors_key) - self.semantic_max_entries
        if extra <= 0:
            return
        ages = sorted((json.loads(raw).get("t", 0), qhash) for qhash, raw in self.client.hgetall(vectors_key).items())
        self.client.hdel(vectors_key, *[qhash for _, qhash in ages[:extra]])

    def invalidate_document(self, document_id: str) -> int:
        """Delete every cached answer that cited ``document_id``."""
        try:
            set_key = self._doc_set(document_id)
            keys = list(self.client.smembers(set_key))
            deleted = self.client.delete(*keys) if keys else 0
            self.client.delete(set_key)
            return int(deleted)
        except Exception as exc:
            logger.warning("Cache invalidation failed (the index version still protects against stale answers): %s", exc)
            return 0

    def bump_index_version(self) -> int:
        try:
            return int(self.client.incr(self._version_key))
        except Exception as exc:
            logger.warning("Could not bump the cache index version: %s", exc)
            return 0

    def ping(self) -> bool:
        try:
            return bool(self.client.ping())
        except Exception:
            return False


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return (v / n).astype(np.float32) if n else v.astype(np.float32)


def _short(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:8]


def build_cache(settings: Settings, prompt_version: str, model_version: str, client=None) -> Cache:
    """Create the cache from the settings (a client can be injected for tests)."""
    if not settings.cache_enabled:
        return NullCache()
    if client is None:
        if not settings.redis_url:
            logger.warning("REDIS_URL is not set: answers are not cached")
            return NullCache()
        import redis
        client = redis.Redis.from_url(settings.redis_url.get_secret_value(), decode_responses=True,
                                      socket_connect_timeout=2, socket_timeout=2)
    return RedisAnswerCache(client, settings.cache_key_prefix, prompt_version, model_version,
                            settings.semantic_cache_enabled, settings.semantic_cache_threshold,
                            settings.semantic_cache_max_entries)
