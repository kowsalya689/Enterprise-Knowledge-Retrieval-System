"""Authentication (API keys) and role -> access-tag mapping.

* ``data/roles.json`` maps each role to the access tags it holds.
* ``API_KEYS`` in ``.env`` maps each secret key to a role: ``key1:employee,key2:hr``.
* The role is always taken from the server-side key table, never from the request body,
  so a user cannot ask for more access than the key grants.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from app.config import Settings

logger = logging.getLogger(__name__)


class AuthenticationError(Exception):
    """Missing or invalid API key."""


class PermissionDeniedError(Exception):
    """Valid user, but not allowed to do this."""


class ConfigurationError(Exception):
    """Security configuration is missing or inconsistent."""


@dataclass(frozen=True)
class Principal:
    """The authenticated caller: who they are and which content they may see."""
    user: str
    role: str
    access_tags: frozenset[str]

    @property
    def scope(self) -> str:
        """Stable fingerprint of the access tags; part of every cache key."""
        return hashlib.sha256(",".join(sorted(self.access_tags)).encode()).hexdigest()[:16]


def load_roles(path: Path) -> dict[str, frozenset[str]]:
    path = Path(path)
    if not path.is_file():
        raise ConfigurationError(f"Roles file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    roles = {name: frozenset(tags) for name, tags in data.get("roles", {}).items()}
    if not roles or any(not tags for tags in roles.values()):
        raise ConfigurationError("roles.json must define at least one role and every role needs at least one tag")
    return roles


def parse_api_keys(raw: str) -> dict[str, str]:
    """'key1:employee,key2:hr' -> {'key1': 'employee', 'key2': 'hr'}"""
    pairs: dict[str, str] = {}
    for item in (raw or "").split(","):
        item = item.strip()
        if not item:
            continue
        key, sep, role = item.rpartition(":")
        if not sep or not key or not role:
            raise ConfigurationError("API_KEYS must look like 'key1:role1,key2:role2'")
        pairs[key.strip()] = role.strip()
    return pairs


class Authenticator:
    def __init__(self, roles: dict[str, frozenset[str]], keys: dict[str, str]):
        unknown = sorted({r for r in keys.values() if r not in roles})
        if unknown:
            raise ConfigurationError(f"API_KEYS refer to unknown role(s): {', '.join(unknown)}")
        self.roles = roles
        # only a hash of each key is kept in memory
        self._by_digest = {self._digest(k): r for k, r in keys.items()}

    @staticmethod
    def _digest(key: str) -> str:
        return hashlib.sha256(key.encode()).hexdigest()

    @classmethod
    def from_settings(cls, settings: Settings) -> "Authenticator":
        raw = settings.api_keys.get_secret_value() if settings.api_keys else ""
        keys = parse_api_keys(raw)
        if not keys:
            logger.warning("API_KEYS is empty: every protected endpoint will answer 401")
        auth = cls(load_roles(settings.roles_path), keys)
        if settings.web_ui_role and settings.web_ui_role not in auth.roles:
            raise ConfigurationError(f"WEB_UI_ROLE '{settings.web_ui_role}' is not a role in roles.json")
        return auth

    def principal_for_role(self, role: str, label: str = "web") -> Principal:
        """The caller for a request that carries no key (the web pages): a fixed, server-side role."""
        return Principal(user=f"{label}-{role}", role=role, access_tags=self.roles[role])

    @property
    def known_tags(self) -> frozenset[str]:
        return frozenset(t for tags in self.roles.values() for t in tags)

    def authenticate(self, api_key: Optional[str]) -> Principal:
        if not api_key:
            raise AuthenticationError("Missing API key (send it in the X-API-Key header)")
        digest = self._digest(api_key)
        role = self._by_digest.get(digest)
        if role is None:
            raise AuthenticationError("Invalid API key")
        return Principal(user=f"{role}-{digest[:6]}", role=role, access_tags=self.roles[role])
