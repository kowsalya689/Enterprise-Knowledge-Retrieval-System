"""Central settings, read from environment variables and the .env file.

Every tunable value of the pipeline lives here so that nothing is hard-coded
in the processing modules.  Copy ``.env.example`` to ``.env`` and edit it.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ------------------------------------------------------------------ paths
    raw_data_dir: Path = PROJECT_ROOT / "data" / "raw"
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"
    inventory_path: Path = PROJECT_ROOT / "data" / "processed" / "document_inventory.csv"

    # -------------------------------------------------------------- ingestion
    max_file_size_mb: int = 50
    chunk_max_words: int = 380          # hard limit, title path included
    chunk_overlap_words: int = 30       # only used when a section must be split
    table_rows_per_chunk: int = 12      # spreadsheets and CSV
    docx_page_mode: Literal["auto", "off"] = "auto"
    soffice_path: Optional[str] = None  # LibreOffice, only used to find DOCX pages

    # ------------------------------------------------------------- embeddings
    embedding_model: str = "nomic-ai/nomic-embed-text-v2-moe"
    embedding_dim: Optional[int] = 768  # 768 = full size; 256 is also supported by Nomic v2
    embedding_doc_prefix: str = "search_document: "
    embedding_query_prefix: str = "search_query: "
    embedding_batch_size: int = 8
    embedding_max_seq_length: int = 512
    embedding_trust_remote_code: bool = True
    embedding_device: Optional[str] = None  # None = automatic (cuda if available, else cpu)

    # -------------------------------------------------------- api and security
    api_keys: Optional[SecretStr] = None        # "key1:employee,key2:hr" (see .env.example)
    roles_path: Path = PROJECT_ROOT / "data" / "roles.json"
    # Role used by the two web pages (Ask, Upload). The pages never ask for or store a key.
    # Empty = pages switched off (use the API with a key). Keep it at the least-privileged role.
    web_ui_role: str = "employee"
    upload_dir: Path = PROJECT_ROOT / "data" / "documents"

    # -------------------------------------------------------------- retrieval
    retrieval_top_k: int = 8
    relevance_threshold: float = 0.30   # minimum cosine score of a useful chunk (final task: 0.30; calibrate, see README)
    min_evidence_chunks: int = 1        # chunks above the threshold needed to attempt an answer
    max_context_words: int = 1800       # words of context sent to the language model
    strict_citations: bool = False      # True: an answer without [n] markers becomes "not found"

    # ----------------------------------------------------------------- ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"
    ollama_timeout: int = 180
    ollama_temperature: float = 0.0
    ollama_num_ctx: int = 4096
    ollama_keep_alive: str = "24h"      # keep the model in memory; the default (5 min) reloads it after every pause
    ollama_num_predict: int = 300        # maximum answer length in tokens (a short answer finishes sooner)

    # ------------------------------------------------------------------ redis
    redis_url: Optional[SecretStr] = None       # redis://localhost:6379/0 (may contain a password)
    cache_enabled: bool = True
    cache_ttl_answer_seconds: int = 86400       # grounded answers: 24 hours ("not found" is never cached)
    cache_key_prefix: str = "rag"
    semantic_cache_enabled: bool = True         # reuse an answer for a differently worded but same-meaning question
    semantic_cache_threshold: float = 0.90      # minimum cosine similarity between two questions
    semantic_cache_max_entries: int = 300       # per access scope; the oldest are dropped first

    # ----------------------------------------------------------------- qdrant
    qdrant_url: Optional[str] = None
    qdrant_api_key: Optional[SecretStr] = None
    qdrant_local_path: Optional[Path] = None  # on-disk mode, no server and no cloud account
    qdrant_collection: str = "enterprise_kb"
    qdrant_batch_size: int = 64
    qdrant_timeout: int = 60

    @field_validator(
        "soffice_path", "embedding_device", "embedding_dim",
        "qdrant_url", "qdrant_api_key", "qdrant_local_path", "api_keys", "redis_url",
        mode="before",
    )
    @classmethod
    def _blank_to_none(cls, value):
        """An empty value in .env (``QDRANT_URL=``) means 'not set'."""
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
