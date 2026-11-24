"""OFFLINE DEMO of the API (Swagger UI) with stand-ins for the heavy parts.

    python -m tests.demo_server            ->  http://localhost:8001/docs

Uses an in-memory Qdrant filled with the 11 dataset documents, a bag-of-words stand-in for the
embedding model and an extractive stand-in for Ollama.  It shows the API behaviour (upload, access
control, citations, Not found, cache) but NOT the answer quality of the real models.
Keys:  employee = demo-employee   hr = demo-hr   manager = demo-manager   finance = demo-finance
"""
import tempfile
from pathlib import Path

import fakeredis
import uvicorn
from pydantic import SecretStr
from qdrant_client import QdrantClient

from app.cache.redis_cache import RedisAnswerCache
from app.config import Settings
from app.dependencies import assemble_services
from app.ingestion.indexer import Indexer
from app.ingestion.metadata import load_inventory
from app.main import create_app
from app.rag.prompts import PROMPT_VERSION
from app.retrieval.vector_store import QdrantStore
from app.security.permissions import Authenticator, load_roles, parse_api_keys
from tests.doubles import BagOfWordsEmbedder, FakeLLM

ROOT = Path(__file__).resolve().parent.parent


def main(port: int = 8001):
    keys = "demo-employee:employee,demo-hr:hr,demo-manager:manager,demo-finance:finance"
    settings = Settings(_env_file=None, qdrant_collection="demo", embedding_dim=None, docx_page_mode="off",
                        relevance_threshold=0.15, api_keys=SecretStr(keys), upload_dir=Path(tempfile.mkdtemp()))
    store = QdrantStore(settings, client=QdrantClient(":memory:"))
    embedder = BagOfWordsEmbedder()
    Indexer(settings, embedder, store, load_inventory(settings.inventory_path)).ingest_directory(settings.raw_data_dir)
    cache = RedisAnswerCache(fakeredis.FakeRedis(decode_responses=True), "demo", PROMPT_VERSION, "demo")
    auth = Authenticator(load_roles(settings.roles_path), parse_api_keys(keys))
    services = assemble_services(settings, embedder=embedder, store=store, cache=cache, llm=FakeLLM(), authenticator=auth)
    uvicorn.run(create_app(settings, services), host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()
