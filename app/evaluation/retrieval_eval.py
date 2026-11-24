"""Retrieval quality and threshold calibration on the labelled questions.

For every question in ``data/evaluation_questions.csv`` the question is searched **as its
role** and the result is compared with the expected document and section::

    python -m app.evaluation.retrieval_eval

Output: Recall@5, Recall@10, MRR, access-control leaks (must be 0), and the top-1 scores of
answerable versus unanswerable questions, with a suggested ``RELEVANCE_THRESHOLD``.
Needs the dataset to be ingested first (and uses the real embedding model).
"""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Optional

from app.config import PROJECT_ROOT, Settings, get_settings
from app.security.permissions import load_roles


def _matches(payload: dict, doc_id: str, section: str) -> bool:
    return payload.get("document_id") == doc_id and payload.get("section_path", "").casefold() == section.casefold()


def evaluate(settings: Settings, embedder, store, questions_path: Optional[Path] = None, k_max: int = 10) -> dict:
    roles = load_roles(settings.roles_path)
    path = questions_path or PROJECT_ROOT / "data" / "evaluation_questions.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    hit5 = hit10 = rr = n = leaks = 0
    answerable_scores: list[float] = []
    unanswerable_scores: list[float] = []
    misses: list[str] = []

    for q in rows:
        tags = sorted(roles[q["user_role"]])
        hits = store.search(embedder.embed_query(q["question"]), tags, limit=k_max)
        leaks += sum(1 for h in hits if not set(h.payload.get("access_tags", [])) & set(tags))
        top = hits[0].score if hits else 0.0
        if q["answerable"] == "no" or q["permission_test_case"] == "denied":
            unanswerable_scores.append(top)              # nothing the user may see answers it
            continue
        answerable_scores.append(top)
        sections = [x.strip() for x in q["expected_section"].split(" ; ")]
        expected = [(q["expected_document_id"], sec) for sec in sections]
        n += 1
        ranks = [next((i + 1 for i, h in enumerate(hits) if _matches(h.payload, d, s)), None) for d, s in expected]
        found5 = sum(1 for r in ranks if r and r <= 5) / len(ranks)
        found10 = sum(1 for r in ranks if r and r <= 10) / len(ranks)
        hit5 += found5
        hit10 += found10
        first = min((r for r in ranks if r), default=None)
        rr += 1 / first if first else 0
        if not first:
            misses.append(q["question_id"])

    def pct(values, p):
        values = sorted(values)
        return values[min(len(values) - 1, int(p * len(values)))] if values else 0.0

    low_answerable = pct(answerable_scores, 0.10)
    high_unanswerable = pct(unanswerable_scores, 0.90)
    return {
        "embedding_model": getattr(embedder, "name", "?"),
        "answerable_questions": n,
        "recall_at_5": round(hit5 / n, 3) if n else None,
        "recall_at_10": round(hit10 / n, 3) if n else None,
        "mrr": round(rr / n, 3) if n else None,
        "access_control_leaks": leaks,
        "missed_questions": misses,
        "top1_score": {
            "answerable_median": round(statistics.median(answerable_scores), 3) if answerable_scores else None,
            "answerable_p10": round(low_answerable, 3),
            "unanswerable_median": round(statistics.median(unanswerable_scores), 3) if unanswerable_scores else None,
            "unanswerable_p90": round(high_unanswerable, 3),
        },
        "suggested_relevance_threshold": round((low_answerable + high_unanswerable) / 2, 3),
        "note": "A threshold between the answerable p10 and the unanswerable p90 is a starting point; check it against real questions.",
    }


def main() -> int:
    from app.retrieval.embeddings import SentenceTransformerEmbedder
    from app.retrieval.vector_store import QdrantStore

    settings = get_settings()
    result = evaluate(settings, SentenceTransformerEmbedder(settings), QdrantStore(settings))
    # one file per collection, so that two embedding models (for example Nomic v2 and BGE-small) can be compared
    out = settings.processed_dir / f"retrieval_eval_{settings.qdrant_collection}.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"\nSaved to {out}")
    return 0 if result["access_control_leaks"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
