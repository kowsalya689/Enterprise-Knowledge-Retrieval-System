"""Run the 10 final-task questions against a RUNNING system and write the result table.

    python scripts/run_ten_questions.py --api-key <employee key>
    python scripts/run_ten_questions.py --base-url http://localhost:8001 --api-key demo-employee \
        --label "OFFLINE DEMO (stand-in models, not real results)" --output docs/ten_question_results_OFFLINE_DEMO

Reads  data/final_test_questions.csv  (7 questions about the documents, 3 off-topic).
Writes <output>.md and <output>.csv with: answer, correct?, source cited, top score, cached.

"Correct?" is an automatic first check, YOU still read the answer:
  * in-documents question -> answered, every expected keyword in the answer, expected document cited;
  * off-topic question    -> status not_found and no source.
A cache check at the end repeats question T01 (exact hit expected) and asks it in other words
(semantic hit expected only if the similarity is at or above SEMANTIC_CACHE_THRESHOLD).
"""
from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
REWORDED_T01 = "What is the yearly annual leave entitlement for an employee?"


def ask(client: httpx.Client, question: str, debug: bool = False) -> dict:
    r = client.post("/query", json={"question": question, "debug": debug})
    if r.status_code != 200:
        raise SystemExit(f"POST /query failed ({r.status_code}): {r.text[:300]}")
    return r.json()


def judge(row: dict, resp: dict) -> tuple[bool, str]:
    cited = {c["document_id"] for c in resp["citations"]}
    if row["kind"] == "off_topic":
        ok = resp["status"] == "not_found" and not cited
        return ok, "" if ok else "should be 'not found' but the system answered"
    if resp["status"] != "answered":
        return False, f"not found ({resp.get('reason')})"
    missing = [k for k in row["expected_keywords"].split("|") if k and k.lower() not in resp["answer"].lower()]
    notes = []
    if missing:
        notes.append("answer lacks: " + ", ".join(missing))
    if row["expected_document_id"] not in cited:
        notes.append(f"expected source {row['expected_document_id']} not cited")
    return not notes, "; ".join(notes)


def source_text(resp: dict) -> str:
    if not resp["citations"]:
        return "-"
    parts = []
    for c in resp["citations"]:
        where = ", ".join(x for x in (c["section"], f"p.{c['page']}" if c.get("page") else None,
                                      f"sheet {c['sheet']}" if c.get("sheet") else None) if x)
        parts.append(f"{c['document_name']} ({where})")
    return "; ".join(parts)


def cell(text: str, limit: int = 160) -> str:
    text = " ".join(str(text).split()).replace("|", "/")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--api-key", required=True, help="a key whose role is 'employee' (the questions were chosen for it)")
    ap.add_argument("--questions", default=str(ROOT / "data" / "final_test_questions.csv"))
    ap.add_argument("--output", default=str(ROOT / "docs" / "ten_question_results"), help="path without extension")
    ap.add_argument("--label", default="", help="text printed at the top of the report")
    ap.add_argument("--timeout", type=float, default=300)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.questions, encoding="utf-8")))
    results = []
    with httpx.Client(base_url=args.base_url, headers={"X-API-Key": args.api_key}, timeout=args.timeout) as client:
        health = client.get("/health").json()
        threshold = None
        for row in rows:
            resp = ask(client, row["question"])
            ok, note = judge(row, resp)
            results.append({**row, "status": resp["status"], "answer": resp["answer"], "correct": "yes" if ok else "NO",
                            "note": note, "source": source_text(resp), "top_score": resp.get("top_score"),
                            "cached": resp.get("cache_match") or "no", "ms": resp["timings_ms"].get("total")})
            print(f"{row['test_id']}  {resp['status']:<9} top={resp.get('top_score')}  correct={'yes' if ok else 'NO'}")
        again = ask(client, rows[0]["question"])
        reworded = ask(client, REWORDED_T01)
        cache_check = [("T01 asked again (same words)", again), (f"T01 reworded: “{REWORDED_T01}”", reworded)]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["test_id", "kind", "question", "status", "answer", "correct", "note", "source",
                                          "top_score", "cached", "ms", "expected_answer"], extrasaction="ignore")
        w.writeheader(); w.writerows(results)

    doc_rows = [r for r in results if r["kind"] == "in_documents"]
    off_rows = [r for r in results if r["kind"] == "off_topic"]
    lines = ["# Ten-question test", ""]
    if args.label:
        lines += [f"> **{args.label}**", ""]
    lines += [f"Run: {datetime.now():%Y-%m-%d %H:%M} · API: {args.base_url} · system health: {health.get('status')}", "",
              "| # | Kind | Question | Answer given | Correct? | Source cited | Top score | Cached |",
              "|---|---|---|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['test_id']} | {r['kind'].replace('_', ' ')} | {cell(r['question'], 90)} | {cell(r['answer'])} | "
                     f"{r['correct']}{' – ' + cell(r['note'], 60) if r['note'] else ''} | {cell(r['source'], 110)} | "
                     f"{r['top_score']} | {r['cached']} |")
    lines += ["", f"**Documents questions answered correctly: {sum(r['correct'] == 'yes' for r in doc_rows)}/{len(doc_rows)} · "
              f"off-topic questions correctly refused: {sum(r['correct'] == 'yes' for r in off_rows)}/{len(off_rows)}**", ""]
    answered_scores = [r["top_score"] for r in doc_rows if r["top_score"] is not None]
    refused_scores = [r["top_score"] for r in off_rows if r["top_score"] is not None]
    if answered_scores and refused_scores:
        lines += [f"Top scores: lowest document question {min(answered_scores)}, highest off-topic question {max(refused_scores)} "
                  f"(a good cutoff lies between them; the cutoff in use is `RELEVANCE_THRESHOLD`, default 0.30).", ""]
    lines += ["## Cache check", "", "| Request | Status | Cached | Similarity |", "|---|---|---|---|"]
    for name, resp in cache_check:
        lines.append(f"| {cell(name, 100)} | {resp['status']} | {resp.get('cache_match') or 'no'} | {resp.get('cache_similarity') or '-'} |")
    lines += ["", "Expected: the first line is an **exact** hit. The reworded line is a **semantic** hit only when the question-to-question "
              "similarity is at least `SEMANTIC_CACHE_THRESHOLD` (0.90); otherwise it is answered normally, which is also correct behaviour.", ""]
    out.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWritten: {out.with_suffix('.md')} and {out.with_suffix('.csv')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
