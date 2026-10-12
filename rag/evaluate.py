"""Run every question in eval/questions.jsonl and score the RAG pipeline.

A retrieved passage counts as on-topic when its ruling carries the question's
topic tag from build_corpus.py. Questions with topic null must be refused.

Writes results/rag_eval.json (summary) and results/rag_answers.jsonl (one
record per question, for reading the answers by hand).

Run from the project root:
    .venv\\Scripts\\python.exe -m rag.evaluate
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from rag.answer import answer  # noqa: E402
from rag.retrieve import Retriever  # noqa: E402

QUESTIONS = Path(__file__).resolve().parent / "eval" / "questions.jsonl"
RESULTS_DIR = config.BASE_DIR / "results"


def main() -> None:
    with open(config.RAG_CORPUS, encoding="utf-8") as f:
        topics = {row["ecli"]: row["topics"] for row in map(json.loads, f)}
    with open(QUESTIONS, encoding="utf-8") as f:
        questions = [json.loads(line) for line in f]

    retriever = Retriever()
    records = []
    for q in questions:
        start = time.time()
        result = answer(q["question"], retriever)
        seconds = time.time() - start
        on_topic = [q["topic"] in topics[h.ecli] for h in result.hits] if q["topic"] else []
        records.append({
            "id": q["id"],
            "topic": q["topic"],
            "question": q["question"],
            "dutch_query": result.dutch_query,
            "refused": result.refused,
            "answer": result.text,
            "draft": result.draft,
            "sources": [
                {"ecli": h.ecli, "section": h.section, "rerank": round(h.rerank, 3),
                 "on_topic": ok}
                for h, ok in zip(result.hits, on_topic or [None] * len(result.hits))
            ],
            "cited": result.cited,
            "invalid_citations": result.invalid_citations,
            "seconds": round(seconds, 1),
        })
        print(f"{q['id']} refused={result.refused!s:5} kept={len(result.hits)} "
              f"on_topic={sum(on_topic)} cited={len(result.cited)} "
              f"invalid={len(result.invalid_citations)} {seconds:.0f}s", flush=True)

    topical = [r for r in records if r["topic"]]
    refusal = [r for r in records if not r["topic"]]
    answered = [r for r in records if not r["refused"]]
    all_cited = sum(len(r["cited"]) for r in answered)
    kept = [s for r in topical for s in r["sources"]]

    summary = {
        "questions": len(records),
        "topical_questions": len(topical),
        "refusal_questions": len(refusal),
        "retrieval_hit_rate": sum(any(s["on_topic"] for s in r["sources"]) for r in topical) / len(topical),
        "retrieval_precision": sum(s["on_topic"] for s in kept) / len(kept) if kept else 0.0,
        "topical_answered": sum(not r["refused"] for r in topical) / len(topical),
        "refusal_correct": sum(r["refused"] for r in refusal) / len(refusal),
        "answers_with_citation": sum(bool(r["cited"]) for r in answered) / len(answered) if answered else 0.0,
        "invalid_citation_rate": sum(len(r["invalid_citations"]) for r in answered) / all_cited if all_cited else 0.0,
        "median_seconds": statistics.median(r["seconds"] for r in records),
        "settings": {
            "embed_model": config.EMBED_MODEL,
            "rerank_model": config.RERANK_MODEL,
            "rerank_threshold": config.RERANK_THRESHOLD,
            "llm": config.GEMINI_MODEL,
        },
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "rag_eval.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    with open(RESULTS_DIR / "rag_answers.jsonl", "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
