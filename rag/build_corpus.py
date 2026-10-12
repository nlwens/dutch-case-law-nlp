"""Build the RAG corpus from the test and natural_test splits.

Each topic is a keyword rule on the ruling text. Rulings that match a topic
are kept (up to TOPIC_CAP per topic), the rest of the corpus is filled with
random rulings that match no topic. Rulings mentioning a refusal subject are
dropped, so the refusal questions in eval/questions.jsonl cannot be answered.

Run from the project root:
    .venv\\Scripts\\python.exe -m rag.build_corpus
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

SPLITS = ["test", "natural_test"]
CORPUS_SIZE = 2500
TOPIC_CAP = 100

# (rechtsgebied the ruling must carry, keywords that must all appear,
#  minimum count of the first keyword)
TOPICS = {
    "tenancy": ("Civiel recht", ["verhuurder", "huurovereenkomst"], 3),
    "summary_dismissal": ("Civiel recht", ["ontslag op staande voet"], 3),
    "bijstand_recovery": ("Socialezekerheidsrecht", ["bijstand", "terugvordering"], 3),
    "asylum_credibility": ("Vreemdelingenrecht", ["geloofwaardig", "asielaanvraag"], 3),
    "woz_value": ("Belastingrecht", ["woz-waarde"], 3),
    "contact_arrangement": ("Personen- en familierecht", ["omgangsregeling"], 3),
    "child_support": ("Personen- en familierecht", ["kinderalimentatie"], 3),
    "drink_driving": ("Strafrecht", ["onder invloed", "wegenverkeerswet"], 1),
    "environmental_permit": ("Bestuursrecht", ["omgevingsvergunning"], 3),
    "tbs": ("Strafrecht", ["terbeschikkingstelling"], 3),
    "debt_restructuring": ("Civiel recht", ["schuldsaneringsregeling"], 3),
    "wia": ("Socialezekerheidsrecht", ["wia-uitkering"], 1),
}

REFUSAL_KEYWORDS = [
    "261/2004", "vluchtvertraging", "luchtvaartmaatschappij",
    "transfervergoeding", "transfersom",
    "zonnepaneel", "zonnepanelen",
    "e-scooter", "elektrische step",
    "bitcoin", "cryptovaluta", "cryptomunt",
]


def matched_topics(row: dict) -> list[str]:
    low = row["text"].lower()
    found = []
    for name, (area, keywords, min_count) in TOPICS.items():
        if area not in row["rechtsgebieden"]:
            continue
        if low.count(keywords[0]) >= min_count and all(k in low for k in keywords[1:]):
            found.append(name)
    return found


def main() -> None:
    rng = random.Random(config.RANDOM_SEED)

    rows = []
    for split in SPLITS:
        with open(config.PROCESSED_DIR / f"{split}.jsonl", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                low = row["text"].lower()
                if any(k in low for k in REFUSAL_KEYWORDS):
                    continue
                row["split"] = split
                row["topics"] = matched_topics(row)
                rows.append(row)

    rng.shuffle(rows)
    picked: dict[str, dict] = {}
    for name in TOPICS:
        members = [r for r in rows if name in r["topics"]]
        for row in members[:TOPIC_CAP]:
            picked[row["ecli"]] = row

    fillers = [r for r in rows if not r["topics"]]
    for row in fillers[: max(0, CORPUS_SIZE - len(picked))]:
        picked[row["ecli"]] = row

    config.RAG_DIR.mkdir(parents=True, exist_ok=True)
    keep = ["ecli", "court", "date", "split", "rechtsgebieden",
            "bijzondere_kenmerken", "topics", "text"]
    with open(config.RAG_CORPUS, "w", encoding="utf-8") as f:
        for row in sorted(picked.values(), key=lambda r: r["ecli"]):
            f.write(json.dumps({k: row[k] for k in keep}, ensure_ascii=False) + "\n")

    per_topic = Counter(t for r in picked.values() for t in r["topics"])
    print(f"Corpus: {len(picked)} rulings -> {config.RAG_CORPUS}")
    print(f"  no topic (distractors): {sum(not r['topics'] for r in picked.values())}")
    for name in TOPICS:
        print(f"  {name:22s} {per_topic[name]}")


if __name__ == "__main__":
    main()
