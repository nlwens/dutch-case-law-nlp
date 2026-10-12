"""Answer a question from retrieved ruling passages, with ECLI citations.

Two refusal paths: the reranker keeps no passage (the LLM is not called), or
the LLM's draft says the passages do not address the question. Citations to
rulings outside the given passages are reported as invalid.

Run from the project root:
    .venv\\Scripts\\python.exe -m rag.answer "When do courts extend a TBS order?"
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rag import llm  # noqa: E402
from rag.retrieve import Hit, Retriever  # noqa: E402

NOT_FOUND = "Not found in the provided rulings."

SECTION_LABELS = {
    "court": "COURT'S ASSESSMENT",
    "decision": "DECISION",
    "parties": "PARTY ARGUMENT (not a finding of the court)",
    "facts": "FACTS",
    "procedure": "PROCEDURE",
    "header": "HEADER",
    "advisory": "ADVISORY OPINION of the Advocaat-Generaal (not binding)",
}

ANSWER_PROMPT = """You are a legal research assistant. Answer the question using only the passages from Dutch court rulings below.

Rules:
- Base the answer on the court's assessment and decision. A PARTY ARGUMENT is what one side claimed, not what the court found; if you use it, say it was a party's argument.
- After every claim, cite the ruling by its ECLI in square brackets, exactly as written above the passage, for example [{example}]. Never refer to passage numbers.
- Describe what these rulings show. Do not claim what courts do in general.
- If the passages cover only part of the question, answer that part. Only if no passage is about the question, say so in one sentence.
- Answer in English, in at most 6 sentences.

Passages:
{passages}

Question: {question}
Answer:"""

VERIFY_PROMPT = """Question: {question}

Answer: {answer}

Does this answer actually answer the question, or does it say that the rulings do not address it?
Reply with one word: ANSWERED or NOT_ANSWERED."""

_ECLI = re.compile(r"ECLI:[A-Z]{2}:[A-Z0-9]+:\d{4}:[A-Z0-9]+")


@dataclass
class Answer:
    question: str
    dutch_query: str
    text: str
    draft: str
    hits: list[Hit]
    cited: list[str]
    invalid_citations: list[str]

    @property
    def refused(self) -> bool:
        return self.text == NOT_FOUND


def format_passages(hits: list[Hit]) -> str:
    blocks = []
    for hit in hits:
        label = SECTION_LABELS.get(hit.section, hit.section.upper())
        blocks.append(f"[{hit.ecli}] {hit.court}, {hit.date} | {label}\n{hit.text}")
    return "\n\n".join(blocks)


def answers_question(question: str, draft: str) -> bool:
    reply = llm.generate(VERIFY_PROMPT.format(question=question, answer=draft))
    return not reply.strip().upper().startswith("NOT")


def answer(
    question: str,
    retriever: Retriever,
    k: int = 5,
    rechtsgebied: str | None = None,
) -> Answer:
    dutch = retriever.translate(question)
    hits = retriever.search(question, dutch, k=k, rechtsgebied=rechtsgebied)

    draft = ""
    text = NOT_FOUND
    if hits:
        prompt = ANSWER_PROMPT.format(
            example=hits[0].ecli, passages=format_passages(hits), question=question
        )
        draft = llm.generate(prompt).replace(NOT_FOUND, "").strip()
        if draft and answers_question(question, draft):
            text = draft

    cited = list(dict.fromkeys(_ECLI.findall(text))) if text != NOT_FOUND else []
    allowed = {h.ecli for h in hits}
    return Answer(
        question=question,
        dutch_query=dutch,
        text=text,
        draft=draft,
        hits=hits,
        cited=cited,
        invalid_citations=[e for e in cited if e not in allowed],
    )


def main() -> None:
    question = " ".join(sys.argv[1:]) or input("Question: ")
    result = answer(question, Retriever())
    print(f"\nDutch query: {result.dutch_query}\n")
    print(result.text)
    if result.invalid_citations:
        print(f"\nWarning: cited outside the given passages: {result.invalid_citations}")
    if not result.refused:
        print("\nSources:")
        for hit in result.hits:
            print(f"  {hit.ecli}  {hit.court}, {hit.date}  [{hit.section}]")


if __name__ == "__main__":
    main()
