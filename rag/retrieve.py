"""Translate a question to Dutch, fetch candidate chunks, rerank them.

Retrieval is two-stage. multilingual-e5 pulls a wide candidate set from
Chroma; a cross-encoder then reads each (query, passage) pair and scores
relevance. Only passages at or above RERANK_THRESHOLD are returned, so an
empty result means the corpus has nothing on the question.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import chromadb
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from rag import llm  # noqa: E402

N_CANDIDATES = 30
TOP_K = 5
MAX_PER_RULING = 2

TRANSLATE_PROMPT = """Translate this question into Dutch, using the legal terms a Dutch court would use.
Output only the Dutch question, nothing else.

Question: {question}"""


@dataclass
class Hit:
    ecli: str
    court: str
    date: str
    section: str
    text: str
    score: float
    rerank: float = 0.0


class Retriever:
    def __init__(self) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(config.EMBED_MODEL, device=device)
        self.reranker = CrossEncoder(config.RERANK_MODEL, device=device)
        client = chromadb.PersistentClient(path=str(config.RAG_INDEX_DIR))
        self.collection = client.get_collection(config.RAG_COLLECTION)

    def translate(self, question: str) -> str:
        dutch = llm.generate(TRANSLATE_PROMPT.format(question=question))
        return dutch.strip().strip('"').splitlines()[0].strip()

    def candidates(self, queries: list[str]) -> list[Hit]:
        """N_CANDIDATES nearest chunks per query by cosine; each chunk keeps its best score."""
        vectors = self.model.encode(["query: " + q for q in queries], normalize_embeddings=True)
        result = self.collection.query(query_embeddings=vectors.tolist(), n_results=N_CANDIDATES)
        best: dict[str, Hit] = {}
        for ids, metas, docs, dists in zip(
            result["ids"], result["metadatas"], result["documents"], result["distances"]
        ):
            for chunk_id, meta, doc, dist in zip(ids, metas, docs, dists):
                if chunk_id not in best or best[chunk_id].score < 1 - dist:
                    best[chunk_id] = Hit(ecli=meta["ecli"], court=meta["court"], date=meta["date"],
                                         section=meta["section"], text=doc, score=1 - dist)
        return list(best.values())

    def search(self, question: str, dutch_query: str) -> list[Hit]:
        """Top TOP_K reranked passages at or above the threshold, MAX_PER_RULING per ruling.

        The original question is searched next to its translation, so a passage
        is still found when the translation gets a legal term wrong.
        """
        queries = [dutch_query, question]
        pool = self.candidates(queries)
        scores = self.reranker.predict(
            [(q, h.text) for h in pool for q in queries], batch_size=32,
            show_progress_bar=False, activation_fct=torch.nn.Sigmoid(),
        )
        for i, hit in enumerate(pool):
            hit.rerank = float(max(scores[i * len(queries) : (i + 1) * len(queries)]))

        hits: list[Hit] = []
        per_ruling: dict[str, int] = {}
        for hit in sorted(pool, key=lambda h: h.rerank, reverse=True):
            if hit.rerank < config.RERANK_THRESHOLD:
                break
            if per_ruling.get(hit.ecli, 0) >= MAX_PER_RULING:
                continue
            per_ruling[hit.ecli] = per_ruling.get(hit.ecli, 0) + 1
            hits.append(hit)
            if len(hits) == TOP_K:
                break
        return hits
