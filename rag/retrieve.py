"""Translate a question to Dutch, fetch candidate chunks, rerank them.

Retrieval is two-stage. multilingual-e5 pulls a wide candidate set from
Chroma; a cross-encoder then reads each (question, passage) pair and scores
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

TRANSLATE_PROMPT = """Translate this question into Dutch, using the legal terms a Dutch court would use.
Output only the Dutch question, nothing else.

Question: {question}"""


@dataclass
class Hit:
    chunk_id: str
    ecli: str
    court: str
    date: str
    section: str
    heading: str
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

    def candidates(
        self, queries: list[str], n: int, rechtsgebied: str | None = None
    ) -> list[Hit]:
        """Vector search with every query; each chunk keeps its best score."""
        vectors = self.model.encode(
            ["query: " + q for q in queries], normalize_embeddings=True
        )
        result = self.collection.query(
            query_embeddings=vectors.tolist(),
            n_results=n,
            where={f"rg:{rechtsgebied}": True} if rechtsgebied else None,
        )
        best: dict[str, Hit] = {}
        for ids, metas, docs, dists in zip(
            result["ids"], result["metadatas"], result["documents"], result["distances"]
        ):
            for chunk_id, meta, doc, dist in zip(ids, metas, docs, dists):
                score = 1 - dist
                if chunk_id in best and best[chunk_id].score >= score:
                    continue
                best[chunk_id] = Hit(
                    chunk_id=chunk_id, ecli=meta["ecli"], court=meta["court"],
                    date=meta["date"], section=meta["section"], heading=meta["heading"],
                    text=doc, score=score,
                )
        return list(best.values())

    def search(
        self,
        question: str,
        dutch_query: str,
        k: int = 5,
        rechtsgebied: str | None = None,
        n_candidates: int = 30,
        max_per_ruling: int = 2,
        threshold: float = config.RERANK_THRESHOLD,
    ) -> list[Hit]:
        """Top-k reranked passages at or above the threshold, capped per ruling.

        Both the original question and its translation go into the vector
        search and the reranker, so a detail lost in translation can still be
        matched by the original.
        """
        pool = self.candidates([dutch_query, question], n_candidates, rechtsgebied)
        if not pool:
            return []
        queries = [dutch_query, question]
        pairs = [(q, h.text) for h in pool for q in queries]
        scores = self.reranker.predict(
            pairs, batch_size=32, show_progress_bar=False,
            activation_fct=torch.nn.Sigmoid(),
        )
        for i, hit in enumerate(pool):
            hit.rerank = float(max(scores[i * len(queries) : (i + 1) * len(queries)]))

        hits: list[Hit] = []
        per_ruling: dict[str, int] = {}
        for hit in sorted(pool, key=lambda h: h.rerank, reverse=True):
            if hit.rerank < threshold:
                break
            if per_ruling.get(hit.ecli, 0) >= max_per_ruling:
                continue
            per_ruling[hit.ecli] = per_ruling.get(hit.ecli, 0) + 1
            hits.append(hit)
            if len(hits) == k:
                break
        return hits
