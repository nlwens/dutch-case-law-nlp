"""Embed every chunk with multilingual-e5 and store it in a local Chroma index.

The index is rebuilt from scratch on each run. Each rechtsgebied becomes a
boolean metadata field ("rg:Strafrecht": True) so retrieval can filter on it.
Topic tags are left out on purpose: they are the evaluation ground truth.

Run from the project root:
    .venv\\Scripts\\python.exe -m rag.build_index
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import chromadb
import torch
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

BATCH_SIZE = 64
ADD_BATCH = 2000


def metadata(chunk: dict) -> dict:
    meta = {
        "ecli": chunk["ecli"],
        "court": chunk["court"],
        "date": chunk["date"],
        "section": chunk["section"],
        "heading": chunk["heading"],
    }
    for label in chunk["rechtsgebieden"]:
        meta[f"rg:{label}"] = True
    return meta


def main() -> None:
    with open(config.RAG_CHUNKS, encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(config.EMBED_MODEL, device=device)
    if device == "cuda":
        model.half()

    print(f"Embedding {len(chunks)} chunks on {device} ...")
    start = time.time()
    vectors = model.encode(
        ["passage: " + c["text"] for c in chunks],
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    print(f"  done in {time.time() - start:.0f}s")

    client = chromadb.PersistentClient(path=str(config.RAG_INDEX_DIR))
    try:
        client.delete_collection(config.RAG_COLLECTION)
    except Exception:
        pass
    collection = client.create_collection(
        config.RAG_COLLECTION, metadata={"hnsw:space": "cosine"}
    )
    for i in range(0, len(chunks), ADD_BATCH):
        batch = chunks[i : i + ADD_BATCH]
        collection.add(
            ids=[c["chunk_id"] for c in batch],
            embeddings=vectors[i : i + ADD_BATCH].tolist(),
            documents=[c["text"] for c in batch],
            metadatas=[metadata(c) for c in batch],
        )
    print(f"Index: {collection.count()} chunks -> {config.RAG_INDEX_DIR}")


if __name__ == "__main__":
    main()
