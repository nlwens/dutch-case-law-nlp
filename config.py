"""Paths and settings used by the notebooks, predict.py, and the API."""

from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

try:
    from dotenv import load_dotenv

    load_dotenv(BASE_DIR / ".env")
except ImportError:
    pass

DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RAG_DIR = DATA_DIR / "rag"
RAG_CORPUS = RAG_DIR / "corpus.jsonl"
RAG_CHUNKS = RAG_DIR / "chunks.jsonl"
RAG_INDEX_DIR = RAG_DIR / "chroma"
RAG_COLLECTION = "justid_chunks"
EMBED_MODEL = "intfloat/multilingual-e5-base"
RERANK_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
RERANK_THRESHOLD = 0.5
# Retrieval runs locally; the question and the retrieved passages are sent
# to the Gemini API for translation and answering.
GEMINI_MODEL = "gemini-3.5-flash-lite"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

MODELS_DIR = BASE_DIR / "models"
RG_MODEL_DIR = MODELS_DIR / "rg"
BK_MODEL_DIR = MODELS_DIR / "bk"
RG_HUB_ID = "newnus/justid-rechtsgebieden"
BK_HUB_ID = "newnus/justid-bijzondere-kenmerken"
MAX_LEN = 512
LABEL_THRESHOLD = 0.5

RANDOM_SEED = 42
