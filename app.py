"""FastAPI service around the two JustID BERTje classifiers and the RAG search.

Classifier endpoints match the 2024 delivery: paste text, or upload a PDF.
/ask answers a question from the indexed rulings; it is only
available when data/rag/chroma has been built (see rag/build_index.py).
"""

from __future__ import annotations

import io
from contextlib import asynccontextmanager

import pdfplumber
import requests
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

import config
from predict import JustIdModels
from rag.answer import answer
from rag.retrieve import Retriever

_models: JustIdModels | None = None
_retriever: Retriever | None = None


class TextInput(BaseModel):
    text: str = Field(..., description="Dutch ruling text")


class PredictionOutput(BaseModel):
    rechtsgebieden: dict[str, float]
    bijzondere_kenmerken: dict[str, float]


class AskInput(BaseModel):
    question: str = Field(..., min_length=3, description="Question in any language; the answer uses the same language")


class Source(BaseModel):
    ecli: str
    court: str
    date: str
    section: str
    rerank: float
    text: str


class AskOutput(BaseModel):
    answer: str
    refused: bool
    dutch_query: str
    sources: list[Source]
    invalid_citations: list[str]


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _models, _retriever
    _models = JustIdModels()
    if config.RAG_INDEX_DIR.exists():
        _retriever = Retriever()
    yield
    _models = None
    _retriever = None


app = FastAPI(
    title="JustID",
    description="Multi-label rechtsgebieden and bijzondere kenmerken for Dutch rulings, "
    "and question answering over indexed rulings.",
    version="2026.0",
    lifespan=lifespan,
)


def _predict(text: str) -> PredictionOutput:
    if _models is None:
        raise HTTPException(status_code=503, detail="Models are not loaded")
    try:
        result = _models.predict(text)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return PredictionOutput(**result)


def _pdf_text(upload: UploadFile) -> str:
    name = (upload.filename or "").lower()
    if not name.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    data = upload.file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Could not read the PDF") from exc

    text = "\n".join(pages).strip()
    if not text:
        raise HTTPException(status_code=400, detail="No text could be extracted from the PDF")
    return text


@app.get("/")
def root() -> dict:
    return {
        "service": "JustID",
        "endpoints": {
            "predict_text": "POST /predict-text",
            "predict_document": "POST /predict-document",
            "ask": "POST /ask",
            "health": "GET /health",
            "docs": "GET /docs",
        },
    }


@app.get("/health")
def health() -> dict:
    ready = _models is not None
    return {
        "status": "healthy" if ready else "unhealthy",
        "model_loaded": ready,
        "rag_loaded": _retriever is not None,
        "device": None if _models is None else str(_models.device),
    }


@app.post("/predict-text", response_model=PredictionOutput)
def predict_text(payload: TextInput) -> PredictionOutput:
    return _predict(payload.text)


@app.post("/predict-document", response_model=PredictionOutput)
def predict_document(file: UploadFile = File(...)) -> PredictionOutput:
    return _predict(_pdf_text(file))


@app.post("/ask", response_model=AskOutput)
def ask(payload: AskInput) -> AskOutput:
    if _retriever is None:
        raise HTTPException(status_code=503, detail="RAG index is not built")
    if not config.GEMINI_API_KEY:
        raise HTTPException(status_code=503, detail="GEMINI_API_KEY is not set")
    try:
        result = answer(payload.question, _retriever)
    except (requests.RequestException, RuntimeError) as exc:
        raise HTTPException(status_code=502, detail="Gemini request failed") from exc
    return AskOutput(
        answer=result.text,
        refused=result.refused,
        dutch_query=result.dutch_query,
        sources=[
            Source(ecli=h.ecli, court=h.court, date=h.date, section=h.section,
                   rerank=round(h.rerank, 3), text=h.text)
            for h in result.hits
        ],
        invalid_citations=result.invalid_citations,
    )


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
