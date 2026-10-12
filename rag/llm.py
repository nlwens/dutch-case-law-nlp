"""Thin wrapper around the Gemini API."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
RETRY_STATUS = {429, 500, 503}
MAX_RETRIES = 5


def generate(prompt: str, temperature: float = 0.0) -> str:
    if not config.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set. Put it in .env in the project root.")
    for attempt in range(MAX_RETRIES):
        response = requests.post(
            URL.format(model=config.GEMINI_MODEL),
            headers={"x-goog-api-key": config.GEMINI_API_KEY},
            json={
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": temperature},
            },
            timeout=120,
        )
        if response.status_code in RETRY_STATUS and attempt < MAX_RETRIES - 1:
            time.sleep(5 * 2**attempt)
            continue
        response.raise_for_status()
        break

    candidates = response.json().get("candidates", [])
    if not candidates or "content" not in candidates[0]:
        raise RuntimeError(f"Gemini returned no text: {response.json()}")
    parts = candidates[0]["content"].get("parts", [])
    return "".join(p.get("text", "") for p in parts).strip()
