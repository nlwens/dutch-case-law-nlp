"""Score a Dutch ruling with the two JustID BERTje models.

Loads local `models/rg` and `models/bk` when those folders have weights.
Otherwise it downloads `newnus/justid-rechtsgebieden` and
`newnus/justid-bijzondere-kenmerken` from the Hugging Face Hub.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import config

TaskScores = dict[str, float]


class JustIdModels:
    """Both classifiers, shared tokenizer settings, one `predict` call."""

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rg_source = _resolve(config.RG_MODEL_DIR, config.RG_HUB_ID)
        bk_source = _resolve(config.BK_MODEL_DIR, config.BK_HUB_ID)
        self.rg_labels = _load_labels(rg_source)
        self.bk_labels = _load_labels(bk_source)
        self.rg_tokenizer = AutoTokenizer.from_pretrained(rg_source)
        self.bk_tokenizer = AutoTokenizer.from_pretrained(bk_source)
        self.rg_model = _load_model(rg_source, len(self.rg_labels), self.device)
        self.bk_model = _load_model(bk_source, len(self.bk_labels), self.device)

    def predict(self, text: str) -> dict[str, TaskScores]:
        """Return kept labels with scores for both tasks."""
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty")
        return {
            "rechtsgebieden": self._score(text, self.rg_tokenizer, self.rg_model, self.rg_labels),
            "bijzondere_kenmerken": self._score(
                text, self.bk_tokenizer, self.bk_model, self.bk_labels
            ),
        }

    def _score(self, text: str, tokenizer, model, labels: list[str]) -> TaskScores:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=config.MAX_LEN,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.sigmoid(logits).squeeze(0).cpu().tolist()
        kept = {
            label: round(float(score), 4)
            for label, score in zip(labels, probs)
            if score >= config.LABEL_THRESHOLD
        }
        return dict(sorted(kept.items(), key=lambda item: item[1], reverse=True))


def _resolve(local: Path, hub_id: str) -> str:
    if (local / "model.safetensors").exists() or (local / "pytorch_model.bin").exists():
        return str(local)
    return hub_id


def _load_labels(source: str) -> list[str]:
    path = Path(source) / "labels.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    downloaded = hf_hub_download(source, "labels.json")
    return json.loads(Path(downloaded).read_text(encoding="utf-8"))


def _load_model(source: str, n_labels: int, device: torch.device):
    model = AutoModelForSequenceClassification.from_pretrained(
        source,
        num_labels=n_labels,
        problem_type="multi_label_classification",
    )
    model.to(device)
    model.eval()
    return model


def main() -> None:
    print("Loading models")
    bundle = JustIdModels()
    print("device:", bundle.device)
    print("Paste a ruling, then a blank line:\n")
    lines: list[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line == "" and lines:
            break
        lines.append(line)
    text = "\n".join(lines).strip()
    out = bundle.predict(text)
    print("\nrechtsgebieden:", out["rechtsgebieden"])
    print("bijzondere_kenmerken:", out["bijzondere_kenmerken"])


if __name__ == "__main__":
    main()
