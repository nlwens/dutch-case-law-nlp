"""Split the RAG corpus into overlapping chunks, tagged by section.

A heading is a short line that matches one of SECTION_PATTERNS. Everything
after it, up to the next heading, belongs to that section. Chunks never
cross a section boundary, so each chunk has exactly one section label.

Run from the project root:
    .venv\\Scripts\\python.exe -m rag.chunk
"""

from __future__ import annotations

import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

TARGET_WORDS = 250
OVERLAP_WORDS = 40
MAX_HEADING_CHARS = 70
MIN_CHUNK_WORDS = 20

# Checked in this order: "Beoordeling van het geschil" is the court's
# assessment, not the parties' dispute. "Conclusie inzake" opens the
# Advocaat-Generaal's opinion, which is advice to the Hoge Raad, not a ruling.
SECTION_PATTERNS = [
    ("advisory", [r"^conclusie inzake", r"advocaat-generaal"]),
    ("decision", [r"^(de )?beslissing", r"^recht doende", r"^dictum", r"^beslist"]),
    ("court", [r"beoordeling", r"^(het |de )?oordeel", r"overweging", r"bewezenverklaring",
               r"^(het )?bewijs", r"motivering", r"strafbaarheid", r"oplegging",
               r"^conclusie", r"^slotsom", r"waardering"]),
    ("parties", [r"standpunt", r"^(het )?geschil", r"^(de )?vordering", r"\bverweer\b",
                 r"grief", r"^(de )?stellingen", r"beroepsgronden"]),
    ("facts", [r"^(de )?(vaststaande )?feiten", r"^uitgangspunten"]),
    ("procedure", [r"procesverloop", r"procedure", r"^(het )?geding", r"^inleiding",
                   r"onderzoek (van de zaak|ter terechtzitting)", r"tenlastelegging",
                   r"ontstaan en loop"]),
]
_COMPILED = [(name, [re.compile(p) for p in pats]) for name, pats in SECTION_PATTERNS]
_NUMBER_ONLY = re.compile(r"^\d+(\.\d+)*\.?$")
_LEADING_NUMBER = re.compile(r"^(\d+(\.\d+)*\.?|[IVX]+\.)\s*")


def heading_section(line: str) -> str | None:
    text = _LEADING_NUMBER.sub("", line).strip(" :").lower()
    if not text or len(text) > MAX_HEADING_CHARS or text.endswith((".", ",", ";")):
        return None
    for name, patterns in _COMPILED:
        if any(p.search(text) for p in patterns):
            return name
    return None


def paragraphs(text: str) -> list[str]:
    """Non-empty lines, with a bare paragraph number glued to the next line."""
    out: list[str] = []
    pending = ""
    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            continue
        if _NUMBER_ONLY.match(line):
            pending = line + " "
            continue
        out.append(pending + line)
        pending = ""
    return out


def sections(text: str) -> list[tuple[str, str, list[str]]]:
    """(section, heading, paragraphs) blocks in document order."""
    blocks = [("header", "", [])]
    for para in paragraphs(text):
        name = heading_section(para)
        if name:
            blocks.append((name, para, []))
        else:
            blocks[-1][2].append(para)
    return [b for b in blocks if b[2]]


def window(paras: list[str]) -> list[str]:
    """Pack paragraphs into ~TARGET_WORDS chunks with OVERLAP_WORDS carried over."""
    words_per_para = []
    for para in paras:
        words = para.split()
        for i in range(0, len(words), TARGET_WORDS):
            words_per_para.append(words[i : i + TARGET_WORDS])

    chunks: list[str] = []
    current: list[str] = []
    for words in words_per_para:
        if current and len(current) + len(words) > TARGET_WORDS:
            chunks.append(" ".join(current))
            current = current[-OVERLAP_WORDS:]
        current.extend(words)
    if current and (not chunks or len(current) > OVERLAP_WORDS):
        chunks.append(" ".join(current))
    return chunks


def main() -> None:
    n_rulings = 0
    no_court = 0
    chunks_out = []
    with open(config.RAG_CORPUS, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            n_rulings += 1
            blocks = sections(row["text"])
            if not any(name in ("court", "decision") for name, _, _ in blocks):
                no_court += 1
            index = 0
            for name, heading, paras in blocks:
                for text in window(paras):
                    if len(text.split()) < MIN_CHUNK_WORDS:
                        continue
                    index += 1
                    chunks_out.append({
                        "chunk_id": f"{row['ecli']}#{index}",
                        "ecli": row["ecli"],
                        "court": row["court"],
                        "date": row["date"],
                        "rechtsgebieden": row["rechtsgebieden"],
                        "topics": row["topics"],
                        "section": name,
                        "heading": heading,
                        "text": text,
                    })

    with open(config.RAG_CHUNKS, "w", encoding="utf-8") as f:
        for chunk in chunks_out:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    lengths = [len(c["text"].split()) for c in chunks_out]
    print(f"{len(chunks_out)} chunks from {n_rulings} rulings -> {config.RAG_CHUNKS}")
    print(f"  words per chunk: median {statistics.median(lengths):.0f}, "
          f"min {min(lengths)}, max {max(lengths)}")
    print(f"  rulings without a court/decision section: {no_court}")
    for name, count in Counter(c["section"] for c in chunks_out).most_common():
        print(f"  {name:10s} {count:6d}  ({count / len(chunks_out):.0%})")


if __name__ == "__main__":
    main()
