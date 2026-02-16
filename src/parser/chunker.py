from __future__ import annotations
from typing import List
import re

def norm_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def chunk_text(text: str, chunk_chars: int = 1600, overlap: int = 250) -> List[str]:
    text = norm_text(text)
    if len(text) <= chunk_chars:
        return [text]

    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_chars)
        window = text[start:end]

        cut = window.rfind("\n\n")
        if cut > int(chunk_chars * 0.6):
            window = window[:cut].strip()
            end = start + cut

        window = window.strip()
        if window:
            chunks.append(window)

        if end >= len(text):
            break
        start = max(0, end - overlap)

    return chunks
