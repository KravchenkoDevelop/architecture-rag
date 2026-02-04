from __future__ import annotations

import os
import json
from typing import List, Dict, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from .prompts import build_prompt
from .ollama_client import generate


# ------------------------------------------------------------------
# PATHS
# ------------------------------------------------------------------
OUT_DIR = os.environ.get("OUT_DIR", "/data/knowledge_base")

SNIPPETS_DIR = os.path.join(OUT_DIR, "snippets")
META_PATH = os.path.join(OUT_DIR, "snippets_meta.jsonl")
FAISS_INDEX_PATH = os.path.join(OUT_DIR, "faiss.index")

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
EMB_MODEL_NAME = os.environ.get(
    "EMB_MODEL_NAME",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)

TOP_K = int(os.environ.get("TOP_K", "5"))
MIN_SCORE = float(os.environ.get("MIN_SCORE", "0.25"))


class RagBot:
    def __init__(self) -> None:
        if not os.path.exists(FAISS_INDEX_PATH):
            raise RuntimeError(
                f"FAISS index not found: {FAISS_INDEX_PATH}"
            )

        if not os.path.exists(META_PATH):
            raise RuntimeError(
                f"Meta file not found: {META_PATH}"
            )

        self.embedder = SentenceTransformer(EMB_MODEL_NAME)
        self.index = faiss.read_index(FAISS_INDEX_PATH)
        self.meta = self._load_meta()


    # ------------------------------
    # LOADERS
    # ------------------------------
    def _load_meta(self) -> List[Dict]:
        out = []
        with open(META_PATH, "r", encoding="utf-8") as f:
            for line in f:
                out.append(json.loads(line))
        return out

    def _load_snippet_text(self, sid: str) -> str:
        with open(os.path.join(SNIPPETS_DIR, f"{sid}.txt"), "r", encoding="utf-8") as f:
            return f.read().strip()

    # ------------------------------
    # RETRIEVAL
    # ------------------------------
    def _embed_query(self, q: str) -> np.ndarray:
        v = self.embedder.encode([q]).astype(np.float32)
        faiss.normalize_L2(v)
        return v

    def retrieve(self, question: str) -> List[Tuple[float, Dict, str]]:
        qv = self._embed_query(question)
        D, I = self.index.search(qv, TOP_K)

        results = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0:
                continue
            m = self.meta[idx]
            text = self._load_snippet_text(m["id"])
            results.append((float(score), m, text))

        return results

    # ------------------------------
    # ANSWER
    # ------------------------------
    def answer(self, question: str) -> str:
        retrieved = self.retrieve(question)

        if not retrieved:
            return "Я не знаю."

        good = [r for r in retrieved if r[0] >= MIN_SCORE]
        if not good:
            return "Я не знаю."

        context_blocks = []
        for score, m, text in good:
            context_blocks.append(
                f"[{score:.3f}] {m.get('title','')} ({m.get('source_url','')})\n{text}"
            )

        context = "\n\n---\n\n".join(context_blocks)

        prompt = build_prompt(context, question)
        out = generate(prompt["system"], prompt["user"])

        return out if out else "Я не знаю."
