from __future__ import annotations

import os
import json
import logging
from typing import List, Dict, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from .prompts import build_prompt
from .ollama_client import generate

log = logging.getLogger("rag")

OUT_DIR = os.environ.get("OUT_DIR", "/data/knowledge_base")

SNIPPETS_DIR = os.path.join(OUT_DIR, "snippets")
META_PATH = os.path.join(OUT_DIR, "snippets_meta.jsonl")
FAISS_INDEX_PATH = os.path.join(OUT_DIR, "faiss.index")

EMB_MODEL_NAME = os.environ.get(
    "EMB_MODEL_NAME",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)

TOP_K = int(os.environ.get("TOP_K", "5"))
MIN_SCORE = float(os.environ.get("MIN_SCORE", "0.25"))


class RagBot:
    def __init__(self) -> None:
        if not os.path.exists(META_PATH):
            raise RuntimeError(f"Meta file not found: {META_PATH}")
        if not os.path.exists(FAISS_INDEX_PATH):
            raise RuntimeError(f"FAISS index not found: {FAISS_INDEX_PATH}")
        if not os.path.isdir(SNIPPETS_DIR):
            raise RuntimeError(f"Snippets dir not found: {SNIPPETS_DIR}")

        self.embedder = SentenceTransformer(EMB_MODEL_NAME)
        self.index = faiss.read_index(FAISS_INDEX_PATH)
        self.meta = self._load_meta()

    def _load_meta(self) -> List[Dict]:
        out: List[Dict] = []
        with open(META_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                out.append(json.loads(line))
        return out

    def _load_snippet_text(self, sid: str) -> str:
        p = os.path.join(SNIPPETS_DIR, f"{sid}.txt")
        with open(p, "r", encoding="utf-8") as f:
            return f.read().strip()

    def _embed_query(self, q: str) -> np.ndarray:
        v = self.embedder.encode([q]).astype(np.float32)
        faiss.normalize_L2(v)
        return v

    def retrieve(self, question: str) -> List[Tuple[float, Dict, str]]:
        qv = self._embed_query(question)
        D, I = self.index.search(qv, TOP_K)

        results: List[Tuple[float, Dict, str]] = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0:
                continue
            if idx >= len(self.meta):
                continue
            m = self.meta[idx]
            text = self._load_snippet_text(m["id"])
            results.append((float(score), m, text))
        return results

    def answer(self, question: str) -> str:
        retrieved = self.retrieve(question)
        log.info("retrieve: %d candidates for %r", len(retrieved), question[:80])
        for score, m, _ in retrieved:
            log.info("  score=%.3f  title=%r  url=%s", score, m.get("title", ""), m.get("source_url", ""))

        if not retrieved:
            return "Я не знаю."

        good = [r for r in retrieved if r[0] >= MIN_SCORE]
        log.info("retrieve: %d snippets pass MIN_SCORE=%.2f", len(good), MIN_SCORE)
        if not good:
            return "Я не знаю."

        context_blocks = [
            f"[{score:.3f}] {m.get('title','')} ({m.get('source_url','')})\n{text}"
            for score, m, text in good
        ]
        context = "\n\n---\n\n".join(context_blocks)

        prompt = build_prompt(context, question)
        log.info("prompt user len=%d chars", len(prompt["user"]))
        out = generate(prompt["system"], prompt["user"])
        log.info("llm response len=%d chars", len(out) if out else 0)
        return out if out else "Я не знаю."
