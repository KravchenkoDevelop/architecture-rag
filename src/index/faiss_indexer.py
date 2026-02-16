from __future__ import annotations

import os
import json
from typing import List, Dict, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

OUT_DIR = os.environ.get("OUT_DIR", "/data/knowledge_base")
SNIPPETS_DIR = os.path.join(OUT_DIR, "snippets")
META_PATH = os.path.join(OUT_DIR, "snippets_meta.jsonl")
FAISS_INDEX_PATH = os.path.join(OUT_DIR, "faiss.index")

EMB_MODEL_NAME = os.environ.get(
    "EMB_MODEL_NAME",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)

BATCH_SIZE = int(os.environ.get("EMB_BATCH_SIZE", "32"))


def load_meta_in_order() -> List[Dict]:
    meta: List[Dict] = []
    with open(META_PATH, "r", encoding="utf-8") as f:
        for line in f:
            meta.append(json.loads(line))
    return meta


def load_texts(meta: List[Dict]) -> List[str]:
    texts: List[str] = []
    for m in meta:
        sid = m["id"]
        path = os.path.join(SNIPPETS_DIR, f"{sid}.txt")
        with open(path, "r", encoding="utf-8") as f:
            texts.append(f.read().strip())
    return texts


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine similarity
    index.add(embeddings.astype(np.float32))
    return index


def run_index() -> int:
    if not os.path.exists(META_PATH):
        raise RuntimeError("Нет snippets_meta.jsonl — сначала должен отработать crawler.")

    meta = load_meta_in_order()
    texts = load_texts(meta)

    model = SentenceTransformer(EMB_MODEL_NAME)
    embs = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=False,
    )
    embs = np.asarray(embs, dtype=np.float32)

    index = build_faiss_index(embs)
    faiss.write_index(index, FAISS_INDEX_PATH)

    print(f"[OK] faiss index built: vectors={len(texts)} path={FAISS_INDEX_PATH}")
    return len(texts)


if __name__ == "__main__":
    run_index()
