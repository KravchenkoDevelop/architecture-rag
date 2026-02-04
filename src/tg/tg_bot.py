
import os
import time
from rag.bot import RagBot

OUT_DIR = os.environ.get("OUT_DIR", "/data/knowledge_base")
META_PATH = os.path.join(OUT_DIR, "snippets_meta.jsonl")
FAISS_INDEX_PATH = os.path.join(OUT_DIR, "faiss.index")

_rag_instance: RagBot | None = None

def get_rag() -> RagBot:
    global _rag_instance
    if _rag_instance is not None:
        return _rag_instance

    # ждать базу (жёстко, без “магии compose”)
    deadline = time.time() + int(os.environ.get("KB_WAIT_SEC", "300"))
    while time.time() < deadline:
        if os.path.exists(META_PATH) and os.path.exists(FAISS_INDEX_PATH):
            _rag_instance = RagBot()
            return _rag_instance
        time.sleep(2)

    raise RuntimeError(
        f"Knowledge base not ready. Missing: "
        f"{'meta ' if not os.path.exists(META_PATH) else ''}"
        f"{'faiss ' if not os.path.exists(FAISS_INDEX_PATH) else ''}"
        f"paths: meta={META_PATH} index={FAISS_INDEX_PATH}"
    )
