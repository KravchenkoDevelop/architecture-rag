from __future__ import annotations

import os
import time
import logging

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters

from rag.bot import RagBot
from rag.ollama_client import OllamaError


log = logging.getLogger("tg-bot")

OUT_DIR = os.environ.get("OUT_DIR", "/data/knowledge_base")
META_PATH = os.path.join(OUT_DIR, "snippets_meta.jsonl")
FAISS_INDEX_PATH = os.path.join(OUT_DIR, "faiss.index")

KB_WAIT_SEC = int(os.environ.get("KB_WAIT_SEC", "300"))

_rag: RagBot | None = None


def _wait_kb() -> None:
    deadline = time.time() + KB_WAIT_SEC
    while time.time() < deadline:
        if os.path.exists(META_PATH) and os.path.exists(FAISS_INDEX_PATH):
            return
        time.sleep(2)
    raise RuntimeError(f"Knowledge base not ready. meta={META_PATH} index={FAISS_INDEX_PATH}")


def get_rag() -> RagBot:
    global _rag
    if _rag is not None:
        return _rag
    _wait_kb()
    _rag = RagBot()
    return _rag


async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("OK. Send a question.")


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    q = (update.message.text or "").strip()
    if not q:
        return

    try:
        rag = get_rag()
        a = rag.answer(q)

    except OllamaError as e:
        log.exception("LLM failed")
        a = f"Ошибка. LLM (Ollama) недоступна.\n{e}"

    except RuntimeError as e:
        log.exception("KB not ready or runtime error")
        a = f"Ошибка. База знаний не готова.\n{e}"

    except Exception as e:
        log.exception("RAG failed")
        a = f"Ошибка. RAG/LLM недоступны.\n{type(e).__name__}: {e}"


    await update.message.reply_text(a)


def build_app(token: str | None = None) -> Application:
    # совместимость: можно передать token аргументом, можно через env
    token = token or os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")

    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    return app
