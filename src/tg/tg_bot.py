
from __future__ import annotations
import uuid

import logging
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

from rag.bot import RagBot

# ---------------------------------------------------
# LOGGING
# ---------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("tg-bot")

# ---------------------------------------------------
# RAG CORE
# ---------------------------------------------------
rag = RagBot()

HELP_TEXT = """Я RAG-бот по строительным нормам.

Я отвечаю строго на основе внутренней базы знаний.
Если данных нет — отвечу: "Я не знаю".

Примеры вопросов:
- Какие требования к ограждениям лестниц?
- Когда допускается отклонение от нормы?
- Какие условия применения указаны для пункта?
"""

# ---------------------------------------------------
# HANDLERS
# ---------------------------------------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(HELP_TEXT)


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(HELP_TEXT)



async def handle_question(update: Update, context: ContextTypes.DEFAULT_TYPE):
    question = (update.message.text or "").strip()
    if not question:
        return

    await update.message.chat.send_action("typing")

    try:
        answer = rag.answer(question)
    except Exception:
        err_id = uuid.uuid4().hex[:8]
        log.exception("RAG failed | err_id=%s | q=%r", err_id, question)
        answer = f"Внутренняя ошибка обработки. Код: {err_id}"

    if len(answer) > 3500:
        answer = answer[:3500] + "\n\n[ответ сокращён]"

    await update.message.reply_text(answer)



# ---------------------------------------------------
# APP BUILDER
# ---------------------------------------------------
def build_app(token: str) -> Application:
    app = Application.builder().token(token).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_question))

    return app
