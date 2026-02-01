import os
import logging

from tg.tg_bot import build_app
from rag.bot import RagBot

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("tg-bot")

def main():
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")

    # ---- SELF TEST (КРИТИЧНО) ----
    log.info("Running RAG self-test...")
    RagBot().answer("self-test")
    log.info("RAG self-test OK")

    # ---- TG START ----
    app = build_app(token)
    app.run_polling()

if __name__ == "__main__":
    main()
