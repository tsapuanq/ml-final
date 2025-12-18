#src/bot/app.py
import logging
from functools import partial

from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, CallbackQueryHandler, filters
from bot_rag.config import TELEGRAM_BOT_TOKEN, OPENAI_API_KEY, SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY
from bot_rag.rag.rag2 import RAG2
from bot_rag.bot.handlers import start, help_cmd, on_text, on_callback, clear_cmd

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("tg-rag-bot")


def main():
    rag = RAG2(
        openai_api_key=OPENAI_API_KEY,
        supabase_url=SUPABASE_URL,
        supabase_service_key=SUPABASE_SERVICE_ROLE_KEY,
    )

    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("clear", clear_cmd))
    app.add_handler(CallbackQueryHandler(on_callback))

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, partial(on_text, rag=rag)))

    app.run_polling()


if __name__ == "__main__":
    main()
