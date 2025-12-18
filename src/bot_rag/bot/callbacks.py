# src/bot/callbacks.py
from telegram import Update
from telegram.ext import ContextTypes

from bot_rag.bot.ui import CB_RULES, CB_EXAMPLES, rules_text, examples_text, start_inline_kb


async def on_ui_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    if not q:
        return

    await q.answer()

    if q.data == CB_RULES:
        await q.message.reply_text(rules_text(), reply_markup=start_inline_kb(), parse_mode="Markdown")
        return

    if q.data == CB_EXAMPLES:
        await q.message.reply_text(examples_text(), reply_markup=start_inline_kb(), parse_mode="Markdown")
        return
