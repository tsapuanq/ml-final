#src/bot/ui.py
from telegram import (
    ReplyKeyboardMarkup, KeyboardButton,
    InlineKeyboardMarkup, InlineKeyboardButton
)

# menu (always at bottom)
MENU_RULES = "📌 Rules"
MENU_EXAMPLES = "🧪 Examples"

# callback data for inline feedback
CB_FB_UP = "fb:up"
CB_FB_DOWN = "fb:down"


def menu_kb():
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(MENU_RULES), KeyboardButton(MENU_EXAMPLES)],
        ],
        resize_keyboard=True,
        one_time_keyboard=False,
        input_field_placeholder="Сұрағыңды жаз…/ Type your question… / Напиши свой вопрос…",
    )


def feedback_inline_kb():
    # Inline buttons under the bot answer (not at the bottom)
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("👍 Helpful", callback_data=CB_FB_UP),
            InlineKeyboardButton("👎 Not helpful", callback_data=CB_FB_DOWN),
        ]
    ])