#src/bot/handlers.py
import re
import time
import logging
from telegram import Update, ReplyKeyboardRemove
from telegram.ext import ContextTypes
from src.rag.query_preprocess import build_query_candidates
from src.config import TOPK_INDEX, SIM_NO_ANSWER, FOLLOWUP_MIN_SCORE, DEBUG
from src.rag.lang import detect_lang, not_found_msg
from src.rag.memory import push_history, format_history
from src.rag.llm import (
    is_followup_llm,
    rewrite_to_standalone,
    generate_answer_from_knowledge,
    verify_answer_supported,
)
from src.bot.ui import (
    menu_kb, feedback_inline_kb,
    MENU_RULES, MENU_EXAMPLES,
    CB_FB_UP, CB_FB_DOWN
)

log = logging.getLogger("tg-rag-bot")


def normalize_q(t: str) -> str:
    t = (t or "").strip()
    t = re.sub(r"^[\-\•\*\u2022]+\s*", "", t)
    t = re.sub(r"\s+", " ", t)
    return t


def debug_log(rag, user_q: str, rewritten_q: str, hits: list[dict], cand: list[tuple], answers_map: dict):
    if not DEBUG:
        return
    log.info("=== DEBUG ===")
    log.info("USER_Q=%s", user_q)
    log.info("REWRITTEN_Q=%s", rewritten_q)
    for i, h in enumerate(hits[:5], start=1):
        log.info("HIT#%d score=%.3f answer_id=%s", i, float(h.get("score", 0.0)), str(h.get("answer_id")))
    for i, (aid, sc) in enumerate(cand[:6], start=1):
        row = answers_map.get(aid) or {}
        txt = (row.get("answer_clean") or row.get("answer") or "")[:120].replace("\n", " ")
        log.info("CAND#%d id=%s score=%.3f preview=%s", i, str(aid), float(sc), txt)


def rules_text() -> str:
    return (
        "📌 SDU AI Assistant • Rules / Ережелер / Правила\n\n"
        "🇬🇧 EN:\n"
        "• I’m an SDU AI assistant — I answer using the SDU knowledge base.\n"
        "• If the info isn’t in the base, I’ll tell you honestly.\n"
        "• Best: one clear question per message.\n"
        "• Please write the full question and include key words.\n"
        "• You can find sample questions in the “Examples” section.\n"
        "• Tip: add details (faculty/program/year) if needed.\n\n"
        "🇰🇿 KZ:\n"
        "• Мен SDU бойынша AI көмекшімін — жауапты тек білім базасына сүйеніп беремін.\n"
        "• Базада ақпарат болмаса — оны ашық айтамын.\n"
        "• Ең дұрысы: бір хабарламада бір нақты сұрақ.\n"
        "• Сұрақты толық жазыңыз және негізгі кілт сөздерді қосыңыз.\n"
        "• Мысал сұрақтарды “Examples” бөлімінен көре аласыз.\n"
        "• Кеңес: қажет болса факультет/бағдарлама/курсты көрсетіңіз.\n\n"
        "🇷🇺 RU:\n"
        "• Я AI-ассистент SDU — отвечаю только по базе знаний SDU.\n"
        "• Если информации в базе нет — скажу об этом напрямую.\n"
        "• Лучше всего: один понятный вопрос = одно сообщение.\n"
        "• Пишите вопрос полностью и добавляйте ключевые слова.\n"
        "• Примеры вопросов можно посмотреть в разделе “Examples”.\n"
        "• Совет: при необходимости добавьте детали (факультет/программа/курс).\n"
    )


def examples_text() -> str:
    return (
        "🧪 Examples / Мысалдар / Примеры\n\n"
        "• Что такое Syllabus в СДУ?\n"
        "• 'Welcome party' деген не??\n"
        "• When can I learn retakes?\n"
        "• Где посмотреть pre-final\n"
        "• How to use Moodle?\n"
        "• Kак получить справку с места учебы?\n"
        "• 1 кредиттің бағасы қанша?\n"
    )


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🇬🇧 EN: Hello! I’m the SDU AI Assistant. I answer questions using the SDU knowledge base.\n"
        "🇰🇿 KZ: Сәлем! Мен SDU AI көмекшісімін. Жауапты SDU білім базасына сүйеніп беремін.\n"
        "🇷🇺 RU: Привет! Я SDU AI-ассистент. Отвечаю по базе знаний SDU.\n\n"
        "Topics: dormitory, FX/exchange, grades & retakes, documents, portal/Moodle, student life.\n"
        "Please send one clear question per message.",
        reply_markup=menu_kb(),
    )
async def clear_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # полностью очищаем user_data (история, last payload, feedback flags)
    context.user_data.clear()
    await update.message.reply_text(
        "✅ Chat history cleared.\n"
        "✅ История очищена.\n"
        "✅ Чат тарихы тазаланды.",
        reply_markup=menu_kb(),
    )

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Send your question. Use the buttons below 👇",
        reply_markup=menu_kb(),
    )


async def on_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles inline feedback buttons under bot answers."""
    q = update.callback_query
    if not q or not q.data:
        return

    data = q.data
    last = context.user_data.get("last_answer_payload") or {}

    # Always answer callback to stop Telegram loading spinner
    try:
        await q.answer()
    except Exception:
        pass

    # Remove inline buttons after click (avoid multiple votes)
    try:
        await q.edit_message_reply_markup(reply_markup=None)
    except Exception:
        pass

    if data == CB_FB_UP:
        log.info("POS_FEEDBACK last=%s", last)
        await q.message.reply_text("Thanks! ✅", reply_markup=menu_kb())
        return

    if data == CB_FB_DOWN:
        log.info("NEG_FEEDBACK last=%s", last)
        context.user_data["awaiting_feedback_text"] = True
        await q.message.reply_text(
            "Got it. What was wrong? (one short sentence)",
            reply_markup=ReplyKeyboardRemove(),
        )
        return


async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE, rag):
    user_text = normalize_q(update.message.text or "")
    if not user_text:
        return

    # ---- feedback comment mode (not part of history) ----
    if context.user_data.get("awaiting_feedback_text"):
        context.user_data["awaiting_feedback_text"] = False
        last = context.user_data.get("last_answer_payload") or {}
        log.info("NEG_FEEDBACK_COMMENT last=%s comment=%s", last, user_text)
        await update.message.reply_text("Saved ✅ Thanks!", reply_markup=menu_kb())
        return

    # ---- menu buttons (not part of history) ----
    if user_text == MENU_RULES:
        await update.message.reply_text(rules_text(), reply_markup=menu_kb())
        return

    if user_text == MENU_EXAMPLES:
        await update.message.reply_text(examples_text(), reply_markup=menu_kb())
        return

    # ---- normal question ----
    lang = detect_lang(user_text)
    nf = not_found_msg(lang)

    try:
        hist_txt = format_history(context.user_data)

        rewritten = user_text
        query_candidates = build_query_candidates(rewritten)
        best_hits = []
        best_q = rewritten
        best_score = -1.0

        emb_cache = {}  # ✅ embeddings per candidate text

        for qc in query_candidates:
            if qc not in emb_cache:
                emb_cache[qc] = rag.embed(qc)  # ✅ embedding соответствует qc

            hits_try = rag.search_hybrid_with_embedding(qc, emb_cache[qc], top_k=TOPK_INDEX)
            sc = float(hits_try[0].get("score", 0.0)) if hits_try else 0.0

            if sc > best_score:
                best_score = sc
                best_hits = hits_try
                best_q = qc

        hits = best_hits
        rewritten = best_q
        top0 = best_score

        # 2) follow-up detection & rewrite
        followup_like = any(p in user_text.lower() for p in [
            "они", "это", "там", "про них", "а что", "подробнее", "нет про",
            "what about", "tell me more", "about it"
        ])
        if hist_txt.strip() and (top0 < FOLLOWUP_MIN_SCORE or len(user_text.split()) <= 8 or followup_like):
            try:
                if is_followup_llm(rag.client, user_text, hist_txt):
                    rewritten = rewrite_to_standalone(rag.client, user_text, hist_txt)

                    query_candidates = build_query_candidates(rewritten)

                    best_hits = []
                    best_q = rewritten
                    best_score = -1.0

                    for qc in query_candidates:
                        hits_try = rag.search_hybrid(qc, top_k=TOPK_INDEX)
                        sc = float(hits_try[0].get("score", 0.0)) if hits_try else 0.0
                        if sc > best_score:
                            best_score = sc
                            best_hits = hits_try
                            best_q = qc

                    hits = best_hits
                    rewritten = best_q
                    top0 = best_score
            except Exception:
                log.exception("Follow-up rewrite failed")


        if not hits:
            await update.message.reply_text(nf, reply_markup=menu_kb())
            return

        top1 = float(hits[0].get("score", 0.0))
        log.info("Q='%s' top1(score)=%.3f rewritten='%s'", user_text, top1, rewritten)

        if top1 < SIM_NO_ANSWER:
            await update.message.reply_text(nf, reply_markup=menu_kb())
            return

        # 4) candidates -> answers_map
        cand = rag.pick_candidates(hits, max_unique=6)
        answer_ids = [aid for aid, _ in cand]
        answers_map = rag.fetch_answers(answer_ids)

        # 5) optional rerank
        chosen_id = rag.rerank_if_needed(user_text, cand, answers_map)
        chosen = answers_map.get(chosen_id)
        if not chosen:
            await update.message.reply_text(nf, reply_markup=menu_kb())
            return

        # 6) KNOWLEDGE blocks
        def get_ans(aid: int) -> str:
            row = answers_map.get(aid) or {}
            return (row.get("answer_clean") or row.get("answer") or "").strip()

        knowledge_blocks = []
        seen = set()

        kb0 = get_ans(chosen_id)
        if kb0:
            knowledge_blocks.append(kb0)
            seen.add(chosen_id)

        for aid, _ in cand:
            if aid in seen:
                continue
            kb = get_ans(aid)
            if kb:
                knowledge_blocks.append(kb)
                seen.add(aid)

        debug_log(rag, user_text, rewritten, hits, cand, answers_map)

        # 7) generate answer strictly from knowledge (+ history only for references)
        final = generate_answer_from_knowledge(
            client=rag.client,
            user_question=user_text,
            hist_txt=hist_txt,
            knowledge_blocks=knowledge_blocks,
            lang=lang,
        )

        supported = verify_answer_supported(rag.client, final, knowledge_blocks, lang)
        if not supported:
            log.warning("Verifier UNSUPPORTED. Falling back to raw KB. q=%s", user_text)
            final = kb0 if kb0 else nf

        # 8) save payload for feedback
        context.user_data["last_answer_payload"] = {
            "question": user_text,
            "rewritten": rewritten,
            "answer_id": chosen_id,
            "score": top1,
            "lang": lang,
            "ts": time.time(),
        }

        # ✅ IMPORTANT: inline feedback buttons UNDER the answer
        await update.message.reply_text(final, reply_markup=feedback_inline_kb())

        # 9) store real history
        push_history(context.user_data, "user", user_text)
        push_history(context.user_data, "assistant", final)

    except Exception:
        log.exception("Error handling message")
        await update.message.reply_text("Error. Please try again 🙏", reply_markup=menu_kb())