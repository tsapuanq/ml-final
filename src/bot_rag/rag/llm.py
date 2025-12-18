#src/rag/llm.py

from openai import OpenAI
from bot_rag.config import REWRITE_MODEL, ANSWER_MODEL, VERIFIER_MODEL
from bot_rag.rag.lang import not_found_msg

def is_followup_llm(client: OpenAI, question: str, hist_txt: str) -> bool:
    if not hist_txt.strip():
        return False

    r = client.responses.create(
        model=REWRITE_MODEL,
        instructions=(
            "You are a classifier.\n"
            "Decide if the user's last question depends on the previous conversation context.\n"
            "Return ONLY 'YES' or 'NO'."
        ),
        input=f"CONVERSATION:\n{hist_txt}\n\nLAST USER QUESTION:\n{question}\n",
    )
    ans = (r.output_text or "").strip().upper()
    return ans.startswith("YES")

def rewrite_to_standalone(client: OpenAI, question: str, hist_txt: str) -> str:
    if not hist_txt.strip():
        return question

    r = client.responses.create(
        model=REWRITE_MODEL,
        instructions=(
            "You are a query rewriter for a retrieval system.\n"
            "Rewrite the user's last question into a standalone question using the conversation context.\n"
            "Rules:\n"
            "- Do NOT answer the question.\n"
            "- Do NOT add any new facts.\n"
            "- Keep the same language as the user's last message.\n"
            "- Output ONLY the rewritten standalone question text.\n"
        ),
        input=f"CONVERSATION:\n{hist_txt}\n\nLAST USER QUESTION:\n{question}\n",
    )
    out = (r.output_text or "").strip()
    return out if out else question

def generate_answer_from_knowledge(client: OpenAI, user_question: str, hist_txt: str, knowledge_blocks: list[str], lang: str) -> str:
    knowledge = "\n\n---\n\n".join([kb for kb in knowledge_blocks if kb.strip()])
    knowledge = knowledge[:7000]

    nf = not_found_msg(lang)
    if lang == "kk":
        style = "Жауапты қазақ тілінде жаз. 2–6 сөйлем, керек болса маркер қолдан.\n"
    elif lang == "ru":
        style = "Пиши по-русски. 2–6 предложений, можно списком.\n"
    else:
        style = "Write in English. 2–6 sentences, bullets allowed.\n"

    instructions = (
        "You are a factual assistant for a closed-book Q&A bot.\n"
        "You MUST answer using ONLY the text inside KNOWLEDGE.\n"
        "Chat history is provided only to resolve references (e.g., 'it', 'there'), but is NOT a knowledge source.\n"
        "Do NOT add any new facts.\n"
        "If KNOWLEDGE does not contain the answer, reply exactly with:\n"
        f"{nf}\n"
        f"{style}"
    )

    prompt = (
        f"CHAT HISTORY (context only):\n{hist_txt}\n\n"
        f"USER QUESTION:\n{user_question}\n\n"
        f"KNOWLEDGE:\n{knowledge}\n"
    )

    r = client.responses.create(
        model=ANSWER_MODEL,
        instructions=instructions,
        input=prompt,
    )
    return (r.output_text or "").strip()

def verify_answer_supported(client: OpenAI, answer: str, knowledge_blocks: list[str], lang: str) -> bool:
    """
    Anti-hallucination: if ANY claim isn't supported by KNOWLEDGE -> UNSUPPORTED.
    """
    nf = not_found_msg(lang)
    if not answer or answer.strip() == nf:
        return True

    knowledge = "\n\n---\n\n".join([kb for kb in knowledge_blocks if kb.strip()])
    knowledge = knowledge[:7000]
    answer = answer[:2500]

    r = client.responses.create(
        model=VERIFIER_MODEL,
        instructions=(
            "You are a strict verifier for a RAG assistant.\n"
            "Check whether EVERY factual claim in ANSWER is directly supported by the text in KNOWLEDGE.\n"
            "If even one claim is not supported, return UNSUPPORTED.\n"
            "Return ONLY one word: SUPPORTED or UNSUPPORTED."
        ),
        input=f"KNOWLEDGE:\n{knowledge}\n\nANSWER:\n{answer}\n",
    )
    verdict = (r.output_text or "").strip().upper()
    return verdict.startswith("SUPPORTED")
