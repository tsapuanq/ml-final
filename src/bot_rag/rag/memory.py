#src/rag/memory.py
from typing import List, Dict
from bot_rag.config import HISTORY_MAX_TURNS

def push_history(user_data: dict, role: str, text: str):
    hist: List[Dict] = user_data.get("history", [])
    hist.append({"role": role, "text": (text or "")[:1200]})
    if len(hist) > HISTORY_MAX_TURNS * 2:
        hist = hist[-HISTORY_MAX_TURNS * 2:]
    user_data["history"] = hist

def format_history(user_data: dict) -> str:
    hist = user_data.get("history", [])
    lines = []
    for h in hist:
        r = "USER" if h["role"] == "user" else "ASSISTANT"
        lines.append(f"{r}: {h['text']}")
    return "\n".join(lines)
