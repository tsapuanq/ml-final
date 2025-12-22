#src/bot_rag/config.py
import os
from dotenv import load_dotenv

load_dotenv()

def _need(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing required env var: {name}. Check your .env or shell env.")
    return v

OPENAI_API_KEY = _need("OPENAI_API_KEY")
SUPABASE_URL = _need("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = _need("SUPABASE_SERVICE_ROLE_KEY")
TELEGRAM_BOT_TOKEN = _need("TELEGRAM_BOT_TOKEN")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
REWRITE_MODEL = os.getenv("REWRITE_MODEL", "gpt-4o-mini")
ANSWER_MODEL = os.getenv("ANSWER_MODEL", "gpt-4o-mini")
VERIFIER_MODEL = os.getenv("VERIFIER_MODEL", "gpt-4o-mini")
RERANK_MODEL = os.getenv("RERANK_MODEL", "gpt-4o-mini")

TOPK_INDEX = int(os.getenv("TOPK_INDEX", "20"))
SIM_NO_ANSWER = float(os.getenv("SIM_NO_ANSWER", "0.38"))
FOLLOWUP_MIN_SCORE = float(os.getenv("FOLLOWUP_MIN_SCORE", "0.55"))

HISTORY_MAX_TURNS = int(os.getenv("HISTORY_MAX_TURNS", "8"))

DEBUG = os.getenv("DEBUG", "0") == "1"