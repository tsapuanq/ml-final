#src/rag/lang.py
KZ_CHARS = set("әөүұқғңһі")

def detect_lang(text: str) -> str:
    t = (text or "").lower()
    if any(ch in t for ch in KZ_CHARS):
        return "kk"
    if any("а" <= ch <= "я" for ch in t) or "ё" in t:
        return "ru"
    return "en"

def not_found_msg(lang: str) -> str:
    if lang == "kk":
        return "Базада бұл туралы ақпарат жоқ."
    if lang == "ru":
        return "В базе нет информации."
    return "Not found in the knowledge base."