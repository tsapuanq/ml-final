#src/rag/query_preprocess.py
import re
import unicodedata
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from difflib import SequenceMatcher

# 1) CANONICAL TERMS (то, как термин встречается в БД / в индексе)
# Формат: (pattern_regex, [canonical_tokens])
TERM_RULES: List[Tuple[str, List[str]]] = [
    # Portal / MySDU
    (r"\boldmy\.sdu\.edu\.kz\b", ["mysdu.edu.kz", "mysdu", "portal"]),
    (r"\bmysdu\b|\bmy\s*sdu\b|\bмойсду\b|\bмайсду\b|\bмйсду\b", ["mysdu", "portal"]),
    (r"\bпортал\b|\bличн(ый|ом)\s*кабинет\b|\bкабинет\b", ["portal", "mysdu"]),

    # Moodle
    (r"\bmoodle\b|\bмудл\b|\bмудле\b|\bмодл\b|\bмудлe\b", ["moodle"]),

    # Retake / Re-exam
    (r"\bretake\b|\bпересдач(а|у|и|е)\b|\bретейк\b|\bперездача\b", ["retake"]),

    # Transcript / SPT / GPA
    (r"\btranscript\b|\bтранскрипт\b|\bвыписк(а|у)\s*оценок\b", ["transcript"]),
    (r"\bspt\b|\bstudent\s*points\b|\bстудент\s*поинтс\b", ["SPT"]),
    (r"\bgpa\b|\bгпа\b", ["GPA"]),

    # FX (если у тебя FX в индексе как FX)
    (r"\bfx\b|\bфх\b", ["FX"]),
]

# 2) FUZZY LEXICON — если пользователь печатает с ошибками
FUZZY_CANON = ["moodle", "retake", "transcript", "mysdu", "portal", "syllabus"]

def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s

def _tokenize(s: str) -> List[str]:
    # простая токенизация: слова/домены/цифры
    return re.findall(r"[a-zA-Z0-9\.\-]+|[а-яА-ЯёЁәөұүқғңһі]+", s)

def _sim(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

@dataclass
class QueryVariants:
    original: str
    augmented: str
    added_tokens: List[str]

def add_canonical_terms(q: str, fuzzy_threshold: float = 0.86) -> QueryVariants:
    """
    Возвращает:
      - original
      - augmented: original + canonical tokens (без дублей)
      - added_tokens: что именно добавили
    """
    q0 = _norm(q)
    q_low = q0.lower()

    added: List[str] = []

    # regex rules
    for pattern, canon in TERM_RULES:
        if re.search(pattern, q_low, flags=re.IGNORECASE):
            added.extend(canon)

    # fuzzy for typos (по токенам)
    toks = _tokenize(q0)
    for t in toks:
        tl = t.lower()
        # пропускаем нормальные длинные русские слова (чтобы не шуметь)
        if len(tl) < 4:
            continue
        for canon in FUZZY_CANON:
            if _sim(tl, canon) >= fuzzy_threshold:
                added.append(canon)

    # дедуп + не добавлять если уже есть
    added_norm = []
    seen = set()
    for x in added:
        xl = x.lower()
        if xl in seen:
            continue
        seen.add(xl)
        if xl not in q_low:
            added_norm.append(x)

    if not added_norm:
        return QueryVariants(original=q0, augmented=q0, added_tokens=[])

    augmented = q0 + " " + " ".join(added_norm)
    return QueryVariants(original=q0, augmented=augmented, added_tokens=added_norm)

def build_query_candidates(user_q: str, expand_fn=None) -> List[str]:
    """
    Делает список кандидатов-запросов:
      1) original
      2) augmented (original + canonical tokens)
      3) expanded(original/augmented) — если передали expand_fn
    """
    v = add_canonical_terms(user_q)
    cands = [v.original]

    if v.augmented != v.original:
        cands.append(v.augmented)

    if expand_fn:
        ex1 = expand_fn(v.original)
        if ex1 and ex1 not in cands:
            cands.append(ex1)

        ex2 = expand_fn(v.augmented)
        if ex2 and ex2 not in cands:
            cands.append(ex2)

    # финальная дедупликация
    out = []
    seen = set()
    for q in cands:
        qn = _norm(q)
        if qn.lower() in seen:
            continue
        seen.add(qn.lower())
        out.append(qn)
    return out