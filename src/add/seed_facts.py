# scripts/seed_facts.py
import os
import re
import json
import time
import hashlib
from datetime import datetime, timezone
from typing import List, Dict, Any, Tuple, Iterable

from dotenv import load_dotenv
from openai import OpenAI
from supabase import create_client


# ----------------------------
# helpers
# ----------------------------
def _need(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing env var: {name}")
    return v


def norm_text(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def batched(xs: List[Any], n: int) -> Iterable[List[Any]]:
    for i in range(0, len(xs), n):
        yield xs[i : i + n]


def now_iso() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def rpc_with_retry(call_fn, tries: int = 6, base_sleep: float = 1.0, label: str = "op"):
    last_err = None
    for attempt in range(tries):
        try:
            return call_fn()
        except Exception as e:
            last_err = e
            sleep_s = min(30.0, base_sleep * (2 ** attempt))
            print(f"[WARN] {label} failed attempt={attempt+1}/{tries}: {type(e).__name__}: {e}")
            time.sleep(sleep_s)
    raise last_err


# ----------------------------
# supabase ops
# ----------------------------
def upsert_answer(sb, row: Dict[str, Any]) -> int:
    # Upsert by answer_hash, then fetch answer_id
    rpc_with_retry(
        lambda: sb.table("qa_answers").upsert(row, on_conflict="answer_hash").execute(),
        label="qa_answers.upsert",
    )

    got = rpc_with_retry(
        lambda: (
            sb.table("qa_answers")
            .select("answer_id")
            .eq("answer_hash", row["answer_hash"])
            .limit(1)
            .execute()
        ),
        label="qa_answers.fetch",
    )

    if not got.data:
        raise RuntimeError("Upserted answer but cannot fetch answer_id")
    return int(got.data[0]["answer_id"])


def upsert_index_rows(sb, rows: List[Dict[str, Any]]):
    if not rows:
        return
    rpc_with_retry(
        lambda: sb.table("qa_index").upsert(rows, on_conflict="search_hash").execute(),
        label=f"qa_index.upsert[{len(rows)}]",
    )


# ----------------------------
# main
# ----------------------------
def main():
    load_dotenv()

    OPENAI_API_KEY = _need("OPENAI_API_KEY")
    SUPABASE_URL = _need("SUPABASE_URL")
    SUPABASE_SERVICE_ROLE_KEY = _need("SUPABASE_SERVICE_ROLE_KEY")

    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    # throttle knobs (safe defaults)
    SEED_BATCH = int(os.getenv("SEED_BATCH", "64"))               # embedding batch size
    SEED_UPSERT_BATCH = int(os.getenv("SEED_UPSERT_BATCH", "100"))# upsert batch size
    SLEEP_BETWEEN_EMBED = float(os.getenv("SEED_SLEEP_EMBED", "0.10"))
    SLEEP_BETWEEN_UPSERT = float(os.getenv("SEED_SLEEP_UPSERT", "0.20"))

    client = OpenAI(api_key=OPENAI_API_KEY)
    sb = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

    updated_at = now_iso()

    # ----------------------------
    # FACTS: answers (RU/KZ/EN) + queries per language
    # ----------------------------
    FACTS: List[Dict[str, Any]] = [
        {
            "fact_key": "faculties",
            "answers": {
                "ru": (
                    "В SDU есть 3 школы (факультета):\n"
                    "• SSBL — School of Social Sciences, Business and Law\n"
                    "• SEH — School of Education and Humanities\n"
                    "• SITAM — School of Information Technology and Applied Mathematics"
                ),
                "kk": (
                    "SDU-де 3 мектеп (факультет) бар:\n"
                    "• SSBL — School of Social Sciences, Business and Law\n"
                    "• SEH — School of Education and Humanities\n"
                    "• SITAM — School of Information Technology and Applied Mathematics"
                ),
                "en": (
                    "SDU has 3 schools (faculties):\n"
                    "• SSBL — School of Social Sciences, Business and Law\n"
                    "• SEH — School of Education and Humanities\n"
                    "• SITAM — School of Information Technology and Applied Mathematics"
                ),
            },
            "queries": {
                "ru": [
                    "какие факультеты в sdu",
                    "сколько факультетов в sdu",
                    "какие школы в sdu",
                    "ssbl что это",
                    "seh что это",
                    "sitam что это",
                    "sitam факультет",
                    "seh факультет",
                    "ssbl факультет",
                    "факультеты sdu список",
                ],
                "kk": [
                    "sdu факультеттері қандай",
                    "sdu-де қанша факультет бар",
                    "sdu мектептері қандай",
                    "ssbl деген не",
                    "seh деген не",
                    "sitam деген не",
                    "sitam факультет",
                ],
                "en": [
                    "what faculties are in sdu",
                    "how many faculties in sdu",
                    "sdu schools list",
                    "what is ssbl",
                    "what is seh",
                    "what is sitam",
                    "sitam faculty",
                ],
            },
        },
        {
            "fact_key": "dorm_price",
            "answers": {
                "ru": (
                    "Общежитие SDU:\n"
                    "• 477 000 ₸ за семестр\n"
                    "• 954 000 ₸ за год\n"
                    "• скидка 5% при оплате сразу за год"
                ),
                "kk": (
                    "SDU жатақханасы:\n"
                    "• 477 000 ₸ / семестр\n"
                    "• 954 000 ₸ / жыл\n"
                    "• жылына бірден төлесе 5% жеңілдік"
                ),
                "en": (
                    "SDU dormitory price:\n"
                    "• 477,000 ₸ per semester\n"
                    "• 954,000 ₸ per year\n"
                    "• 5% discount if paid for the whole year at once"
                ),
            },
            "queries": {
                "ru": [
                    "dorm price",
                    "цена общежития sdu",
                    "сколько стоит общага sdu",
                    "жатақхана багасы sdu",
                    "общежитие цена за семестр",
                    "общежитие 477000",
                    "сколько стоит dorm",
                    "оплата общежития за год скидка",
                ],
                "kk": [
                    "жатақхана бағасы",
                    "sdu жатақхана бағасы",
                    "жатақхана 477000",
                    "жатақхана жылдық төлем",
                    "жатақхана жеңілдік 5%",
                ],
                "en": [
                    "sdu dorm price",
                    "how much is dorm in sdu",
                    "dormitory cost per semester",
                    "sdu dormitory discount",
                    "477000 dorm sdu",
                ],
            },
        },
        {
            "fact_key": "elevators",
            "answers": {
                "ru": "В университете 5 лифтов: Library (библиотека), а также блоки D, E, H, I.",
                "kk": "Университетте 5 лифт бар: кітапхана (Library), сондай-ақ D, E, H, I блоктарында.",
                "en": "There are 5 elevators: Library and blocks D, E, H, I.",
            },
            "queries": {
                "ru": [
                    "сколько лифтов в sdu",
                    "лифты в sdu",
                    "где находятся лифты",
                    "лифт библиотека",
                    "лифт блок D E H I",
                ],
                "kk": [
                    "sdu-де қанша лифт бар",
                    "лифт қай блокта",
                    "лифт кітапхана",
                ],
                "en": [
                    "how many elevators in sdu",
                    "where are the elevators in sdu",
                    "elevators library block d e h i",
                ],
            },
        },
        {
            "fact_key": "blocks",
            "answers": {
                "ru": "Блоки (корпуса) университета SDU: A, B, C, D, E, F, G, H, I.",
                "kk": "SDU университетінің блоктары: A, B, C, D, E, F, G, H, I.",
                "en": "SDU university blocks: A, B, C, D, E, F, G, H, I.",
            },
            "queries": {
                "ru": [
                    "какие блоки есть в sdu",
                    "блоки sdu",
                    "корпуса sdu",
                    "список блоков sdu",
                    "A B C D E F G H I блоки",
                ],
                "kk": [
                    "sdu блоктары қандай",
                    "sdu корпустары",
                    "A B C D E F G H I блок",
                ],
                "en": [
                    "what blocks are in sdu",
                    "sdu blocks list",
                    "sdu buildings a b c d e f g h i",
                ],
            },
        },
        {
            "fact_key": "portal_grades",
            "answers": {
                "ru": "Портал для оценок: раньше был oldmy.sdu.edu.kz, сейчас — mysdu.edu.kz.",
                "kk": "Бағалар порталы: бұрын oldmy.sdu.edu.kz болды, қазір — mysdu.edu.kz.",
                "en": "Grades portal: it used to be oldmy.sdu.edu.kz, now it is mysdu.edu.kz.",
            },
            "queries": {
                "ru": [
                    "где смотреть оценки sdu",
                    "портал оценок sdu",
                    "mysdu оценки",
                    "oldmy.sdu.edu.kz",
                    "почему oldmy не работает",
                    "как зайти в mysdu",
                    "где pre-final смотреть",
                ],
                "kk": [
                    "бағаларды қайдан көремін",
                    "sdu бағалар порталы",
                    "mysdu бағалар",
                    "oldmy.sdu.edu.kz",
                ],
                "en": [
                    "where to see grades sdu",
                    "sdu grades portal",
                    "mysdu grades",
                    "oldmy sdu portal",
                    "pre-final grades where",
                ],
            },
        },
        # UPDATED scholarship block (includes pedagogy)
        {
            "fact_key": "scholarship",
            "answers": {
                "ru": (
                    "Стипендия:\n"
                    "• стандартная — 52 372 ₸\n"
                    "• повышенная — 57 000 ₸\n"
                    "• для педагогических направлений — 84 000 ₸"
                ),
                "kk": (
                    "Шәкіртақы:\n"
                    "• стандартты — 52 372 ₸\n"
                    "• жоғарылатылған — 57 000 ₸\n"
                    "• педагогикалық бағыттарға — 84 000 ₸"
                ),
                "en": (
                    "Scholarship:\n"
                    "• standard — 52,372 ₸\n"
                    "• increased — 57,000 ₸\n"
                    "• for pedagogy programs — 84,000 ₸"
                ),
            },
            "queries": {
                "ru": [
                    "какая стипендия в sdu",
                    "сколько стипендия sdu",
                    "стипендия 52372",
                    "стипендия 52000",
                    "повышенная стипендия 57000",
                    "размер стипендии",
                    "стипендия педагогов",
                    "стипендия педагоги 84000",
                    "педагогическая стипендия sdu",
                    "стипендия на педагогическом",
                    "какая стипендия в сду",
                    "сколько стипендия в сду",
                    "сколько стипендия у студентов сду",
                    "стипендия сду",
                    "размер стипендии сду",
                    "стипендия 52372",
                    "стипендия 52 372",
                    "стипендия педагоги 84000",
                    "стипендия у педагогов в сду",
                    "педагогика стипендия 84000",
                    "какая стипендия в SDU",
                    "сколько стипендия в SDU",
                    "стипендия SDU",
                ],
                "kk": [
                    "sdu шәкіртақы қанша",
                    "шәкіртақы 52372",
                    "шәкіртақы 52000",
                    "жоғары шәкіртақы 57000",
                    "педагогика шәкіртақы 84000",
                    "педагогикалық шәкіртақы",
                ],
                "en": [
                    "how much is scholarship in sdu",
                    "sdu scholarship amount",
                    "standard scholarship 52372",
                    "standard scholarship 52000",
                    "increased scholarship 57000",
                    "pedagogy scholarship 84000",
                    "scholarship for pedagogy students",
                ],
            },
        },
        # SDU overview (updated with pedagogy scholarship + exact 52 372)
        {
            "fact_key": "sdu_overview",
            "answers": {
                "ru": (
                    "SDU (Suleyman Demirel University) — университет, расположенный в Алматинской области, г. Каскелен.\n"
                    "SDU часто упоминается среди ведущих университетов Казахстана.\n\n"
                    "Факультеты (школы):\n"
                    "• SSBL — School of Social Sciences, Business and Law\n"
                    "• SEH — School of Education and Humanities\n"
                    "• SITAM — School of Information Technology and Applied Mathematics\n\n"
                    "МИССИЯ:\n"
                    "SDU стремится к развитию через создание и распространение выдающихся знаний и подготовку выпускников — граждан мира, "
                    "чьи ценности формируются посредством целостного образования.\n\n"
                    "ЦЕЛЬ:\n"
                    "Стать международно-ориентированным университетом в Центральной Азии, признанным за преподавание и обучение, инновации, "
                    "исследования и высокий уровень подготовки выпускников.\n\n"
                    "ЦЕННОСТИ:\n"
                    "• Честность\n"
                    "• Уважение\n"
                    "• Открытость\n"
                    "• Ответственность\n"
                    "• Единство"
                ),
                "kk": (
                    "SDU (Suleyman Demirel University) — Алматы облысы, Қаскелең қаласында орналасқан университет.\n"
                    "SDU жиі Қазақстандағы жетекші университеттердің бірі ретінде аталады.\n\n"
                    "Мектептер (факультеттер):\n"
                    "• SSBL — School of Social Sciences, Business and Law\n"
                    "• SEH — School of Education and Humanities\n"
                    "• SITAM — School of Information Technology and Applied Mathematics\n\n"
                    "МИССИЯ:\n"
                    "SDU дамуға білімді жасау және тарату арқылы, сондай-ақ құндылықтары тұтас білім беру арқылы қалыптасатын "
                    "әлем азаматы болатын түлектерді даярлау арқылы ұмтылады.\n\n"
                    "МАҚСАТ:\n"
                    "Орталық Азиядағы халықаралық-бағытталған университетке айналу: оқыту, инновациялар, зерттеулер және түлектердің "
                    "жоғары даярлық деңгейі арқылы мойындалу.\n\n"
                    "ҚҰНДЫЛЫҚТАР:\n"
                    "• Адалдық\n"
                    "• Құрмет\n"
                    "• Ашықтық\n"
                    "• Жауапкершілік\n"
                    "• Бірлік"
                ),
                "en": (
                    "SDU (Suleyman Demirel University) is a university located in Kaskelen, Almaty region.\n"
                    "SDU is often mentioned among leading universities in Kazakhstan.\n\n"
                    "Schools (faculties):\n"
                    "• SSBL — School of Social Sciences, Business and Law\n"
                    "• SEH — School of Education and Humanities\n"
                    "• SITAM — School of Information Technology and Applied Mathematics\n\n"
                    "MISSION:\n"
                    "SDU aims to develop through creating and sharing outstanding knowledge and educating graduates as global citizens, "
                    "whose values are shaped through holistic education.\n\n"
                    "GOAL:\n"
                    "To become an internationally oriented university in Central Asia, recognized for teaching and learning, innovation, "
                    "research, and a high level of graduate preparation.\n\n"
                    "VALUES:\n"
                    "• Integrity\n"
                    "• Respect\n"
                    "• Openness\n"
                    "• Responsibility\n"
                    "• Unity"
                ),
            },
            "queries": {
                "ru": [
                    "расскажи про sdu",
                    "что такое sdu",
                    "информация про sdu",
                    "sdu университет",
                    "где находится sdu",
                    "sdu каскелен",
                    "миссия sdu",
                    "цель sdu",
                    "ценности sdu",
                    "какие факультеты в sdu",
                ],
                "kk": [
                    "sdu туралы айтшы",
                    "sdu деген не",
                    "sdu туралы ақпарат",
                    "sdu қайда орналасқан",
                    "sdu қаскелең",
                    "sdu миссиясы",
                    "sdu мақсаты",
                    "sdu құндылықтары",
                    "sdu факультеттері қандай",
                ],
                "en": [
                    "tell me about sdu",
                    "what is sdu",
                    "sdu university information",
                    "where is sdu located",
                    "sdu kaskelen",
                    "sdu mission",
                    "sdu goal",
                    "sdu values",
                    "what faculties are in sdu",
                ],
            },
        },
    ]

    # 1) upsert answers -> get answer_id per (fact_key, lang)
    lang_answer_ids: Dict[Tuple[str, str], int] = {}
    for f in FACTS:
        fact_key = f["fact_key"]
        for lang, ans in f["answers"].items():
            ans_clean = norm_text(ans)
            a_hash = sha256(f"{lang}::{ans_clean}")
            row = {
                "answer": ans,
                "answer_clean": ans,  # keep same
                "lang": lang,
                "meta": {
                    "type": "fact",
                    "fact_key": fact_key,
                    "source": "manual",
                    "updated_at": updated_at,
                    "confidence": "high",
                },
                "answer_hash": a_hash,
            }
            aid = upsert_answer(sb, row)
            lang_answer_ids[(fact_key, lang)] = aid
            print(f"[OK] qa_answers upsert fact={fact_key} lang={lang} answer_id={aid}")

    # 2) build index rows (answer_id, lang, search_text, embedding)
    triples: List[Tuple[int, str, str]] = []
    for f in FACTS:
        fact_key = f["fact_key"]
        for lang, queries in f["queries"].items():
            aid = lang_answer_ids[(fact_key, lang)]
            for q in queries:
                qn = norm_text(q)
                if qn:
                    triples.append((aid, lang, qn))

    print(f"\nPreparing embeddings for {len(triples)} queries...")

    index_rows: List[Dict[str, Any]] = []
    for batch in batched(triples, SEED_BATCH):
        texts = [t[2] for t in batch]

        emb = rpc_with_retry(
            lambda: client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=texts,
                encoding_format="float",
            ),
            label=f"embeddings.create[{len(texts)}]",
        )
        vectors = [d.embedding for d in emb.data]

        for (aid, lang, qtext), vec in zip(batch, vectors):
            s_hash = sha256(f"{aid}::{lang}::{qtext}")
            index_rows.append(
                {
                    "answer_id": aid,
                    "lang": lang,
                    "search_text": qtext,
                    "embedding": vec,
                    "weight": 1.0,
                    "meta": {
                        "type": "fact_query",
                        "source": "manual",
                        "updated_at": updated_at,
                    },
                    "search_hash": s_hash,
                }
            )

        time.sleep(SLEEP_BETWEEN_EMBED)

    before = len(index_rows)
    dedup = {}
    for r in index_rows:
        dedup[r["search_hash"]] = r   # keeps last one if duplicates exist
    index_rows = list(dedup.values())
    after = len(index_rows)
    print(f"[DEDUP] index_rows: {before} -> {after} (removed {before-after})")
    
    # 3) upsert qa_index in batches
    print(f"\nUpserting {len(index_rows)} rows into qa_index...")
    for b in batched(index_rows, SEED_UPSERT_BATCH):
        upsert_index_rows(sb, b)
        print(f"[OK] qa_index upserted batch size={len(b)}")
        time.sleep(SLEEP_BETWEEN_UPSERT)

    print("\nDONE ✅ Facts seeded successfully.")
    print(f"Total qa_index rows prepared: {len(index_rows)}")


if __name__ == "__main__":
    main()