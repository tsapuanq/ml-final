# pipelines/ingestion/ingest_sdu_programs_json.py
import os
import json
import time
import hashlib
from typing import Dict, List, Any, Tuple

from dotenv import load_dotenv
from tqdm import tqdm
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_ROLE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
sb = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

# ВАЖНО: модель эмбеддинга должна совпадать с той, что использует твой RAG при поиске
# (иначе размерность будет отличаться и/или качество будет хуже).
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")

# OpenAI ключ обязателен (если нет — скрипт упадёт с понятной ошибкой)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

TABLE_QA_CHUNKS = "qa_chunks"
BATCH_SIZE = 128          # эмбеддинги лучше считать небольшими батчами
INSERT_BATCH_SIZE = 200   # в Supabase можно вставлять крупнее


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def norm(s: str) -> str:
    return " ".join((s or "").strip().split())


def get_lang_value(d: Dict[str, Any], lang: str) -> str:
    if not isinstance(d, dict):
        return norm(str(d))
    return norm(d.get(lang) or "")


def build_answer_prefix(program_name: str, faculty_name: str, program_id: str, faculty_code: str, lang: str) -> str:
    return f"[SDU|{lang}|{faculty_code}|{program_id}] {program_name} — {faculty_name}. "


def q_templates(lang: str) -> Dict[str, List[str]]:
    # Чуть расширил: добавил короткие варианты “как обычно спрашивают”.
    if lang == "en":
        return {
            "about": [
                "What is {program} about?",
                "Tell me about {program}.",
                "Describe {program}.",
                "What will I study in {program}?",
                "Overview of {program}."
            ],
            "degree": [
                "What degree do I get after {program}?",
                "Degree title for {program}?",
                "{program} degree?"
            ],
            "length": [
                "How many years is {program}?",
                "{program} duration?",
                "How long is {program}?"
            ],
            "ects": [
                "How many ECTS does {program} have?",
                "{program} ECTS?",
                "Total credits for {program}?"
            ],
            "english": [
                "What English level is required for {program}?",
                "{program} English requirement?"
            ],
            "threshold_state": [
                "Grant threshold for {program}?",
                "{program} grant score?",
                "How many points for a grant in {program}?"
            ],
            "threshold_paid": [
                "Paid threshold for {program}?",
                "{program} paid score?",
                "How many points for paid admission in {program}?"
            ],
        }

    if lang == "kz":
        return {
            "about": [
                "{program} бағдарламасы не туралы?",
                "{program} туралы қысқаша айтшы.",
                "{program} сипаттамасы қандай?",
                "SDU-да {program} бойынша не оқимын?"
            ],
            "degree": [
                "{program} бітіргенде қандай дәреже беріледі?",
                "{program} дәрежесі қандай?"
            ],
            "length": [
                "{program} қанша жыл?",
                "{program} оқу ұзақтығы қандай?"
            ],
            "ects": [
                "{program} қанша ECTS?",
                "{program} кредиті қанша?"
            ],
            "english": [
                "{program} үшін ағылшын деңгейі қандай?",
                "{program} ағылшын талабы қандай?"
            ],
            "threshold_state": [
                "{program} грант шекті бал қанша?",
                "{program} грантқа қанша бал керек?"
            ],
            "threshold_paid": [
                "{program} ақылы шекті бал қанша?",
                "{program} ақылыға қанша бал керек?"
            ],
        }

    # ru
    return {
        "about": [
            "О чём программа {program}?",
            "Расскажи про {program}.",
            "Опиши программу {program}.",
            "Что изучают на {program}?",
            "{program} кратко."
        ],
        "degree": [
            "Какая степень после {program}?",
            "Как называется степень у {program}?",
            "{program} диплом какой?"
        ],
        "length": [
            "Сколько лет учиться на {program}?",
            "{program} длительность какая?",
            "Сколько длится {program}?"
        ],
        "ects": [
            "Сколько ECTS на {program}?",
            "{program} ECTS сколько?",
            "Сколько кредитов у {program}?"
        ],
        "english": [
            "Какой уровень английского нужен на {program}?",
            "{program} требования по английскому?"
        ],
        "threshold_state": [
            "Сколько нужно баллов на грант на {program}?",
            "{program} проходной на грант?",
            "Грант {program} сколько баллов?"
        ],
        "threshold_paid": [
            "Сколько нужно баллов на платное на {program}?",
            "{program} проходной на платное?",
            "Платное {program} сколько баллов?"
        ],
    }


def make_qa_rows(data: Dict[str, Any]) -> List[str]:
    """
    Возвращает список text_chunk (строки).
    dedup делаем по sha1(lower(text_chunk)).
    """
    rows: List[str] = []
    seen = set()

    faculties = data.get("faculties") or []
    degree_level = norm(data.get("degree_level") or "bachelor")
    source = norm(data.get("source") or "unknown")
    schema_version = norm(data.get("schema_version") or "unknown")

    for faculty in faculties:
        faculty_code = norm(faculty.get("faculty_code") or "")
        faculty_name = faculty.get("faculty_name") or {}
        programs = faculty.get("programs") or []

        for prog in programs:
            program_id = norm(prog.get("program_id") or "")
            program_name = prog.get("program_name") or {}
            cards = prog.get("cards") or {}

            degree = cards.get("degree") or {}
            program_length = cards.get("program_length") or {}
            ects = cards.get("ects") or {}
            english_level = norm(cards.get("english_level") or "")
            threshold_state = norm(cards.get("threshold_state") or "")
            threshold_paid = norm(cards.get("threshold_paid") or "")
            descr = prog.get("program_description") or {}

            for lang in ["en", "ru", "kz"]:
                pname = get_lang_value(program_name, lang)
                fname = get_lang_value(faculty_name, lang)
                if not pname or not fname:
                    continue

                prefix = build_answer_prefix(
                    program_name=pname,
                    faculty_name=fname,
                    program_id=program_id,
                    faculty_code=faculty_code,
                    lang=lang
                )

                degree_title = get_lang_value(degree, lang)
                length = get_lang_value(program_length, lang)
                ects_val = get_lang_value(ects, lang)
                descr_text = get_lang_value(descr, lang)

                T = q_templates(lang)

                def add(field: str, q: str, a: str):
                    q = norm(q)
                    a = norm(a)
                    if not q or not a:
                        return
                    # Добавим минимальный “паспорт” в ответ (без meta-колонки)
                    # чтобы в дальнейшем можно было понять источник и контекст.
                    a2 = f"{prefix}{a} (source={source}, level={degree_level}, v={schema_version}, field={field})"
                    text_chunk = f"Вопрос: {q}\nОтвет: {a2}"
                    h = sha1(text_chunk.lower())
                    if h in seen:
                        return
                    seen.add(h)
                    rows.append(text_chunk)

                if descr_text:
                    for q in T["about"]:
                        add("about", q.format(program=pname), descr_text)

                if degree_title:
                    for q in T["degree"]:
                        add("degree", q.format(program=pname), f"Degree: {degree_title}")

                if length:
                    for q in T["length"]:
                        add("program_length", q.format(program=pname), f"Program length: {length}")

                if ects_val:
                    for q in T["ects"]:
                        add("ects", q.format(program=pname), f"ECTS: {ects_val}")

                if english_level:
                    for q in T["english"]:
                        add("english_level", q.format(program=pname), f"English level requirement: {english_level}")

                if threshold_state:
                    for q in T["threshold_state"]:
                        add("threshold_state", q.format(program=pname), f"Grant threshold score: {threshold_state}")

                if threshold_paid:
                    for q in T["threshold_paid"]:
                        add("threshold_paid", q.format(program=pname), f"Paid threshold score: {threshold_paid}")

    return rows


def get_openai_client():
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is missing in environment (.env). Add it and rerun.")
    try:
        from openai import OpenAI
        return OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        raise RuntimeError(
            "OpenAI SDK is not available. Install it: pip install openai\n"
            f"Original error: {e}"
        )


def embed_texts(client, texts: List[str]) -> List[List[float]]:
    """
    Делает embeddings.create() и возвращает список векторов (list[float]).
    """
    # encoding_format="float" чтобы сразу получить list[float]
    resp = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
        encoding_format="float",
    )
    return [item.embedding for item in resp.data]


def insert_qa_chunks(text_chunks: List[str], sleep_s: float = 0.15):
    if not text_chunks:
        print("No rows to insert.")
        return

    client = get_openai_client()

    print(f"Prepared QA chunks: {len(text_chunks)}")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Embedding batch size: {BATCH_SIZE}, Insert batch size: {INSERT_BATCH_SIZE}")

    # 1) сначала считаем embeddings батчами
    embeddings: List[List[float]] = []
    for i in tqdm(range(0, len(text_chunks), BATCH_SIZE), desc="embed"):
        batch = text_chunks[i:i + BATCH_SIZE]

        # retry на случай временных ошибок сети/лимитов
        for attempt in range(5):
            try:
                embs = embed_texts(client, batch)
                embeddings.extend(embs)
                break
            except Exception as e:
                if attempt == 4:
                    raise
                time.sleep(1.0 * (attempt + 1))

        time.sleep(sleep_s)

    if len(embeddings) != len(text_chunks):
        raise RuntimeError("Embedding count mismatch. Something went wrong.")

    # 2) вставляем в Supabase
    payload = [{"text_chunk": t, "embedding": e} for t, e in zip(text_chunks, embeddings)]

    for i in tqdm(range(0, len(payload), INSERT_BATCH_SIZE), desc="insert qa_chunks"):
        chunk = payload[i:i + INSERT_BATCH_SIZE]

        # retry на insert
        for attempt in range(5):
            try:
                sb.table(TABLE_QA_CHUNKS).insert(chunk).execute()
                break
            except Exception as e:
                if attempt == 4:
                    raise
                time.sleep(1.0 * (attempt + 1))

    print("DONE: inserted qa_chunks with embeddings.")


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="Path to SDU bachelor programs JSON")
    ap.add_argument("--limit", type=int, default=0, help="Limit number of generated QA chunks (0 = no limit)")
    args = ap.parse_args()

    with open(args.json, "r", encoding="utf-8") as f:
        data = json.load(f)

    text_chunks = make_qa_rows(data)
    if args.limit and args.limit > 0:
        text_chunks = text_chunks[:args.limit]

    insert_qa_chunks(text_chunks)


if __name__ == "__main__":
    main()