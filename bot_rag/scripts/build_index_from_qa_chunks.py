#bot_rag/scripts/build_index_from_qa_chunks.py

import os, re, hashlib, time
from typing import List, Dict, Tuple
from dotenv import load_dotenv
from supabase import create_client
from openai import OpenAI
from tqdm import tqdm

load_dotenv()

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_ROLE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]

client = OpenAI(api_key=OPENAI_API_KEY)
sb = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

EMB_MODEL = "text-embedding-3-small"
FETCH_LIMIT = 999999

PARAPHRASES_PER_ITEM = 0

KZ_CHARS = set("әөүұқғңһі")

ALIASES = {
    "dorm": "жатақхана общежитие общага dormitory hostel residence price cost payment fee",
    "общежитие": "жатақхана dorm dormitory price cost payment fee",
    "общага": "общежитие жатақхана dorm dormitory price cost payment fee",
    "fx": "foreign exchange валюта обмен курс rate",
    "imo": "international office SDU visa documents",
    "gpa": "grade point average балл оценка",
    "ssc": "student service center SDU справка",
    "spt": "student points transcript SDU",
}

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def normalize(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r"\s+", " ", t)
    return t

def detect_lang(text: str) -> str:
    t = (text or "").lower()
    if any(ch in t for ch in KZ_CHARS): return "kk"
    if any("а" <= ch <= "я" for ch in t) or "ё" in t: return "ru"
    return "en"

def parse_chunk(text_chunk: str) -> Tuple[str, str]:
    parts = re.split(r"Ответ:\s*", text_chunk, maxsplit=1)
    if len(parts) == 2:
        left, ans = parts[0], parts[1]
        q = re.sub(r"^Вопрос:\s*", "", left).strip()
        return normalize(q), normalize(ans)
    return "", normalize(text_chunk)

def make_search_texts(question: str) -> List[str]:
    q = normalize(question)
    out = []
    if not q:
        return out

    out.append(q)

    q_low = q.lower()
    if len(q_low) <= 10 or len(q_low.split()) <= 2:
        out.append(f"{q} что это такое объясни анықтама")

    for k, extra in ALIASES.items():
        if re.search(rf"\b{re.escape(k)}\b", q_low):
            out.append(f"{q} {extra}")

    uniq, seen = [], set()
    for t in out:
        t2 = normalize(t)
        if t2 and t2 not in seen:
            uniq.append(t2); seen.add(t2)
    return uniq

def embed_batch(texts: List[str]) -> List[List[float]]:
    r = client.embeddings.create(
        model=EMB_MODEL,
        input=texts,
        encoding_format="float",
    )
    return [d.embedding for d in r.data]

def fetch_qa_chunks(limit: int) -> List[Dict]:
    out = []
    start = 0
    page = 1000
    while True:
        r = sb.table("qa_chunks").select("id,text_chunk").range(start, start+page-1).execute()
        rows = r.data or []
        out.extend(rows)
        if len(rows) < page or len(out) >= limit:
            return out[:limit]
        start += page

def ensure_answer_ids(answer_hashes: List[str]) -> Dict[str, int]:
    out = {}
    start = 0
    while True:
        r = sb.table("qa_answers").select("answer_id,answer_hash").range(start, start+1000-1).execute()
        rows = r.data or []
        for row in rows:
            if row.get("answer_hash"):
                out[row["answer_hash"]] = row["answer_id"]
        if len(rows) < 1000:
            break
        start += 1000
    missing = [h for h in answer_hashes if h not in out]
    if missing:
        raise RuntimeError(f"Missing answer_ids for {len(missing)} answers (should not happen).")
    return out

def main():
    chunks = fetch_qa_chunks(FETCH_LIMIT)
    print("qa_chunks:", len(chunks))

    parsed = []
    unique_answers = {}
    for row in tqdm(chunks, desc="parse"):
        q, a = parse_chunk(row["text_chunk"])
        if not a:
            continue
        lang = detect_lang(q + " " + a)
        ah = sha1(a.lower())
        parsed.append({"q": q, "a": a, "lang": lang, "ah": ah, "src_id": row["id"]})
        if ah not in unique_answers:
            unique_answers[ah] = {"answer": a, "lang": lang, "answer_hash": ah, "meta": {"source_chunk_id": row["id"]}}

    print("unique answers:", len(unique_answers))

    answers_list = list(unique_answers.values())
    B = 500
    for i in tqdm(range(0, len(answers_list), B), desc="upsert qa_answers"):
        sb.table("qa_answers").upsert(answers_list[i:i+B], on_conflict="answer_hash").execute()

    answer_id_map = ensure_answer_ids(list(unique_answers.keys()))

    index_rows = []
    for rec in tqdm(parsed, desc="build index"):
        answer_id = answer_id_map[rec["ah"]]
        q = rec["q"]
        if not q:
            continue
        lang = rec["lang"]

        for st in make_search_texts(q):
            sh = sha1(f"{answer_id}|{st.lower()}")
            index_rows.append({
                "answer_id": answer_id,
                "lang": lang,
                "search_text": st,
                "search_hash": sh,
                "meta": {"source": "rules", "src_chunk_id": rec["src_id"]}
            })

    dedup = {}
    for r in index_rows:
        dedup[r["search_hash"]] = r
    index_rows = list(dedup.values())

    print("qa_index rows:", len(index_rows))

    B = 200
    for i in tqdm(range(0, len(index_rows), B), desc="embed+upsert qa_index"):
        batch = index_rows[i:i+B]
        texts = [r["search_text"] for r in batch]
        embs = embed_batch(texts)
        for r, e in zip(batch, embs):
            r["embedding"] = e
        sb.table("qa_index").upsert(batch, on_conflict="search_hash").execute()
        time.sleep(0.15)

    print("DONE")

if __name__ == "__main__":
    main()