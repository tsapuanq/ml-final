import os, re, json, time, hashlib
from typing import List
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

MAX_ITEMS = 350
PARAS_PER = 12

def norm(t: str) -> str:
    t = (t or "").strip()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"^[\-\•\*\u2022]+\s*", "", t)
    return t

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def embed_batch(texts: List[str]) -> List[List[float]]:
    r = client.embeddings.create(
        model=EMB_MODEL,
        input=texts,
        encoding_format="float",
    )
    return [d.embedding for d in r.data]

def gen_paraphrases(base: str, lang: str, n: int) -> List[str]:
    base = norm(base)
    if not base:
        return []

    if lang == "kk":
        prompt = (
            f"Төмендегі сұраққа {n} түрлі қазақша нұсқа жаса.\n"
            f"Тек JSON массив (string[]) қайтар. Басқа мәтін жазба.\n"
            f"Сұрақ: {base}"
        )
    elif lang == "ru":
        prompt = (
            f"Сделай {n} разных русских перефраз вопроса.\n"
            f"Верни ТОЛЬКО JSON массив строк. Без нумерации, без текста.\n"
            f"Вопрос: {base}"
        )
    else:
        prompt = (
            f"Create {n} different English paraphrases of the question.\n"
            f"Return ONLY a JSON array of strings.\n"
            f"Question: {base}"
        )

    r = client.responses.create(
        model="gpt-4o-mini",
        instructions="Return ONLY a valid JSON array of strings. No extra text.",
        input=prompt,
    )
    txt = (r.output_text or "").strip()

    try:
        arr = json.loads(txt)
        if not isinstance(arr, list):
            return []
        out = []
        for x in arr:
            if isinstance(x, str):
                x = norm(x)
                if x:
                    out.append(x)
        uniq, seen = [], set()
        for x in out:
            xl = x.lower()
            if xl not in seen:
                uniq.append(x)
                seen.add(xl)
        return uniq
    except Exception:
        return []

def main():
    res = sb.rpc("get_paraphrase_candidates", {"max_rows": MAX_ITEMS}).execute()
    cand = res.data or []
    print("candidates:", len(cand))
    if not cand:
        print("Nothing to expand.")
        return

    new_rows = []
    done_rows = []

    for c in tqdm(cand, desc="generate"):
        aid = int(c["answer_id"])
        lang = (c.get("lang") or "ru").lower()
        base = norm(c["search_text"])
        base_hash = c["base_search_hash"]

        paras = gen_paraphrases(base, lang, PARAS_PER)

        for p in paras:
            if not p or p.lower() == base.lower():
                continue
            sh = sha1(f"{aid}|{p.lower()}")
            new_rows.append({
                "answer_id": aid,
                "lang": lang,
                "search_text": p,
                "search_hash": sh,
                "meta": {"source": "paraphrase", "base": base[:200]},
            })

        done_rows.append({"base_search_hash": base_hash})
        time.sleep(0.05)

    dedup = {r["search_hash"]: r for r in new_rows}
    new_rows = list(dedup.values())
    print("new rows:", len(new_rows))

    if new_rows:
        B = 200
        for i in tqdm(range(0, len(new_rows), B), desc="embed+upsert"):
            batch = new_rows[i:i+B]
            embs = embed_batch([r["search_text"] for r in batch])
            for r, e in zip(batch, embs):
                r["embedding"] = e
            sb.table("qa_index").upsert(batch, on_conflict="search_hash").execute()
            time.sleep(0.10)

    sb.table("qa_paraphrase_done").upsert(done_rows, on_conflict="base_search_hash").execute()
    print("DONE")

if __name__ == "__main__":
    main()