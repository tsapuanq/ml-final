import os
from dotenv import load_dotenv
from supabase import create_client


CHARS_PER_TOKEN = 3.6

load_dotenv()
sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

def est_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, int(len(text) / CHARS_PER_TOKEN))

def fetch_all(table: str, cols: str):
    out = []
    start = 0
    while True:
        r = sb.table(table).select(cols).range(start, start + 999).execute()
        rows = r.data or []
        out.extend(rows)
        if len(rows) < 1000:
            break
        start += 1000
    return out

def main():
    answers = fetch_all("qa_answers", "answer_id,answer,lang")
    index_rows = fetch_all("qa_index", "index_id,search_text,lang,meta")

    total_in = sum(est_tokens(r.get("answer", "")) for r in answers)
    total_out = int(total_in * 0.9)

    rules = []
    for r in index_rows:
        meta = r.get("meta") or {}
        src = meta.get("source")
        st = r.get("search_text") or ""
        short = (len(st) <= 40) or (len(st.split()) <= 2)
        if src == "rules" and short:
            rules.append(r)

    paras_per = 12
    para_in = len(rules) * 220
    para_out = len(rules) * 300

    print("=== ESTIMATE ===")
    print("qa_answers rows:", len(answers))
    print("qa_index rows:", len(index_rows))
    print("rules+short candidates:", len(rules))
    print()
    print("[Clean answers] approx input tokens:", total_in)
    print("[Clean answers] approx output tokens:", total_out)
    print()
    print("[Paraphrases] approx input tokens:", para_in, f"(~{len(rules)} calls)")
    print("[Paraphrases] approx output tokens:", para_out)
    print()
    print("Embeddings tokens are usually much smaller vs LLM and cheap.")

if __name__ == "__main__":
    main()