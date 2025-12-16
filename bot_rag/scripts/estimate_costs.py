import os
from dotenv import load_dotenv
from supabase import create_client

# Токены оценим приблизительно через длину текста.
# Для RU/KK грубо: 1 токен ~ 3.2-4.0 символа. Возьмём 3.6 (консервативно).
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

    # 1) оценка чистки answers (answer -> answer_clean)
    total_in = sum(est_tokens(r.get("answer", "")) for r in answers)
    # output примерно сравним по размеру (иногда чуть меньше). Возьмём 0.9x
    total_out = int(total_in * 0.9)

    # 2) оценка перефразов для “rules”-строк (только где meta.source='rules' и короткие)
    rules = []
    for r in index_rows:
        meta = r.get("meta") or {}
        src = meta.get("source")
        st = r.get("search_text") or ""
        short = (len(st) <= 40) or (len(st.split()) <= 2)
        if src == "rules" and short:
            rules.append(r)

    # допустим делаем 12 перефраз на строку
    paras_per = 12
    # input на 1 запрос ~ (инструкции+вопрос) ~ 220 токенов
    # output на 12 перефраз ~ 300 токенов
    # сделаем оценку:
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