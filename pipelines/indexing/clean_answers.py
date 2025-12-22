#pipelines/indexing/clean_answers.py
import os, time
from dotenv import load_dotenv
from supabase import create_client
from openai import OpenAI
from tqdm import tqdm

load_dotenv()
sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

LIMIT = 1500   
BATCH = 150

def clean_one(answer: str, lang: str) -> str:
    if lang == "kk":
        instr = (
            "Мәтінді ТЕК берілген ақпаратпен ретте.\n"
            "Жаңа факт қоспа. Қысқа әрі түсінікті қыл€.\n"
            "Формат: 1 сөйлем + 2–6 bullet.\n"
            "Тек дайын жауапты қайтар."
        )
    elif lang == "ru":
        instr = (
            "Отредактируй ответ, используя ТОЛЬКО исходный текст.\n"
            "Запрещено добавлять новые факты.\n"
            "Формат: 1 предложение + 2–6 буллетов.\n"
            "Верни только готовый ответ."
        )
    else:
        instr = (
            "Rewrite using ONLY the provided text, no new facts.\n"
            "Format: 1 sentence + 2–6 bullets.\n"
            "Return only the final answer."
        )

    r = client.responses.create(
        model="gpt-4o-mini",
        instructions=instr,
        input=answer,
    )
    return (r.output_text or "").strip()

def main():
    res = (sb.table("qa_answers")
           .select("answer_id,answer,lang,answer_clean")
           .is_("answer_clean", "null")
           .limit(LIMIT)
           .execute())
    rows = res.data or []
    print("to_clean:", len(rows))
    if not rows:
        print("nothing to clean")
        return

    for i in tqdm(range(0, len(rows), BATCH), desc="clean+update"):
        batch = rows[i:i+BATCH]
        for r in batch:
            aid = r["answer_id"]
            lang = (r.get("lang") or "ru").lower()
            cleaned = clean_one(r["answer"], lang)
            sb.table("qa_answers").update({"answer_clean": cleaned}).eq("answer_id", aid).execute()
            time.sleep(0.03)

    print("DONE")

if __name__ == "__main__":
    main()