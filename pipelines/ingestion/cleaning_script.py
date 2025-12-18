import re
from pathlib import Path
import pandas as pd

INPUT_PATH = "QA_SDU.csv"       
DOCS_DIR = "docs"               
LOWERCASE_EMBEDDING = True      


def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip()             
    s = re.sub(r"\s+", " ", s) 
    return s                 

GARBAGE_TOKENS = {",", ".", "m", "mm", "?", "-", "!"}


KZ_CHARS = set("ӘәӨөҮүҰұІіҚқҒғҢңҺһ")

CYRILLIC_RE = re.compile(r"[А-Яа-яЁёІіҢңҒғҚқҰұҮүҺһӘәӨө]")
LATIN_RE = re.compile(r"[A-Za-z]")


def detect_language(text: str) -> str:
    """Примитивный детектор языка: kk / ru / en / mixed / other."""
    if not isinstance(text, str):
        return "other"
    text = text.strip()
    if not text:
        return "other"

    has_kz = any(ch in KZ_CHARS for ch in text)
    has_cyr = bool(CYRILLIC_RE.search(text))
    has_lat = bool(LATIN_RE.search(text))

    if has_kz:
        return "kk"

    if has_cyr and not has_lat:
        return "ru"

    if has_lat and not has_cyr:
        return "en"

    if has_cyr and has_lat:
        return "mixed"

    return "other"


def main():
    df = pd.read_csv(INPUT_PATH)

    df["Question"] = df["Question"].astype(str)
    df["Answer"] = df["Answer"].astype(str)

    df["q_len"] = df["Question"].str.len()
    df["a_len"] = df["Answer"].str.len()

    mask_garbage = (
        df["Question"].isin(GARBAGE_TOKENS)
        & df["Answer"].isin(GARBAGE_TOKENS)
    ) | (
        (df["q_len"] <= 2) & (df["a_len"] <= 2)
    )

    garbage_df = df[mask_garbage].copy()
    df = df[~mask_garbage].reset_index(drop=True)

    df["Question"] = df["Question"].apply(clean_text)
    df["Answer"] = df["Answer"].apply(clean_text)

    df["q_len"] = df["Question"].str.len()
    df["a_len"] = df["Answer"].str.len()

    dup_mask = df.duplicated(subset=["Question", "Answer"], keep="first")
    df = df[~dup_mask].reset_index(drop=True)

    df["embedding_text"] = df["Question"].str.cat(df["Answer"], sep=" ", na_rep="")
    if LOWERCASE_EMBEDDING:
        df["embedding_text"] = df["embedding_text"].str.lower()

    df["language"] = (df["Question"] + " " + df["Answer"]).apply(detect_language)

    df.insert(0, "id", range(1, len(df) + 1))

    docsdir = Path(DOCS_DIR)
    docsdir.mkdir(exist_ok=True)

    df.to_excel(docsdir / "QA_clean_base.xlsx", index=False)

    print("Готово!")
    print(f"Итоговый датасет: {len(df)} строк")
    print(f"Удалён мусор: {len(garbage_df)} строк")

    print("Распределение по language (auto):")
    print(df["language"].value_counts(dropna=False))
    print("Excel для ручной правки лежит в папке:", docsdir.resolve())

if __name__ == "__main__":
    main()