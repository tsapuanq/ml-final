#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#scrap_sdu/scripts/extract_ssbl_ru.py

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Optional

from bs4 import BeautifulSoup


def norm(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def norm_key(s: str) -> str:
    s = norm(s).lower()
    s = re.sub(r"[^\w\s-]+", "", s, flags=re.UNICODE)
    return s


def to_int(x: Optional[str]) -> Optional[int]:
    if not x:
        return None
    m = re.search(r"(\d+)", x)
    return int(m.group(1)) if m else None


# RU labels / ключевые слова для матчей по карточкам
CARD_HINTS = {
    "degree": ["степень"],
    "program_length": ["продолжительность"],
    "threshold_grant": ["пороговый балл", "грант"],
    "threshold_paid": ["пороговый балл", "платное", "платное отделение"],
    "ects": ["ects"],
    "english_level": ["требуемый уровень", "английского"],
}


def extract_cards_ru(soup: BeautifulSoup) -> dict:
    """
    Вытаскиваем 6 карточек слева: title + description.
    Очень устойчиво: собираем все пары (title -> desc) и затем матчим по подсказкам.
    """
    out = {
        "degree": None,
        "program_length": None,
        "threshold_grant": None,
        "threshold_paid": None,
        "ects": None,
        "english_level": None,
    }

    pairs = []  # (title, desc)
    for wrapper in soup.select(".elementor-icon-box-wrapper"):
        t_el = wrapper.select_one(".elementor-icon-box-title")
        d_el = wrapper.select_one(".elementor-icon-box-description")
        title = norm(t_el.get_text(" ", strip=True)) if t_el else ""
        desc = norm(d_el.get_text(" ", strip=True)) if d_el else ""
        if title and desc:
            pairs.append((title, desc))

    # fallback — иногда структура не wrapper, а content
    if not pairs:
        for node in soup.select(".elementor-widget-icon-box, .elementor-icon-box-content"):
            t_el = node.select_one(".elementor-icon-box-title") or node.find(["h3", "h4", "h5"])
            d_el = node.select_one(".elementor-icon-box-description") or node.find("p")
            title = norm(t_el.get_text(" ", strip=True)) if t_el else ""
            desc = norm(d_el.get_text(" ", strip=True)) if d_el else ""
            if title and desc:
                pairs.append((title, desc))

    def pick(field: str) -> Optional[str]:
        hints = [norm_key(h) for h in CARD_HINTS[field]]
        for title, desc in pairs:
            t = norm_key(title)
            # threshold_grant vs threshold_paid: различаем по словам "грант/платное"
            if field == "threshold_grant":
                if "пороговый балл" in t and ("грант" in t):
                    return desc
            if field == "threshold_paid":
                if "пороговый балл" in t and ("платн" in t):
                    return desc

            # общий случай
            if all(h in t for h in hints if h):
                return desc

        # слабый матч: любое вхождение
        for title, desc in pairs:
            t = norm_key(title)
            if any(h and h in t for h in hints):
                return desc
        return None

    out["degree"] = pick("degree")
    out["program_length"] = pick("program_length")
    out["english_level"] = pick("english_level")

    out["threshold_grant"] = to_int(pick("threshold_grant"))
    out["threshold_paid"] = to_int(pick("threshold_paid"))
    out["ects"] = to_int(pick("ects"))

    # если англ уровень реально пуст — ставим “Не требуется” (как ты хотел)
    if not out["english_level"]:
        out["english_level"] = "Не требуется"

    return out


def extract_program_name(soup: BeautifulSoup, fallback: str) -> str:
    h1 = soup.select_one("h1")
    if h1:
        t = norm(h1.get_text(" ", strip=True))
        if t:
            return t
    # Elementor headings
    h = soup.select_one(".elementor-heading-title")
    if h:
        t = norm(h.get_text(" ", strip=True))
        if t:
            return t
    return fallback


def extract_program_description_ru(soup: BeautifulSoup) -> Optional[str]:
    want = "описание программы"

    def n(s: str) -> str:
        return " ".join((s or "").strip().lower().split())

    for a in soup.select("a.elementor-toggle-title"):
        if n(a.get_text(" ", strip=True)) != want:
            continue

        item = a.find_parent(class_=re.compile(r"elementor-toggle-item"))
        if item:
            content = item.select_one(".elementor-toggle-content, .elementor-tab-content")
        else:
            content = a.find_next(class_=re.compile(r"elementor-toggle-content|elementor-tab-content"))

        if content:
            txt = content.get_text("\n", strip=True)
            txt = "\n".join([line.strip() for line in txt.splitlines() if line.strip()])
            return txt or None

    # fallback: иногда в другом теге (редко)
    full = soup.get_text("\n", strip=True)
    return full if False else None


def detect_creative_exam_ru(soup: BeautifulSoup) -> tuple[int, Optional[str]]:
    # На сайте встречается “Творческий экзамен на 2024–2025 учебный год”
    text = " ".join(soup.get_text(" ", strip=True).split())
    pat = r"творческ(ий|ое)\s+экзамен.*2024\s*[-–]\s*2025"
    m = re.search(pat, text, flags=re.I)
    if not m:
        return 0, None
    start = max(m.start() - 60, 0)
    end = min(m.end() + 60, len(text))
    return 1, text[start:end]


def open_html(html_path: Path) -> str:
    if html_path.exists():
        return html_path.read_text(encoding="utf-8", errors="ignore")
    # если путь был сохранен относительным/из другого cwd
    alt = Path.cwd() / html_path
    if alt.exists():
        return alt.read_text(encoding="utf-8", errors="ignore")
    raise FileNotFoundError(f"HTML not found: {html_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    in_path = Path(args.in_jsonl)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            html_path = Path(rec["html_path"])
            html = open_html(html_path)

            # parser
            try:
                soup = BeautifulSoup(html, "lxml")
            except Exception:
                soup = BeautifulSoup(html, "html.parser")

            slug = rec.get("slug") or ""
            program_name = extract_program_name(soup, fallback=slug)

            cards = extract_cards_ru(soup)
            program_description = extract_program_description_ru(soup)
            creative_flag, creative_text = detect_creative_exam_ru(soup)

            rows.append(
                {
                    "lang": "ru",
                    "source_url": rec.get("url"),
                    "slug": slug,
                    "program_name": program_name,

                    "degree": cards["degree"],
                    "program_length": cards["program_length"],
                    "threshold_grant": cards["threshold_grant"],
                    "threshold_paid": cards["threshold_paid"],
                    "ects": cards["ects"],
                    "english_level": cards["english_level"],

                    "program_description": program_description,

                    "creative_exam": creative_flag,
                    "creative_exam_text": creative_text,

                    "http_status": rec.get("http_status"),
                    "ok": rec.get("ok"),
                    "fetched_at": rec.get("fetched_at"),
                }
            )

    cols = list(rows[0].keys()) if rows else []
    with out_path.open("w", encoding="utf-8", newline="") as w:
        wr = csv.DictWriter(w, fieldnames=cols)
        wr.writeheader()
        wr.writerows(rows)

    def miss(col: str) -> int:
        return sum(1 for r in rows if r.get(col) in (None, "", "None"))

    print(f"Saved: {out_path} | rows={len(rows)}")
    for c in [
        "degree",
        "program_length",
        "threshold_grant",
        "threshold_paid",
        "ects",
        "english_level",
        "program_description",
    ]:
        print(f"missing {c}: {miss(c)}")


if __name__ == "__main__":
    main()