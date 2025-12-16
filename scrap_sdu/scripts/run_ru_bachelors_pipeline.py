#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ONE-CLICK pipeline (RU only):
- discovers bachelor program URLs for 3 faculties
- scrapes HTML into data/bachelor_ru/raw_html/html
- writes pages.jsonl
- extracts canonical.csv

Deps:
  pip install requests beautifulsoup4 lxml
"""

import csv
import hashlib
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlsplit, urlunsplit

import requests
from bs4 import BeautifulSoup


# ====== CONFIG ======

FACULTIES = [
    # SSBL (у них индекс бакалавриата обычно отдельный "bachelor-bslawru")
    {
        "code": "ssbl",
        "home_url": "https://sdu.edu.kz/ru/shkola-socialnyh-nauk-businesa-i-prava/",
        "fallback_bachelor_index": "https://sdu.edu.kz/ru/shkola-socialnyh-nauk-businesa-i-prava/bachelor-bslawru/",
    },
    # Education & Humanities
    {
        "code": "edu_hum",
        "home_url": "https://sdu.edu.kz/ru/%d1%84%d0%b0%d0%ba%d1%83%d0%bb%d1%8c%d1%82%d0%b5%d1%82-%d0%bf%d0%b5%d0%b4%d0%b0%d0%b3%d0%be%d0%b3%d0%b8%d0%ba%d0%b8-%d0%b8-%d0%b3%d1%83%d0%bc%d0%b0%d0%bd%d0%b8%d1%82%d0%b0%d1%80%d0%bd%d1%8b%d1%85/",
        "fallback_bachelor_index": None,
    },
    # Engineering & Natural Sciences
    {
        "code": "eng_sci",
        "home_url": "https://sdu.edu.kz/ru/%d0%b8%d0%bd%d0%b6%d0%b5%d0%bd%d0%b5%d1%80%d0%bd%d1%8b%d0%b5-%d0%b8-%d0%b5%d1%81%d1%82%d0%b5%d1%81%d1%82%d0%b2%d0%b5%d0%bd%d0%bd%d1%8b%d0%b5-%d0%bd%d0%b0%d1%83%d0%ba%d0%b8/",
        "fallback_bachelor_index": None,
    },
]

OUT_ROOT = Path("data/bachelor_ru")
OUT_URLS = OUT_ROOT / "urls" / "program_urls.txt"
OUT_JSONL = OUT_ROOT / "raw" / "pages.jsonl"
OUT_HTML_DIR = OUT_ROOT / "raw_html" / "html"
OUT_CSV = OUT_ROOT / "parsed" / "canonical.csv"

TIMEOUT = 30
RETRIES = 2
SLEEP = 1.0


# ====== HELPERS ======

def now_iso_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def norm_url(u: str) -> str:
    if not u:
        return ""
    p = urlsplit(u.strip())
    path = (p.path or "").rstrip("/")
    return urlunsplit((p.scheme.lower() or "https", p.netloc.lower(), path, "", ""))


def safe_slug_from_url(url: str) -> str:
    p = urlsplit(url)
    path = (p.path or "").strip("/")
    last = path.split("/")[-1] if path else ""
    last = last.strip() or "index"

    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", last)
    slug = re.sub(r"-{2,}", "-", slug).strip("-_")

    if not slug:
        slug = hashlib.md5(url.encode("utf-8")).hexdigest()[:16]
    return slug


def fetch(session: requests.Session, url: str, timeout: int = TIMEOUT, retries: int = RETRIES, sleep: float = SLEEP) -> tuple[int, str]:
    last_exc = None
    for attempt in range(retries + 1):
        try:
            r = session.get(url, timeout=timeout)
            return r.status_code, (r.text or "")
        except Exception as e:
            last_exc = e
            if attempt < retries:
                time.sleep(sleep)
    raise RuntimeError(f"Failed to fetch after retries: {url} | last_error={last_exc}")


def bs(html: str) -> BeautifulSoup:
    try:
        return BeautifulSoup(html, "lxml")
    except Exception:
        return BeautifulSoup(html, "html.parser")


def norm_text(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


def is_internal_ru(u: str) -> bool:
    u = norm_url(u)
    return u.startswith("https://sdu.edu.kz/ru/")


# ====== DISCOVER ======

BACHELOR_HINTS_TEXT = {"бакалавр", "бакалавриат", "bachelor", "bakalavr"}
BACHELOR_HINTS_PATH = ("бакалавр", "бакалавриат", "bachelor", "bakalavr", "bachelor-")

EXCLUDE_PATH_PARTS = (
    "магистрат", "doktor", "докторан", "phd", "minor", "news", "events", "kontakt", "contacts",
    "приемная-комиссия", "admission", "karta-sayta", "карта-сайта"
)

def find_bachelor_index_url(soup: BeautifulSoup, base_url: str) -> Optional[str]:
    """
    Ищем на странице факультета ссылку на индекс бакалавриата.
    Приоритет: href содержит '/программы/бакалавриат' или похожее.
    """
    candidates = []
    for a in soup.select("a[href]"):
        href = (a.get("href") or "").strip()
        if not href:
            continue
        full = norm_url(urljoin(base_url, href))
        if not is_internal_ru(full):
            continue

        t = norm_text(a.get_text(" ", strip=True))
        path = urlsplit(full).path.lower()

        if any(h in path for h in BACHELOR_HINTS_PATH) or any(h in t for h in BACHELOR_HINTS_TEXT):
            # желательно чтобы это был "индекс", а не конкретная программа:
            # часто индекс содержит "программы/бакалавриат" или "bachelor-...ru" без хвоста
            score = 0
            if "программы" in path and "бакалавриат" in path:
                score += 10
            if "bachelor" in path and path.count("/") <= 5:
                score += 6
            if "бакалавр" in path:
                score += 4
            # короткие пути чаще индексы
            score += max(0, 8 - path.count("/"))
            candidates.append((score, full))

    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def discover_program_urls_for_faculty(session: requests.Session, fac: dict) -> list[str]:
    code = fac["code"]
    home = fac["home_url"]

    status, html = fetch(session, home)
    if status != 200 or len(html) < 2000:
        return []

    soup = bs(html)

    bachelor_index = find_bachelor_index_url(soup, home)
    if not bachelor_index and fac.get("fallback_bachelor_index"):
        bachelor_index = fac["fallback_bachelor_index"]

    if not bachelor_index:
        # Последний шанс: иногда бакалавриат можно угадать по паттерну
        # (мы не делаем “магии”, просто вернем пусто)
        return []

    st2, html2 = fetch(session, bachelor_index)
    if st2 != 200 or len(html2) < 2000:
        return []

    soup2 = bs(html2)

    urls = set()
    for a in soup2.select("a[href]"):
        href = (a.get("href") or "").strip()
        if not href:
            continue
        full = norm_url(urljoin(bachelor_index, href))
        if not is_internal_ru(full):
            continue

        p = urlsplit(full).path.lower()
        # выкидываем сам индекс бакалавриата, якоря и мусор
        if full == norm_url(bachelor_index):
            continue
        if any(x in p for x in EXCLUDE_PATH_PARTS):
            continue
        if full.endswith((".pdf", ".jpg", ".jpeg", ".png", ".zip")):
            continue

        # Очень частая структура: ссылки на конкретные программы идут “глубже”, чем индекс.
        # Оставим только то, что выглядит как отдельная страница (не раздел/каталог).
        if p.count("/") < 4:
            continue

        urls.add(full)

    return sorted(urls)


# ====== EXTRACT (RU) ======

CARD_HINTS = {
    "degree": ["степень"],
    "program_length": ["продолжительность"],
    "threshold_grant": ["пороговый балл", "грант"],
    "threshold_paid": ["пороговый балл", "платн"],
    "ects": ["ects"],
    "english_level": ["требуемый уровень", "английск"],
}


def norm_spaces(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def norm_key(s: str) -> str:
    s = norm_spaces(s).lower()
    s = re.sub(r"[^\w\s-]+", "", s, flags=re.UNICODE)
    return s


def to_int(x: Optional[str]) -> Optional[int]:
    if not x:
        return None
    m = re.search(r"(\d+)", x)
    return int(m.group(1)) if m else None


def extract_cards_ru(soup: BeautifulSoup) -> dict:
    out = {
        "degree": None,
        "program_length": None,
        "threshold_grant": None,
        "threshold_paid": None,
        "ects": None,
        "english_level": None,
    }

    pairs = []
    for wrapper in soup.select(".elementor-icon-box-wrapper"):
        t_el = wrapper.select_one(".elementor-icon-box-title")
        d_el = wrapper.select_one(".elementor-icon-box-description")
        title = norm_spaces(t_el.get_text(" ", strip=True)) if t_el else ""
        desc = norm_spaces(d_el.get_text(" ", strip=True)) if d_el else ""
        if title and desc:
            pairs.append((title, desc))

    if not pairs:
        for node in soup.select(".elementor-widget-icon-box, .elementor-icon-box-content"):
            t_el = node.select_one(".elementor-icon-box-title") or node.find(["h3", "h4", "h5"])
            d_el = node.select_one(".elementor-icon-box-description") or node.find("p")
            title = norm_spaces(t_el.get_text(" ", strip=True)) if t_el else ""
            desc = norm_spaces(d_el.get_text(" ", strip=True)) if d_el else ""
            if title and desc:
                pairs.append((title, desc))

    def pick(field: str) -> Optional[str]:
        hints = [norm_key(h) for h in CARD_HINTS[field]]

        # спец-логика для грант/платное
        for title, desc in pairs:
            t = norm_key(title)
            if field == "threshold_grant" and ("пороговый балл" in t) and ("грант" in t):
                return desc
            if field == "threshold_paid" and ("пороговый балл" in t) and ("платн" in t):
                return desc

        # общий матч
        for title, desc in pairs:
            t = norm_key(title)
            if any(h and h in t for h in hints):
                return desc
        return None

    out["degree"] = pick("degree")
    out["program_length"] = pick("program_length")

    out["threshold_grant"] = to_int(pick("threshold_grant"))
    out["threshold_paid"] = to_int(pick("threshold_paid"))
    out["ects"] = to_int(pick("ects"))

    out["english_level"] = pick("english_level") or "Не требуется"
    return out


def extract_program_name(soup: BeautifulSoup, fallback: str) -> str:
    for sel in ("h1", ".elementor-heading-title"):
        el = soup.select_one(sel)
        if el:
            t = norm_spaces(el.get_text(" ", strip=True))
            if t:
                return t
    return fallback


def extract_program_description_ru(soup: BeautifulSoup) -> Optional[str]:
    want = "описание программы"

    for a in soup.select("a.elementor-toggle-title"):
        if norm_text(a.get_text(" ", strip=True)) != want:
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

    return None


def detect_creative_exam_ru(soup: BeautifulSoup) -> tuple[int, Optional[str]]:
    text = " ".join(soup.get_text(" ", strip=True).split())
    pat = r"творческ(ий|ое)\s+экзамен.*2024\s*[-–]\s*2025"
    m = re.search(pat, text, flags=re.I)
    if not m:
        return 0, None
    start = max(m.start() - 60, 0)
    end = min(m.end() + 60, len(text))
    return 1, text[start:end]


# ====== PIPELINE ======

def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    OUT_URLS.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML_DIR.mkdir(parents=True, exist_ok=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (compatible; scrap_sdu/1.0; +https://sdu.edu.kz/)",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "ru,en;q=0.8",
            "Connection": "keep-alive",
        }
    )

    # 1) DISCOVER
    all_urls = []
    per_fac = {}
    for fac in FACULTIES:
        urls = discover_program_urls_for_faculty(session, fac)
        per_fac[fac["code"]] = urls
        all_urls.extend([(fac["code"], u) for u in urls])
        print(f"[DISCOVER] {fac['code']}: found program urls = {len(urls)}")

    # save urls list
    OUT_URLS.write_text("\n".join([u for _, u in all_urls]) + ("\n" if all_urls else ""), encoding="utf-8")

    # 2) SCRAPE -> pages.jsonl + html files
    ok_cnt = 0
    with OUT_JSONL.open("w", encoding="utf-8") as w:
        for idx, (fac_code, url) in enumerate(all_urls, start=1):
            base_slug = safe_slug_from_url(url)
            slug = f"{fac_code}__{base_slug}"
            html_path = OUT_HTML_DIR / f"{slug}.html"

            http_status = 0
            html = ""
            ok = False
            try:
                http_status, html = fetch(session, url)
                html_path.write_text(html, encoding="utf-8", errors="ignore")
                ok = (http_status == 200 and len(html) > 5000)
                ok_cnt += int(ok)
            except Exception:
                pass

            rec = {
                "lang": "ru",
                "faculty": fac_code,
                "url": url,
                "slug": slug,
                "html_path": str(html_path),
                "http_status": int(http_status),
                "ok": bool(ok),
                "fetched_at": now_iso_utc(),
            }
            w.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"[{idx}/{len(all_urls)}] {slug} | status={http_status} | ok={ok}")

    print(f"\nSaved: {OUT_JSONL} | urls={len(all_urls)} | ok={ok_cnt}")

    # 3) EXTRACT -> canonical.csv
    rows = []
    with OUT_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            html_path = Path(rec["html_path"])
            if not html_path.exists():
                continue
            html = html_path.read_text(encoding="utf-8", errors="ignore")
            soup = bs(html)

            program_name = extract_program_name(soup, fallback=rec.get("slug", ""))
            cards = extract_cards_ru(soup)
            program_description = extract_program_description_ru(soup)
            creative_flag, creative_text = detect_creative_exam_ru(soup)

            rows.append(
                {
                    "lang": "ru",
                    "faculty": rec.get("faculty"),
                    "source_url": rec.get("url"),
                    "slug": rec.get("slug"),
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
    with OUT_CSV.open("w", encoding="utf-8", newline="") as w:
        wr = csv.DictWriter(w, fieldnames=cols)
        wr.writeheader()
        wr.writerows(rows)

    def miss(col: str) -> int:
        return sum(1 for r in rows if r.get(col) in (None, "", "None"))

    print(f"\nSaved: {OUT_CSV} | rows={len(rows)}")
    for c in ["degree", "program_length", "threshold_grant", "threshold_paid", "ects", "english_level", "program_description"]:
        print(f"missing {c}: {miss(c)}")

    for fac in FACULTIES:
        code = fac["code"]
        cnt = sum(1 for r in rows if r.get("faculty") == code)
        print(f"[FINAL] {code}: rows in csv = {cnt}")


if __name__ == "__main__":
    main()