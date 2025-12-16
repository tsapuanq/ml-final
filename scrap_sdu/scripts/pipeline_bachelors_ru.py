#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import hashlib
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlsplit, urlunsplit, unquote

import requests
from bs4 import BeautifulSoup


# -----------------------------
# Utils
# -----------------------------

def now_iso_utc() -> str:
    return datetime.now(timezone.utc).isoformat()

def norm(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def norm_key(s: str) -> str:
    s = norm(s).lower()
    s = re.sub(r"[^\w\s-]+", "", s, flags=re.UNICODE)
    return s

def norm_url(u: str) -> str:
    if not u:
        return ""
    u = unquote(str(u)).strip()
    p = urlsplit(u)
    path = (p.path or "").rstrip("/")
    return urlunsplit((p.scheme.lower(), p.netloc.lower(), path, "", ""))

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

def is_sdu_ru_url(u: str) -> bool:
    if not u:
        return False
    p = urlsplit(u)
    if p.netloc.lower() != "sdu.edu.kz":
        return False
    return p.path.startswith("/ru/")

def is_media(u: str) -> bool:
    return bool(re.search(r"\.(pdf|jpg|jpeg|png|zip|rar|7z)$", u, flags=re.I))

def is_not_bachelor(u: str) -> bool:
    # жёстко отсекаем магистратуру/PhD/minor и т.п.
    bad = [
        "magistr", "master", "phd", "doktor", "minor",
        "магистр", "магист", "докто", "phd", "minor",
        "/magistr", "/phd", "/minor",
        "/магист", "/докто",
    ]
    lu = u.lower()
    return any(b in lu for b in bad)

def looks_like_program_page(u: str) -> bool:
    # эвристика для RU страниц программ бакалавриата
    lu = u.lower()
    if is_media(u) or is_not_bachelor(u):
        return False
    # часто в url есть bakalavr/bachelor/бакалавр
    good = ["bakalavr", "bachelor", "бакалавр", "bakalavriat", "бакалавриат", "bachelor-"]
    if any(g in lu for g in good):
        return True
    # SSBL вариант: /bachelor-...ru/<program>/
    if "/bachelor-" in lu:
        return True
    return False

def looks_like_bachelor_index_anchor(text: str, href: str) -> bool:
    t = norm_key(text)
    h = (href or "").lower()
    if "бакалавриат" in t or "бакалавр" in t:
        return True
    if "бакалавриат" in h or "bakalavriat" in h or "bachelor" in h:
        return True
    return False


# -----------------------------
# URL Builder
# -----------------------------

@dataclass
class ProgramLink:
    faculty: str
    url: str

def fetch(session: requests.Session, url: str, timeout: int, retries: int, sleep: float) -> tuple[int, str]:
    last_exc = None
    for attempt in range(retries + 1):
        try:
            r = session.get(url, timeout=timeout)
            return r.status_code, r.text or ""
        except Exception as e:
            last_exc = e
            if attempt < retries:
                time.sleep(sleep)
    raise RuntimeError(f"Failed to fetch: {url} | last_error={last_exc}")

def collect_anchors(html: str, base_url: str) -> list[tuple[str, str]]:
    soup = BeautifulSoup(html, "lxml")
    out: list[tuple[str, str]] = []
    for a in soup.select("a[href]"):
        href = (a.get("href") or "").strip()
        if not href:
            continue
        full = norm_url(urljoin(base_url, href))
        text = norm(a.get_text(" ", strip=True))
        out.append((full, text))
    return out

def find_bachelor_index_urls(session: requests.Session, faculty_url: str, timeout: int, retries: int, sleep: float) -> list[str]:
    status, html = fetch(session, faculty_url, timeout, retries, sleep)
    if status != 200 or len(html) < 2000:
        return []

    anchors = collect_anchors(html, faculty_url)
    candidates = []
    for href, text in anchors:
        if not is_sdu_ru_url(href) or is_media(href):
            continue
        if looks_like_bachelor_index_anchor(text, href) and not is_not_bachelor(href):
            candidates.append(href)

    # дедуп
    uniq = []
    seen = set()
    for u in candidates:
        if u not in seen:
            seen.add(u)
            uniq.append(u)

    # Если ничего не нашли, fallback: иногда индекс бакалавриата прямо как /ru/.../bachelor-...ru/
    if not uniq:
        for href, _ in anchors:
            lu = href.lower()
            if is_sdu_ru_url(href) and ("/bachelor-" in lu or "bakalavriat" in lu or "бакалавриат" in lu):
                if href not in seen and not is_not_bachelor(href) and not is_media(href):
                    seen.add(href)
                    uniq.append(href)

    return uniq

def collect_program_urls_from_index(session: requests.Session, index_url: str, timeout: int, retries: int, sleep: float) -> set[str]:
    status, html = fetch(session, index_url, timeout, retries, sleep)
    if status != 200 or len(html) < 2000:
        return set()

    anchors = collect_anchors(html, index_url)
    urls = set()
    for href, _ in anchors:
        if not is_sdu_ru_url(href) or is_media(href) or is_not_bachelor(href):
            continue
        if looks_like_program_page(href):
            urls.add(href)
    return urls


def build_all_bachelor_program_urls(
    faculties: dict[str, str],
    timeout: int,
    retries: int,
    sleep: float,
) -> list[ProgramLink]:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (compatible; scrap_sdu/1.0; +https://sdu.edu.kz/)",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "ru,en;q=0.8",
            "Connection": "keep-alive",
        }
    )

    out: list[ProgramLink] = []
    seen = set()

    for faculty, start_url in faculties.items():
        start_url = norm_url(start_url)
        print(f"\n[URLS] faculty={faculty} start={start_url}")

        idx_urls = find_bachelor_index_urls(session, start_url, timeout, retries, sleep)
        if not idx_urls:
            print("  [WARN] bachelor index not found on faculty page, trying to treat start_url as index")
            idx_urls = [start_url]

        all_programs = set()
        for idx_url in idx_urls:
            programs = collect_program_urls_from_index(session, idx_url, timeout, retries, sleep)
            print(f"  index={idx_url} -> programs={len(programs)}")
            all_programs |= programs

        # Если вдруг на индексной странице ссылки “не bakalavr/bachelor”, добавим запасной проход:
        # попробуем собрать все ru-ссылки и оставить те, где встречается "бакалавр" в тексте URL
        # (у тебя это уже покрыто looks_like_program_page, но оставим как safety)
        for u in sorted(all_programs):
            key = (faculty, u)
            if key not in seen:
                seen.add(key)
                out.append(ProgramLink(faculty=faculty, url=u))

    return out


# -----------------------------
# Scraper
# -----------------------------

def scrape_pages(programs: list[ProgramLink], out_jsonl: Path, out_html_dir: Path,
                timeout: int, retries: int, sleep: float) -> None:
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_html_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (compatible; scrap_sdu/1.0; +https://sdu.edu.kz/)",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "ru,en;q=0.8",
            "Connection": "keep-alive",
        }
    )

    ok_cnt = 0
    with out_jsonl.open("w", encoding="utf-8") as w:
        for i, p in enumerate(programs, start=1):
            url = p.url
            slug = safe_slug_from_url(url)
            html_path = out_html_dir / f"{p.faculty}__{slug}.html"

            http_status = 0
            html = ""
            ok = False

            try:
                http_status, html = fetch(session, url, timeout, retries, sleep)
                html_path.write_text(html, encoding="utf-8", errors="ignore")
                ok = (http_status == 200 and len(html) > 5000)
                ok_cnt += int(ok)
            except Exception:
                ok = False

            rec = {
                "key": "bachelors_ru",
                "lang": "ru",
                "faculty": p.faculty,
                "url": url,
                "slug": slug,
                "html_path": str(html_path),
                "http_status": int(http_status),
                "ok": bool(ok),
                "fetched_at": now_iso_utc(),
            }
            w.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"[SCRAPE {i}/{len(programs)}] {p.faculty} | {slug} | status={http_status} | ok={ok}")

    print(f"\nSaved: {out_jsonl} | urls={len(programs)} | ok={ok_cnt}")


# -----------------------------
# Extractor (generic RU)
# -----------------------------

def to_int(x: Optional[str]) -> Optional[int]:
    if not x:
        return None
    m = re.search(r"(\d+)", x)
    return int(m.group(1)) if m else None

CARD_HINTS = {
    "degree": ["степень"],
    "program_length": ["продолжительность"],
    # разруливаем отдельно в коде (грант/платное)
    "threshold_grant": ["пороговый балл", "грант"],
    "threshold_paid": ["пороговый балл", "платн"],
    "ects": ["ects"],
    "english_level": ["требуемый уровень", "английск"],
}

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
        title = norm(t_el.get_text(" ", strip=True)) if t_el else ""
        desc = norm(d_el.get_text(" ", strip=True)) if d_el else ""
        if title and desc:
            pairs.append((title, desc))

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

            if field == "threshold_grant":
                if ("пороговый балл" in t) and ("грант" in t):
                    return desc
            if field == "threshold_paid":
                if ("пороговый балл" in t) and ("платн" in t):
                    return desc

            if all(h in t for h in hints if h):
                return desc

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

    if not out["english_level"]:
        out["english_level"] = "Не требуется"

    return out

def extract_program_name(soup: BeautifulSoup, fallback: str) -> str:
    h1 = soup.select_one("h1")
    if h1:
        t = norm(h1.get_text(" ", strip=True))
        if t:
            return t
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

def open_html(html_path: Path) -> str:
    if html_path.exists():
        return html_path.read_text(encoding="utf-8", errors="ignore")
    alt = Path.cwd() / html_path
    if alt.exists():
        return alt.read_text(encoding="utf-8", errors="ignore")
    raise FileNotFoundError(f"HTML not found: {html_path}")

def extract_dataset(in_jsonl: Path, out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with in_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            html = open_html(Path(rec["html_path"]))
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
                    "faculty": rec.get("faculty"),
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
    with out_csv.open("w", encoding="utf-8", newline="") as w:
        wr = csv.DictWriter(w, fieldnames=cols)
        wr.writeheader()
        wr.writerows(rows)

    def miss(col: str) -> int:
        return sum(1 for r in rows if r.get(col) in (None, "", "None"))

    print(f"\nSaved: {out_csv} | rows={len(rows)}")
    for c in ["degree","program_length","threshold_grant","threshold_paid","ects","english_level","program_description"]:
        print(f"missing {c}: {miss(c)}")


# -----------------------------
# Main pipeline
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data/bachelors_ru", help="папка, куда всё сложить")
    ap.add_argument("--timeout", type=int, default=30)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--sleep", type=float, default=1.0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)

    faculties = {
        # как ты просил — только RU и только эти 3
        "ssbl": "https://sdu.edu.kz/ru/shkola-socialnyh-nauk-businesa-i-prava/",
        "edu_hum": "https://sdu.edu.kz/ru/%D1%84%D0%B0%D0%BA%D1%83%D0%BB%D1%8C%D1%82%D0%B5%D1%82-%D0%BF%D0%B5%D0%B4%D0%B0%D0%B3%D0%BE%D0%B3%D0%B8%D0%BA%D0%B8-%D0%B8-%D0%B3%D1%83%D0%BC%D0%B0%D0%BD%D0%B8%D1%82%D0%B0%D1%80%D0%BD%D1%8B%D1%85/",
        "eng_sci": "https://sdu.edu.kz/ru/%D0%B8%D0%BD%D0%B6%D0%B5%D0%BD%D0%B5%D1%80%D0%BD%D1%8B%D0%B5-%D0%B8-%D0%B5%D1%81%D1%82%D0%B5%D1%81%D1%82%D0%B2%D0%B5%D0%BD%D0%BD%D1%8B%D0%B5-%D0%BD%D0%B0%D1%83%D0%BA%D0%B8/",
    }

    urls_txt = out_dir / "urls" / "bachelor_program_urls.txt"
    pages_jsonl = out_dir / "raw" / "pages.jsonl"
    html_dir = out_dir / "raw_html" / "html"
    canonical_csv = out_dir / "parsed" / "canonical.csv"

    # 1) build urls
    programs = build_all_bachelor_program_urls(
        faculties=faculties,
        timeout=args.timeout,
        retries=args.retries,
        sleep=args.sleep,
    )
    urls_txt.parent.mkdir(parents=True, exist_ok=True)
    urls_txt.write_text(
        "\n".join([f"{p.faculty}\t{p.url}" for p in programs]) + ("\n" if programs else ""),
        encoding="utf-8",
    )
    print(f"\nSaved: {urls_txt} | programs={len(programs)}")

    # 2) scrape
    scrape_pages(
        programs=programs,
        out_jsonl=pages_jsonl,
        out_html_dir=html_dir,
        timeout=args.timeout,
        retries=args.retries,
        sleep=args.sleep,
    )

    # 3) extract
    extract_dataset(in_jsonl=pages_jsonl, out_csv=canonical_csv)


if __name__ == "__main__":
    main()