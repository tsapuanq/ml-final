import re
import time
import json
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple
from urllib.parse import urljoin, urlparse

import requests
import pandas as pd
from bs4 import BeautifulSoup


HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; SDU-Scraper/1.0; +https://example.com/bot)"
}

# Твои ссылки (как прислал)
SEEDS = {
    "BSLAW": {
        "en": "https://sdu.edu.kz/en/school-of-social-sciences-business-and-law/bachelor-bslaw/",
        "kz": "https://sdu.edu.kz/aleumettik-gylymdar-business-zhane-kukyk-mektebi/bachelor-bslawkz/",
        "ru": "https://sdu.edu.kz/ru/shkola-socialnyh-nauk-businesa-i-prava/bachelor-bslawru/",
    },
    "EDU": {
        "en": "https://sdu.edu.kz/en/education-and-humanities/programms/bachelor/",
        "ru": "https://sdu.edu.kz/ru/%d1%84%d0%b0%d0%ba%d1%83%d0%bb%d1%8c%d1%82%d0%b5%d1%82-%d0%bf%d0%b5%d0%b4%d0%b0%d0%b3%d0%be%d0%b3%d0%b8%d0%ba%d0%b8-%d0%b8-%d0%b3%d1%83%d0%bc%d0%b0%d0%bd%d0%b8%d1%82%d0%b0%d1%80%d0%bd%d1%8b%d1%85/%d0%bf%d1%80%d0%be%d0%b3%d1%80%d0%b0%d0%bc%d0%bc%d1%8b/%d0%b1%d0%b0%d0%ba%d0%b0%d0%bb%d0%b0%d0%b2%d1%80%d0%b8%d0%b0%d1%82/",
        "kz": "https://sdu.edu.kz/%d0%bf%d0%b5%d0%b4%d0%b0%d0%b3%d0%be%d0%b3%d0%b8%d0%ba%d0%b0%d0%bb%d1%8b%d2%9b-%d0%b6%d3%99%d0%bd%d0%b5-%d0%b3%d1%83%d0%bc%d0%b0%d0%bd%d0%b8%d1%82%d0%b0%d1%80%d0%bb%d1%8b%d2%9b-%d2%93%d1%8b%d0%bb/%d0%b1%d0%b0%d2%93%d0%b4%d0%b0%d1%80%d0%bb%d0%b0%d0%bc%d0%b0%d0%bb%d0%b0%d1%80/%d0%b1%d0%b0%d0%ba%d0%b0%d0%bb%d0%b0%d0%b2%d1%80%d0%b8%d0%b0%d1%82/",
    },
    "SITAM": {
        "en": "https://sdu.edu.kz/en/engineering-and-natural-sciences/programms/bachelor/",
        "ru": "https://sdu.edu.kz/ru/%d0%b8%d0%bd%d0%b6%d0%b5%d0%bd%d0%b5%d1%80%d0%bd%d1%8b%d0%b5-%d0%b8-%d0%b5%d1%81%d1%82%d0%b5%d1%81%d1%82%d0%b2%d0%b5%d0%bd%d0%bd%d1%8b%d0%b5-%d0%bd%d0%b0%d1%83%d0%ba%d0%b8/%d0%bf%d1%80%d0%be%d0%b3%d1%80%d0%b0%d0%bc%d0%bc%d1%8b/%d0%b1%d0%b0%d0%ba%d0%b0%d0%bb%d0%b0%d0%b2%d1%80%d0%b8%d0%b0%d1%82/",
        "kz": "https://sdu.edu.kz/%d0%b8%d0%bd%d0%b6%d0%b5%d0%bd%d0%b5%d1%80%d0%bb%d1%96%d0%ba-%d0%b6%d3%99%d0%bd%d0%b5-%d0%b6%d0%b0%d1%80%d0%b0%d1%82%d1%8b%d0%bb%d1%8b%d1%81%d1%82%d0%b0%d0%bd%d1%83-%d2%93%d1%8b%d0%bb%d1%8b/%d0%bf%d1%80%d0%be%d0%b3%d1%80%d0%b0%d0%bc%d0%bc%d1%8b/%d0%b1%d0%b0%d0%ba%d0%b0%d0%bb%d0%b0%d0%b2%d1%80%d0%b8%d0%b0%d1%82/",
    },
}

# Маппинг заголовка аккордеона "Program description" на 3 языках
PROGRAM_DESC_TITLES = [
    re.compile(r"\bProgram description\b", re.I),
    re.compile(r"\bОписание программы\b", re.I),
    re.compile(r"\bБағдарлама сипаттамасы\b", re.I),
]

# Карточки слева: label->поле (регексы на 3 языка)
LEFT_CARD_LABELS = [
    (re.compile(r"^\s*Degree\s*$", re.I), "degree"),
    (re.compile(r"^\s*Присваиваемая степень\s*$", re.I), "degree"),
    (re.compile(r"^\s*Дәреже\s*$", re.I), "degree"),

    (re.compile(r"^\s*Program length\s*$", re.I), "program_length"),
    (re.compile(r"^\s*Срок обучения\s*$", re.I), "program_length"),
    (re.compile(r"^\s*Оқу мерзімі\s*$", re.I), "program_length"),

    (re.compile(r"^\s*Threshold score for state scholarship\s*$", re.I), "threshold_state"),
    (re.compile(r"^\s*Проходной балл на гос\.? грант\s*$", re.I), "threshold_state"),
    (re.compile(r"^\s*Мемлекеттік грантқа өту балы\s*$", re.I), "threshold_state"),

    (re.compile(r"^\s*Threshold score for paid department\s*$", re.I), "threshold_paid"),
    (re.compile(r"^\s*Проходной балл на платное отделение\s*$", re.I), "threshold_paid"),
    (re.compile(r"^\s*Ақылы бөлімге өту балы\s*$", re.I), "threshold_paid"),

    (re.compile(r"^\s*ECTS\s*$", re.I), "ects"),
    (re.compile(r"^\s*Уровень английского\s*$", re.I), "english_level"),
    (re.compile(r"^\s*Level of English\s*$", re.I), "english_level"),
    (re.compile(r"^\s*Ағылшын тілі деңгейі\s*$", re.I), "english_level"),
]


STOP_SCHEMES = {"mailto", "tel"}
STOP_HOSTS = {"drive.google.com", "t.me", "www.instagram.com", "www.facebook.com", "www.tiktok.com"}


@dataclass
class Page:
    url: str
    soup: BeautifulSoup


def get_soup(url: str, session: requests.Session) -> Page:
    r = session.get(url, headers=HEADERS, timeout=40)
    r.raise_for_status()
    return Page(url=url, soup=BeautifulSoup(r.text, "lxml"))


def clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def normalize_url(base: str, href: str) -> str:
    return urljoin(base, href.strip())


def detect_lang(soup: BeautifulSoup, url: str) -> str:
    html = soup.find("html")
    if html and html.get("lang"):
        lang = html["lang"].lower()
        if lang.startswith("ru"):
            return "ru"
        if lang.startswith(("kk", "kz")):
            return "kz"
        if lang.startswith("en"):
            return "en"

    p = urlparse(url).path.lower()
    if "/ru/" in p:
        return "ru"
    if "/kz/" in p or "/kk/" in p:
        return "kz"
    # kz-страницы часто без /kz/ в пути, поэтому только фолбэк на en
    return "en"


def is_sdu_url(u: str) -> bool:
    try:
        parsed = urlparse(u)
        if parsed.scheme in STOP_SCHEMES:
            return False
        if parsed.netloc and parsed.netloc != "sdu.edu.kz":
            return False
        if parsed.netloc in STOP_HOSTS:
            return False
        return True
    except Exception:
        return False


def infer_listing_from_seed(seed_url: str) -> str:
    """
    Если seed — это страница конкретной программы, пытаемся подняться на уровень выше до папки bachelor-*.
    """
    path = urlparse(seed_url).path.rstrip("/") + "/"
    # типовые маркеры для BSLAW
    for marker in ["/bachelor-bslaw/", "/bachelor-bslawru/", "/bachelor-bslawkz/"]:
        if marker in path:
            prefix = path.split(marker)[0] + marker
            return f"https://sdu.edu.kz{prefix}"
    # если это уже листинг — просто вернём как есть
    return seed_url.rstrip("/") + "/"


def extract_program_links_from_listing(listing_url: str, page: Page) -> List[str]:
    base = listing_url.rstrip("/") + "/"
    out: Set[str] = set()

    for a in page.soup.select("a[href]"):
        href = a.get("href", "")
        if not href:
            continue
        u = normalize_url(page.url, href)
        if not is_sdu_url(u):
            continue
        u = u.split("#")[0]
        # главное правило: программа обычно "под" листингом
        if u.startswith(base) and u.rstrip("/") != base.rstrip("/"):
            out.add(u)

    return sorted(out)


def extract_language_versions(page: Page) -> Dict[str, str]:
    versions: Dict[str, str] = {}
    for a in page.soup.select("a[href]"):
        txt = clean_text(a.get_text()).lower()
        if txt in {"en", "ru", "kz"}:
            versions[txt] = normalize_url(page.url, a["href"])
    # фолбэк — хотя бы текущая страница
    if not versions:
        versions[detect_lang(page.soup, page.url)] = page.url
    return versions


def extract_program_name(page: Page) -> str:
    h1 = page.soup.find("h1")
    if h1:
        return clean_text(h1.get_text())
    if page.soup.title:
        return clean_text(page.soup.title.get_text())
    return ""


def extract_left_cards(page: Page) -> Dict[str, str]:
    data: Dict[str, str] = {}

    # 1) типичный Elementor icon-box: title + description
    for w in page.soup.select(".elementor-icon-box-wrapper, .elementor-icon-box-content"):
        t = w.select_one(".elementor-icon-box-title")
        d = w.select_one(".elementor-icon-box-description")
        if not t or not d:
            continue
        label = clean_text(t.get_text())
        value = clean_text(d.get_text(" "))
        if label and value:
            data[label] = value

    # 2) нормализуем label -> canonical field
    normalized: Dict[str, str] = {}
    for label, value in data.items():
        for rx, field in LEFT_CARD_LABELS:
            if rx.match(label):
                normalized[field] = value
                break

    # 3) если не нашли через icon-box, попробуем искать по тексту label в документе
    if not normalized:
        for rx, field in LEFT_CARD_LABELS:
            node = page.soup.find(string=rx)
            if not node:
                continue
            # берём ближайший контейнер и весь его текст
            container = node.find_parent(["div", "section", "article"]) or node.parent
            txt = clean_text(container.get_text(" "))
            # вырежем label и оставим хвост как value
            val = re.sub(rx, "", txt).strip(" :-–—")
            if val:
                normalized[field] = val

    return normalized


def extract_program_description(page: Page) -> str:
    # 1) accordion item (Elementor): tab-title + tab-content
    for item in page.soup.select(".elementor-accordion-item"):
        title = item.select_one(".elementor-tab-title")
        content = item.select_one(".elementor-tab-content")
        if not title or not content:
            continue
        t = clean_text(title.get_text())
        if any(rx.search(t) for rx in PROGRAM_DESC_TITLES):
            return clean_text(content.get_text(" "))

    # 2) фолбэк: найдём заголовок “Program description” текстом и возьмём ближайший блок ниже
    for rx in PROGRAM_DESC_TITLES:
        node = page.soup.find(string=rx)
        if node:
            parent = node.find_parent(["div", "section", "article"]) or node.parent
            # пробуем взять следующий соседний контент
            nxt = parent.find_next(["div", "section"])
            if nxt:
                return clean_text(nxt.get_text(" "))

    return ""


def make_group_key(urls: Dict[str, str]) -> str:
    # стабильный ключ для склейки языков одной программы
    joined = "|".join(f"{k}:{urls[k]}" for k in sorted(urls.keys()))
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:12]


def crawl_faculty(faculty: str, seed_by_lang: Dict[str, str], sleep_s: float = 0.4) -> List[Dict]:
    session = requests.Session()

    # 1) приведём seeds к листингам (особенно BSLAW)
    listings = {lang: infer_listing_from_seed(u) for lang, u in seed_by_lang.items()}

    # 2) соберём первичные ссылки на программы с листингов
    to_visit: List[str] = []
    seen: Set[str] = set()

    for lang, listing_url in listings.items():
        try:
            time.sleep(sleep_s)
            page = get_soup(listing_url, session)
            links = extract_program_links_from_listing(listing_url, page)
            to_visit.extend(links)
        except Exception as e:
            print(f"[WARN] listing failed {faculty}/{lang}: {listing_url} -> {e}")

    # 3) BFS: для каждой программы вытаскиваем языковые версии и добавляем их в очередь
    rows: List[Dict] = []
    while to_visit:
        url = to_visit.pop(0)
        url = url.split("#")[0]
        if url in seen:
            continue
        seen.add(url)

        try:
            time.sleep(sleep_s)
            page = get_soup(url, session)

            versions = extract_language_versions(page)  # en/ru/kz ссылки со страницы
            group_key = make_group_key(versions)

            # добавим найденные версии в очередь тоже
            for v_url in versions.values():
                v_url = v_url.split("#")[0]
                if v_url not in seen:
                    to_visit.append(v_url)

            lang = detect_lang(page.soup, page.url)
            row = {
                "faculty": faculty,
                "group_key": group_key,
                "lang": lang,
                "program_name": extract_program_name(page),
                "url": page.url,
                "degree": "",
                "program_length": "",
                "threshold_state": "",
                "threshold_paid": "",
                "ects": "",
                "english_level": "",
                "program_description": "",
            }

            left = extract_left_cards(page)
            for k in ["degree", "program_length", "threshold_state", "threshold_paid", "ects", "english_level"]:
                if k in left:
                    row[k] = left[k]

            row["program_description"] = extract_program_description(page)
            rows.append(row)

        except Exception as e:
            print(f"[WARN] program failed {url}: {e}")

    return rows


def main():
    all_rows: List[Dict] = []
    for faculty, seed_by_lang in SEEDS.items():
        print(f"\n=== {faculty} ===")
        all_rows.extend(crawl_faculty(faculty, seed_by_lang))

    # уберём дубликаты (иногда одна и та же страница попадёт через разные пути)
    df = pd.DataFrame(all_rows).drop_duplicates(subset=["url"]).reset_index(drop=True)

    # сохраним и json (на всякий) и csv (для Excel)
    df.to_csv("/Users/sapuantalaspay/vs_projects/introML/final_ML/data/raw/sdu_bachelor_programs.csv", index=False, encoding="utf-8-sig")
    with open("/Users/sapuantalaspay/vs_projects/introML/final_ML/data/raw/sdu_bachelor_programs.json", "w", encoding="utf-8") as f:
        json.dump(all_rows, f, ensure_ascii=False, indent=2)

    print(f"\nSaved: sdu_bachelor_programs.csv  ({len(df)} rows)")
    print("Saved: sdu_bachelor_programs.json")


if __name__ == "__main__":
    main()