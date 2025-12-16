#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#scrap_sdu/scripts/build_ssbl_ru_urls.py

import argparse
import re
from pathlib import Path
from urllib.parse import urljoin, urlsplit, urlunsplit

import requests
from bs4 import BeautifulSoup


def norm_url(u: str) -> str:
    p = urlsplit(u)
    path = (p.path or "").rstrip("/")
    return urlunsplit((p.scheme.lower(), p.netloc.lower(), path, "", ""))


def collect_urls(html: str, base_url: str, contains: str) -> set[str]:
    soup = BeautifulSoup(html, "lxml")
    urls = set()

    for a in soup.select("a[href]"):
        href = (a.get("href") or "").strip()
        if not href:
            continue

        full = norm_url(urljoin(base_url, href))

        if contains and contains not in full:
            continue
        if re.search(r"\.(pdf|jpg|jpeg|png|zip)$", full, flags=re.I):
            continue

        urls.add(full)

    return urls


def fetch(url: str, timeout: int) -> str:
    r = requests.get(
        url,
        timeout=timeout,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; scrap_sdu/1.0; +https://sdu.edu.kz/)",
            "Accept-Language": "ru,en;q=0.8",
        },
    )
    r.raise_for_status()
    return r.text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--start_url",
        default="https://sdu.edu.kz/ru/shkola-socialnyh-nauk-businesa-i-prava/",
    )
    ap.add_argument("--out", default="data/ssbl/urls/program_urls.txt")
    ap.add_argument(
        "--contains",
        default="/ru/shkola-socialnyh-nauk-businesa-i-prava/bachelor-bslawru/",
        help="Фильтр: ссылка должна содержать этот кусок",
    )
    ap.add_argument("--timeout", type=int, default=30)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    html1 = fetch(args.start_url, args.timeout)
    urls = collect_urls(html1, args.start_url, args.contains)

    # Если нашли только индексную страницу — зайдём внутрь и соберём ссылки уже оттуда
    if len(urls) <= 1:
        # попробуем "базовую" индексную страницу
        idx = "https://sdu.edu.kz/ru/shkola-socialnyh-nauk-businesa-i-prava/bachelor-bslawru/"
        html2 = fetch(idx, args.timeout)
        urls2 = collect_urls(html2, idx, args.contains)
        urls |= urls2

    urls_sorted = sorted(urls)
    out_path.write_text("\n".join(urls_sorted) + ("\n" if urls_sorted else ""), encoding="utf-8")

    print(f"Saved: {out_path} | urls={len(urls_sorted)}")
    if urls_sorted[:10]:
        print("Example:")
        for u in urls_sorted[:10]:
            print(" ", u)


if __name__ == "__main__":
    main()