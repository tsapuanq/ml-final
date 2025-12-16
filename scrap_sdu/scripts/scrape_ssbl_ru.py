#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#scrap_sdu/scripts/scrape_ssbl_ru.py
import argparse
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from urllib.parse import urlsplit

import requests


def now_iso_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_urls(path: Path) -> list[str]:
    urls: list[str] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        urls.append(s)
    return urls


def safe_slug_from_url(url: str) -> str:
    """
    Берем последний сегмент path и делаем "файлово-безопасный" slug.
    Если сегмент пустой/сломанный — fallback на md5(url).
    """
    p = urlsplit(url)
    path = (p.path or "").strip("/")
    last = path.split("/")[-1] if path else ""
    last = last.strip()
    last = last or "index"

    # превращаем в безопасное имя файла (оставим латиницу/цифры/_/-)
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", last)
    slug = re.sub(r"-{2,}", "-", slug).strip("-_")

    if not slug:
        slug = hashlib.md5(url.encode("utf-8")).hexdigest()[:16]
    return slug


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
    raise RuntimeError(f"Failed to fetch after retries: {url} | last_error={last_exc}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_urls", required=True, help="txt со ссылками (RU), по 1 url на строку")
    ap.add_argument("--out_jsonl", required=True, help="куда сохранить pages.jsonl")
    ap.add_argument("--out_html_dir", required=True, help="папка для html файлов")
    ap.add_argument("--timeout", type=int, default=30)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--sleep", type=float, default=1.0)
    args = ap.parse_args()

    in_urls = Path(args.in_urls)
    out_jsonl = Path(args.out_jsonl)
    out_html_dir = Path(args.out_html_dir)

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_html_dir.mkdir(parents=True, exist_ok=True)

    urls = read_urls(in_urls)
    if not urls:
        raise SystemExit(f"No URLs found in {in_urls}")

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
        for idx, url in enumerate(urls, start=1):
            slug = safe_slug_from_url(url)
            html_path = out_html_dir / f"{slug}.html"

            http_status = None
            html = ""
            ok = False

            try:
                http_status, html = fetch(
                    session=session,
                    url=url,
                    timeout=args.timeout,
                    retries=args.retries,
                    sleep=args.sleep,
                )
                html_path.write_text(html, encoding="utf-8", errors="ignore")
                ok = (http_status == 200 and len(html) > 5000)
                ok_cnt += int(ok)
            except Exception as e:
                # сохраняем хотя бы мета-строку в jsonl, чтобы видеть что упало
                html = ""
                ok = False
                http_status = http_status or 0

            rec = {
                "key": "ssbl_ru",
                "lang": "ru",
                "url": url,
                "slug": slug,
                "html_path": str(html_path),
                "http_status": int(http_status) if http_status is not None else None,
                "ok": bool(ok),
                "fetched_at": now_iso_utc(),
            }
            w.write(json.dumps(rec, ensure_ascii=False) + "\n")

            print(f"[{idx}/{len(urls)}] {slug} | status={http_status} | ok={ok}")

    print(f"\nSaved: {out_jsonl} | urls={len(urls)} | ok={ok_cnt}")


if __name__ == "__main__":
    main()