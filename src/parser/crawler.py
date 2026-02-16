# src/parser/sniprf_crawler.py
from __future__ import annotations

import os
import re
import json
import time
import hashlib
from dataclasses import dataclass, asdict
from typing import List, Set, Tuple
from urllib.parse import urljoin, urlparse
from collections import deque

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from .html_extractors import extract_main_text
from .chunker import chunk_text

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
BASE_URL = os.environ.get("BASE_URL", "http://sniprf.ru/snip")
OUT_DIR = os.environ.get("OUT_DIR", "/data/knowledge_base")

SNIPPETS_DIR = os.path.join(OUT_DIR, "snippets")
META_PATH = os.path.join(OUT_DIR, "snippets_meta.jsonl")

# Seeds: разделы, по которым нужно пройтись (razdel-*)
# Пример:
# SEED_SECTIONS="http://sniprf.ru/razdel-1,http://sniprf.ru/razdel-2,..."
SEED_SECTIONS = os.environ.get("SEED_SECTIONS", "").strip()

# Crawl limits
MAX_BFS_DEPTH = int(os.environ.get("MAX_BFS_DEPTH", "4"))
MAX_CRAWL_PAGES = int(os.environ.get("MAX_CRAWL_PAGES", "2000"))  # общий лимит посещённых страниц
MAX_DOC_PAGES = int(os.environ.get("MAX_DOC_PAGES", "400"))       # лимит найденных "документ-страниц"

SLEEP_SEC = float(os.environ.get("SLEEP_SEC", "0.4"))

# Chunking
CHUNK_CHARS = int(os.environ.get("CHUNK_CHARS", "1600"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "250"))

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; rag-crawler/1.0; +local)"}


# -------------------------------------------------------------------
# DATA
# -------------------------------------------------------------------
@dataclass
class Snippet:
    id: str
    source_url: str
    title: str
    text: str
    chunk_no: int


# -------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------
def ensure_dirs() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(SNIPPETS_DIR, exist_ok=True)


def reset_output() -> None:
    # пересборка сниппетов с нуля
    if os.path.exists(META_PATH):
        os.remove(META_PATH)

    if os.path.isdir(SNIPPETS_DIR):
        for fn in os.listdir(SNIPPETS_DIR):
            if fn.endswith(".txt"):
                try:
                    os.remove(os.path.join(SNIPPETS_DIR, fn))
                except OSError:
                    pass


def sha1_id(*parts: str) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def canonical(u: str) -> str:
    p = urlparse(u)
    return p._replace(fragment="", query="").geturl()


def same_host(u: str, base: str) -> bool:
    return urlparse(u).netloc == urlparse(base).netloc


def is_suspect_link(u: str) -> bool:
    path = urlparse(u).path.lower()
    if any(x in path for x in ["/search", "/login", "/register", "/user", "/tag", "/rss"]):
        return True
    return False


def fetch(session: requests.Session, url: str) -> str:
    r = session.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    r.encoding = r.apparent_encoding or "utf-8"
    return r.text


def extract_links(html: str, base_url: str) -> List[str]:
    soup = BeautifulSoup(html, "lxml")
    out: List[str] = []

    for a in soup.find_all("a", href=True):
        href = (a["href"] or "").strip()
        if not href:
            continue
        full = canonical(urljoin(base_url, href))
        if not same_host(full, base_url):
            continue
        if is_suspect_link(full):
            continue
        out.append(full)

    # uniq preserve order
    seen = set()
    uniq: List[str] = []
    for u in out:
        if u in seen:
            continue
        seen.add(u)
        uniq.append(u)
    return uniq


def parse_seed_sections() -> List[str]:
    if not SEED_SECTIONS:
        return [canonical(BASE_URL)]
    parts = [p.strip() for p in SEED_SECTIONS.split(",") if p.strip()]
    return [canonical(p) for p in parts]


def is_probably_doc_page(title: str, text: str) -> bool:
    """
    Документ-страница с нормами:
    - достаточно длинный текст
    - или есть нумерация пунктов (1., 1.1., 2.3.4 ...)
    """
    if len(text) < 800:
        return False
    if re.search(r"(^|\n)\s*\d+(\.\d+){0,3}\s", text):
        return True
    return len(text) >= 1500


def should_follow_link(u: str) -> bool:
    """
    Разрешаем обход:
    - /razdel-* (разделы)
    - /glava-*  (главы)
    - любые подстраницы, если они не мусор и на том же домене
    """
    path = urlparse(u).path.lower()
    if path.startswith("/razdel-") or path.startswith("/glava-"):
        return True
    # дополнительно разрешаем любые страницы внутри домена, т.к. структура может быть смешанной
    return True


def discover_doc_pages(session: requests.Session) -> List[str]:
    """
    Обходит заданные разделы (razdel-*) и их главы, собирает doc-pages.
    Логика:
    - BFS с ограничением глубины
    - если страница похожа на документ (текст нормы) -> добавляем в doc_pages и не углубляемся
    - если страница каталог/раздел -> извлекаем ссылки, продолжаем обход
    """
    seeds = parse_seed_sections()

    visited: Set[str] = set()
    doc_pages: Set[str] = set()

    q = deque([(s, 0) for s in seeds])

    while q and len(visited) < MAX_CRAWL_PAGES and len(doc_pages) < MAX_DOC_PAGES:
        url, depth = q.popleft()
        url = canonical(url)

        if url in visited:
            continue
        visited.add(url)

        try:
            html = fetch(session, url)
        except Exception as e:
            print(f"[WARN] fetch failed: {url} :: {e}")
            continue

        # попытаться классифицировать страницу как документ по тексту
        try:
            title, text = extract_main_text(html, url)
        except Exception:
            title, text = "", ""

        if is_probably_doc_page(title, text):
            doc_pages.add(url)
            time.sleep(SLEEP_SEC)
            continue

        # иначе — раздел/глава: собираем ссылки и идём глубже
        if depth < MAX_BFS_DEPTH:
            links = extract_links(html, url)
            for link in links:
                if not same_host(link, seeds[0]):
                    continue
                if is_suspect_link(link):
                    continue
                if not should_follow_link(link):
                    continue
                if link not in visited:
                    q.append((link, depth + 1))

        time.sleep(SLEEP_SEC)

    return sorted(doc_pages)


def append_meta(sn: Snippet) -> None:
    with open(META_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(sn), ensure_ascii=False) + "\n")


def save_snippet(sn: Snippet) -> None:
    with open(os.path.join(SNIPPETS_DIR, f"{sn.id}.txt"), "w", encoding="utf-8") as f:
        f.write(sn.text)


# -------------------------------------------------------------------
# MAIN CRAWL
# -------------------------------------------------------------------
def run_crawl() -> int:
    ensure_dirs()
    reset_output()

    snippets_count = 0

    with requests.Session() as session:
        doc_pages = discover_doc_pages(session)
        print(f"[INFO] discovered doc_pages={len(doc_pages)} (depth={MAX_BFS_DEPTH}, "
              f"max_crawl_pages={MAX_CRAWL_PAGES}, max_docs={MAX_DOC_PAGES})")

        for url in tqdm(doc_pages, desc="Crawl+Parse"):
            try:
                html = fetch(session, url)
                title, text = extract_main_text(html, url)
                if len(text) < 400:
                    continue

                chunks = chunk_text(text, CHUNK_CHARS, CHUNK_OVERLAP)
                for i, ch in enumerate(chunks, start=1):
                    sid = sha1_id(url, str(i), ch[:80])
                    sn = Snippet(
                        id=sid,
                        source_url=url,
                        title=title,
                        text=ch,
                        chunk_no=i,
                    )
                    save_snippet(sn)
                    append_meta(sn)
                    snippets_count += 1

                time.sleep(SLEEP_SEC)

            except Exception as e:
                print(f"[WARN] parse failed: {url} :: {e}")

    if snippets_count == 0:
        raise RuntimeError("Сниппеты не собраны: требуется настройка extract_main_text / эвристик определения документов.")

    print(f"[OK] crawler produced snippets={snippets_count} meta={META_PATH}")
    return snippets_count


if __name__ == "__main__":
    run_crawl()
