from __future__ import annotations
from typing import Tuple
from urllib.parse import urlparse
from bs4 import BeautifulSoup

from .chunker import norm_text

def extract_main_text(html: str, url: str) -> Tuple[str, str]:
    soup = BeautifulSoup(html, "lxml")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    for tag in soup.find_all(["nav", "footer", "aside"]):
        tag.decompose()

    h1 = soup.find("h1")
    title = norm_text(h1.get_text(" ", strip=True)) if h1 else urlparse(url).path.rsplit("/", 1)[-1]

    candidates = []
    for sel in ["article", "div.content", "div#content", "div.node-content", "div.main", "div.container"]:
        el = soup.select_one(sel)
        if el:
            candidates.append(el)

    main = max(
        candidates,
        key=lambda x: len(x.get_text(" ", strip=True)),
        default=soup.body or soup,
    )

    text = norm_text(main.get_text("\n", strip=True))

    # грубая фильтрация строк
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if len(line) < 3:
            continue
        lines.append(line)
    text = norm_text("\n".join(lines))

    return title, text
