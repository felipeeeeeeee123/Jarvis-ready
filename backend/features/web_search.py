"""Lightweight DuckDuckGo search utility."""

from __future__ import annotations

import logging
from typing import List

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)
session = requests.Session()

def web_search(query: str) -> str:
    """Return the top search results for a query."""

    url = f"https://duckduckgo.com/html/?q={query}"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        res = session.get(url, headers=headers, timeout=10)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "html.parser")
        results: List[str] = [r.get_text() for r in soup.find_all("a", class_="result__a", limit=3)]
        return "\n".join(results) or "No results found."
    except Exception as exc:
        logger.error("Web search failed: %s", exc)
        return f"[Web search error: {exc}]"
