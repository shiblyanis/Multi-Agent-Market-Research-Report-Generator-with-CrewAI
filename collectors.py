"""Web search and scraping helpers (plain functions, wrapped as tools in tools.py)."""
import ipaddress
import os
import re
import socket
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from crew import config

USER_AGENT = "Mozilla/5.0 (compatible; MarketResearchCrew/1.0)"
MAX_RESULTS = 4
MAX_PAGE_BYTES = 500_000
MAX_CHARS_PER_PAGE = 3000
TIMEOUT = 12


def is_public_url(url: str) -> bool:
    """Only http(s) URLs that resolve to public IPs are scraped (basic SSRF guard)."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return False
    try:
        infos = socket.getaddrinfo(parsed.hostname, None)
    except socket.gaierror:
        return False
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            return False
    return True


def search(query: str, max_results: int = MAX_RESULTS) -> list:
    """Return [{'title','url','snippet'}] using Tavily or SerperDev."""
    provider = config.search_provider()
    if provider == "tavily":
        resp = requests.post(
            "https://api.tavily.com/search",
            headers={"Authorization": f"Bearer {os.environ['TAVILY_API_KEY']}"},
            json={"query": query, "max_results": max_results, "search_depth": "basic"},
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        return [{"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("content", "")}
                for r in resp.json().get("results", [])]
    if provider == "serper":
        resp = requests.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": os.environ["SERPER_API_KEY"], "Content-Type": "application/json"},
            json={"q": query, "num": max_results},
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        return [{"title": r.get("title", ""), "url": r.get("link", ""), "snippet": r.get("snippet", "")}
                for r in resp.json().get("organic", [])[:max_results]]
    raise RuntimeError("No search provider configured (set TAVILY_API_KEY or SERPER_API_KEY)")


def scrape(url: str, max_chars: int = MAX_CHARS_PER_PAGE) -> str:
    """Fetch a page and return cleaned visible text (empty string on any failure)."""
    if not is_public_url(url):
        return ""
    try:
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT, stream=True)
        resp.raise_for_status()
        if "html" not in resp.headers.get("Content-Type", "html").lower():
            return ""
        raw = next(resp.iter_content(MAX_PAGE_BYTES), b"")
        html = raw.decode(resp.encoding or "utf-8", errors="replace")
    except requests.RequestException:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "svg"]):
        tag.decompose()
    text = re.sub(r"\s+", " ", soup.get_text(" ")).strip()
    return text[:max_chars]


def search_and_scrape(query: str) -> str:
    results = search(query)
    if not results:
        return f"No search results found for '{query}'."
    blocks = []
    for i, r in enumerate(results, 1):
        body = scrape(r["url"]) or r["snippet"]
        blocks.append(f"[{i}] {r['title']}\nSource: {r['url']}\n{body}")
    return "\n\n".join(blocks)
