"""Web and media tools for the Research AI Agent (DDG News+Text, arXiv, Pollinations Flux)."""

import os
import random
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import List, Dict


def search_web(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """Searches live news and web data (Tavily or DuckDuckGo)."""
    tavily_key = os.getenv("TAVILY_API_KEY", "").strip()
    if tavily_key:
        try:
            from tavily import TavilyClient
            res = TavilyClient(api_key=tavily_key).search(query=query, max_results=max_results)
            return [{"title": r.get("title", "Source"), "url": r.get("url", ""), "content": r.get("content", "")} for r in res.get("results", [])]
        except Exception:
            pass

    results, seen_urls = [], set()
    try:
        from ddgs import DDGS
    except ImportError:
        from duckduckgo_search import DDGS

    try:
        with DDGS() as ddgs:
            for item in list(ddgs.news(query, max_results=max_results)) + list(ddgs.text(query, max_results=max_results)):
                url = (item.get("url") or item.get("href") or item.get("link", "")).strip()
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    date_tag = f"[Date: {item.get('date')}] " if item.get("date") else ""
                    results.append({"title": item.get("title", "Web Source"), "url": url, "content": date_tag + item.get("body", item.get("snippet", ""))})
    except Exception as e:
        print(f"[Search Warning] {query}: {e}")
    return results


def search_academic_arxiv(query: str, max_results: int = 3) -> List[Dict[str, str]]:
    """Searches peer-reviewed academic papers on arXiv (100% free)."""
    papers = []
    try:
        url = f"http://export.arxiv.org/api/query?search_query=all:{urllib.parse.quote(query)}&start=0&max_results={max_results}"
        req = urllib.request.Request(url, headers={"User-Agent": "ResearchAgent/1.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            root = ET.fromstring(resp.read())
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        for entry in root.findall("atom:entry", ns):
            title = (entry.findtext("atom:title", default="", namespaces=ns)).strip().replace("\n", " ")
            summary = (entry.findtext("atom:summary", default="", namespaces=ns)).strip().replace("\n", " ")
            paper_url = (entry.findtext("atom:id", default="", namespaces=ns)).strip()
            date = (entry.findtext("atom:published", default="", namespaces=ns))[:10]
            papers.append({"title": f"📄 [arXiv] {title}", "url": paper_url, "content": f"[Date: {date}] {summary[:500]}..."})
    except Exception as e:
        print(f"[arXiv Warning] {query}: {e}")
    return papers


def get_free_image_url(prompt: str, width: int = 1024, height: int = 640, model: str = "flux") -> str:
    """Generates a free concept visual URL via Pollinations Flux.1."""
    encoded = urllib.parse.quote(prompt.strip())
    return f"https://image.pollinations.ai/prompt/{encoded}?width={width}&height={height}&nologo=true&model={model}&seed={random.randint(1000, 999999)}"
