"""
Search and media tools for the Research AI Agent.
Supports DuckDuckGo (text + fresh news), Tavily, and 100% Free AI Image Generation (Pollinations Flux).
"""

import os
import random
import urllib.parse
from typing import List, Dict, Any


def search_web(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """
    Search the web for a given query.
    Prefers Tavily if TAVILY_API_KEY is configured; otherwise uses DuckDuckGo (text + live news).
    """
    tavily_key = os.getenv("TAVILY_API_KEY", "").strip()
    if tavily_key:
        try:
            from tavily import TavilyClient
            client = TavilyClient(api_key=tavily_key)
            response = client.search(query=query, max_results=max_results)
            results = []
            for item in response.get("results", []):
                results.append({
                    "title": item.get("title", "Untitled"),
                    "url": item.get("url", ""),
                    "content": item.get("content", "")
                })
            if results:
                return results
        except Exception as e:
            print(f"[Search Warning] Tavily search failed ({e}). Falling back to DuckDuckGo...")

    # DuckDuckGo fallback with both news and text backends for real-time coverage
    results = []
    seen_urls = set()

    try:
        from ddgs import DDGS
    except ImportError:
        from duckduckgo_search import DDGS

    try:
        with DDGS() as ddgs:
            # 1. Check live news for breaking announcements / recent developments
            try:
                news_items = list(ddgs.news(query, max_results=max_results))
                for item in news_items:
                    url = item.get("url", item.get("link", "")).strip()
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        results.append({
                            "title": item.get("title", "News Source"),
                            "url": url,
                            "content": f"[Date: {item.get('date', 'Recent')}] " + item.get("body", item.get("snippet", ""))
                        })
            except Exception as news_err:
                pass

            # 2. Check general web search
            try:
                raw_results = list(ddgs.text(query, max_results=max_results))
                for item in raw_results:
                    url = item.get("href", item.get("link", "")).strip()
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        results.append({
                            "title": item.get("title", "Web Source"),
                            "url": url,
                            "content": item.get("body", item.get("snippet", ""))
                        })
            except Exception as text_err:
                pass

    except Exception as e:
        print(f"[Search Error] Search failed for query '{query}': {e}")

    return results


def get_free_image_url(prompt: str, width: int = 1024, height: int = 640, model: str = "flux") -> str:
    """
    Generate a 100% free AI image URL using Pollinations Flux engine.
    Requires ZERO API keys, zero signup, and no external dependencies.
    """
    clean_prompt = prompt.strip()
    encoded_prompt = urllib.parse.quote(clean_prompt)
    seed = random.randint(1000, 999999)
    return f"https://image.pollinations.ai/prompt/{encoded_prompt}?width={width}&height={height}&nologo=true&model={model}&seed={seed}"
