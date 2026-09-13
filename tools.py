"""
Tools & External Service Connectors for the Research AI Agent.

Provides:
1. Dual Web Search (DuckDuckGo Live News + General Web, with Tavily fallback).
2. Academic Paper Search (arXiv Open API - 100% free, peer-reviewed scientific papers).
3. AI Visual Concept Generator (Pollinations Flux.1 - 100% free, zero-key image rendering).
"""

import os
import random
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import List, Dict, Any


# =====================================================================
# 1. Real-Time Web Search (DuckDuckGo News + Text / Tavily)
# =====================================================================

def search_web(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """
    Performs real-time web search across breaking news and general web sources.
    
    Args:
        query: The search term or investigative question.
        max_results: Maximum number of search snippets to retrieve (default: 5).

    Returns:
        A list of dictionaries with keys: 'title', 'url', 'content'.
    """
    # 1. Prefer Tavily if an API key is provided
    tavily_key = os.getenv("TAVILY_API_KEY", "").strip()
    if tavily_key:
        try:
            from tavily import TavilyClient
            client = TavilyClient(api_key=tavily_key)
            response = client.search(query=query, max_results=max_results)
            results = [
                {
                    "title": item.get("title", "Untitled Source"),
                    "url": item.get("url", ""),
                    "content": item.get("content", "")
                }
                for item in response.get("results", [])
            ]
            if results:
                return results
        except Exception as e:
            print(f"[Search Notice] Tavily search fallback triggered ({e}). Using DuckDuckGo.")

    # 2. Dual-backend DuckDuckGo search (News + General Web)
    results: List[Dict[str, str]] = []
    seen_urls = set()

    try:
        from ddgs import DDGS
    except ImportError:
        from duckduckgo_search import DDGS

    try:
        with DDGS() as ddgs:
            # Query live breaking news
            try:
                news_items = list(ddgs.news(query, max_results=max_results))
                for item in news_items:
                    url = (item.get("url") or item.get("link", "")).strip()
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        results.append({
                            "title": item.get("title", "Live News Source"),
                            "url": url,
                            "content": f"[Date: {item.get('date', 'Recent')}] " + item.get("body", item.get("snippet", ""))
                        })
            except Exception:
                pass  # Continue to general text search if news query yields no results

            # Query general web index
            try:
                raw_results = list(ddgs.text(query, max_results=max_results))
                for item in raw_results:
                    url = (item.get("href") or item.get("link", "")).strip()
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        results.append({
                            "title": item.get("title", "Web Source"),
                            "url": url,
                            "content": item.get("body", item.get("snippet", ""))
                        })
            except Exception:
                pass

    except Exception as e:
        print(f"[Search Warning] Search query '{query}' encountered an error: {e}")

    return results


# =====================================================================
# 2. Free Academic & Scientific Paper Search (arXiv Open API)
# =====================================================================

def search_academic_arxiv(query: str, max_results: int = 3) -> List[Dict[str, str]]:
    """
    Searches peer-reviewed academic papers on arXiv for deep scientific/technical queries.
    Completely free, no API key required.

    Args:
        query: Academic topic or technical concept to search.
        max_results: Maximum number of papers to retrieve.

    Returns:
        List of paper dictionaries with title, authors, summary, and arXiv URL.
    """
    papers: List[Dict[str, str]] = []
    try:
        encoded_query = urllib.parse.quote(query)
        url = f"http://export.arxiv.org/api/query?search_query=all:{encoded_query}&start=0&max_results={max_results}"
        
        req = urllib.request.Request(url, headers={"User-Agent": "ResearchAgent/1.0"})
        with urllib.request.urlopen(req, timeout=10) as response:
            xml_data = response.read()

        root = ET.fromstring(xml_data)
        namespace = {"atom": "http://www.w3.org/2005/Atom"}

        for entry in root.findall("atom:entry", namespace):
            title_elem = entry.find("atom:title", namespace)
            summary_elem = entry.find("atom:summary", namespace)
            id_elem = entry.find("atom:id", namespace)
            published_elem = entry.find("atom:published", namespace)

            title = title_elem.text.strip().replace("\n", " ") if title_elem is not None and title_elem.text else "Academic Paper"
            summary = summary_elem.text.strip().replace("\n", " ") if summary_elem is not None and summary_elem.text else ""
            paper_url = id_elem.text.strip() if id_elem is not None and id_elem.text else ""
            pub_date = published_elem.text[:10] if published_elem is not None and published_elem.text else "Recent"

            papers.append({
                "title": f"📄 [arXiv] {title}",
                "url": paper_url,
                "content": f"[Published: {pub_date}] {summary[:600]}..."
            })
    except Exception as e:
        print(f"[arXiv Search Warning] Could not fetch papers for '{query}': {e}")

    return papers


# =====================================================================
# 3. Free Visual Concept Generator (Pollinations Flux.1)
# =====================================================================

def get_free_image_url(prompt: str, width: int = 1024, height: int = 640, model: str = "flux") -> str:
    """
    Generates a high-resolution concept visual URL using Pollinations Flux.1.
    Requires ZERO API keys, accounts, or credits.

    Args:
        prompt: Descriptive scene prompt for image generation.
        width: Image resolution width in pixels (default: 1024).
        height: Image resolution height in pixels (default: 640).
        model: Underlying diffusion model (default: 'flux').

    Returns:
        Direct HTTPS image URL.
    """
    clean_prompt = prompt.strip()
    encoded_prompt = urllib.parse.quote(clean_prompt)
    seed = random.randint(1000, 999999)
    return f"https://image.pollinations.ai/prompt/{encoded_prompt}?width={width}&height={height}&nologo=true&model={model}&seed={seed}"
