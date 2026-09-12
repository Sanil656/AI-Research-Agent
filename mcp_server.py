"""
Model Context Protocol (MCP) Server for Research AI Agent.
Exposes standard MCP tools for Web Searching, Visual Generation, and Web Content Fetching.
Can be used internally by LangGraph or externally by Claude Desktop / Cursor.
"""

import json
from typing import List, Dict, Any
from mcp.server.fastmcp import FastMCP

from tools import search_web as _core_search_web, get_free_image_url as _core_get_image_url

# Initialize FastMCP Server
mcp = FastMCP("ResearchToolsServer")


@mcp.tool()
def search_web(query: str, max_results: int = 5) -> str:
    """
    Search the live web for recent news, technical documentation, and articles.
    Args:
        query: The search query string.
        max_results: Maximum number of results to return (default: 5).
    Returns:
        JSON string containing list of results with title, url, and content.
    """
    results = _core_search_web(query=query, max_results=max_results)
    return json.dumps(results)


@mcp.tool()
def generate_image(prompt: str, width: int = 1024, height: int = 640) -> str:
    """
    Generate a 100% free AI concept image / technical diagram using Pollinations Flux.
    Args:
        prompt: Descriptive text prompt for the image.
        width: Image width in pixels (default: 1024).
        height: Image height in pixels (default: 640).
    Returns:
        Direct HTTPS URL of the generated image.
    """
    return _core_get_image_url(prompt=prompt, width=width, height=height)


@mcp.tool()
def fetch_page(url: str) -> str:
    """
    Fetch and extract clean text content from a given webpage URL.
    Args:
        url: The webpage URL to fetch.
    Returns:
        Clean text content of the webpage.
    """
    try:
        import requests
        from bs4 import BeautifulSoup

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        }
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()

        soup = BeautifulSoup(r.text, "html.parser")
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()

        text = soup.get_text(separator=" ", strip=True)
        # Truncate to reasonable length for LLM context
        return text[:4000]
    except Exception as e:
        return f"Error fetching webpage {url}: {str(e)}"


if __name__ == "__main__":
    # Run standalone stdio server for MCP clients
    mcp.run(transport="stdio")
