"""
Model Context Protocol (MCP) Server for the Research AI Agent.

Exposes standardized FastMCP tools:
1. `search_web`: Real-time web and breaking news search.
2. `search_academic`: Open arXiv peer-reviewed scientific literature search.
3. `generate_image`: 100% free AI concept visual generation via Flux.1.
4. `fetch_page`: Clean web scraper for deep documentation pages.

Can be run standalone (`python mcp_server.py`) for external tools (e.g., Claude Desktop, Cursor)
or invoked programmatically via `mcp_tools.py` in LangGraph.
"""

import json
from mcp.server.fastmcp import FastMCP
from tools import (
    search_web as _core_search_web,
    search_academic_arxiv as _core_search_academic,
    get_free_image_url as _core_get_image_url
)

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
def search_academic(query: str, max_results: int = 3) -> str:
    """
    Search peer-reviewed academic papers and scientific preprints on arXiv.

    Args:
        query: Academic subject or technical research query.
        max_results: Maximum number of papers to retrieve (default: 3).

    Returns:
        JSON string containing list of papers with title, authors, summary, and arXiv URL.
    """
    results = _core_search_academic(query=query, max_results=max_results)
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
        Clean text content of the webpage (up to 4000 chars).
    """
    try:
        import requests
        from bs4 import BeautifulSoup

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
            tag.decompose()

        text = soup.get_text(separator=" ", strip=True)
        return text[:4000]
    except Exception as e:
        return f"Error fetching webpage {url}: {str(e)}"


if __name__ == "__main__":
    # Start standalone stdio server for MCP clients
    mcp.run(transport="stdio")

