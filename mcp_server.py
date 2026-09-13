"""FastMCP Tool Server for Web Search, arXiv, and Image Generation."""

import json
from mcp.server.fastmcp import FastMCP
from tools import search_web as _web, search_academic_arxiv as _arxiv, get_free_image_url as _img

mcp = FastMCP("ResearchToolsServer")

@mcp.tool()
def search_web(query: str, max_results: int = 5) -> str:
    """Search live web and breaking news."""
    return json.dumps(_web(query, max_results))

@mcp.tool()
def search_academic(query: str, max_results: int = 3) -> str:
    """Search peer-reviewed arXiv academic papers."""
    return json.dumps(_arxiv(query, max_results))

@mcp.tool()
def generate_image(prompt: str, width: int = 1024, height: int = 640) -> str:
    """Generate free AI visual concept via Pollinations Flux.1."""
    return _img(prompt, width, height)

@mcp.tool()
def fetch_page(url: str) -> str:
    """Fetch clean text content from a webpage URL."""
    try:
        import requests
        from bs4 import BeautifulSoup
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=8)
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        return soup.get_text(separator=" ", strip=True)[:4000]
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    mcp.run(transport="stdio")

