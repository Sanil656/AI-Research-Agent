"""MCP Client Bridge connecting LangGraph nodes to FastMCP tools."""

import json
from typing import List, Dict, Any
from mcp_server import search_web as _mcp_search, search_academic as _mcp_academic, generate_image as _mcp_gen_img, fetch_page as _mcp_fetch


def mcp_search_web(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """Invokes search_web via MCP with fallback."""
    try:
        res = _mcp_search(query=query, max_results=max_results)
        return json.loads(res) if isinstance(res, str) else res
    except Exception:
        from tools import search_web
        return search_web(query=query, max_results=max_results)


def mcp_search_academic(query: str, max_results: int = 3) -> List[Dict[str, str]]:
    """Invokes search_academic (arXiv) via MCP with fallback."""
    try:
        res = _mcp_academic(query=query, max_results=max_results)
        return json.loads(res) if isinstance(res, str) else res
    except Exception:
        from tools import search_academic_arxiv
        return search_academic_arxiv(query=query, max_results=max_results)


def mcp_generate_image(prompt: str, width: int = 1024, height: int = 640) -> str:
    """Invokes generate_image via MCP with fallback."""
    try:
        return _mcp_gen_img(prompt=prompt, width=width, height=height)
    except Exception:
        from tools import get_free_image_url
        return get_free_image_url(prompt=prompt, width=width, height=height)


def mcp_fetch_page(url: str) -> str:
    """Invokes fetch_page via MCP."""
    try:
        return _mcp_fetch(url=url)
    except Exception as err:
        return f"Error: {err}"
