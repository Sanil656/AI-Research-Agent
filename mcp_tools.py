"""
Model Context Protocol (MCP) Client Bridge for LangGraph.

Connects LangGraph state nodes to standardized FastMCP server tools.
Provides graceful direct-execution fallback if the MCP server transport is offline.
"""

import json
from typing import List, Dict, Any
from mcp_server import (
    search_web as _mcp_search,
    search_academic as _mcp_academic,
    generate_image as _mcp_gen_img,
    fetch_page as _mcp_fetch,
)


def mcp_search_web(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """
    Invokes the search_web tool via the Model Context Protocol (MCP) server.
    """
    try:
        raw_output = _mcp_search(query=query, max_results=max_results)
        if isinstance(raw_output, str):
            return json.loads(raw_output)
        return raw_output
    except Exception as err:
        print(f"[MCP Search Warning] Falling back to direct search ({err})")
        from tools import search_web
        return search_web(query=query, max_results=max_results)


def mcp_search_academic(query: str, max_results: int = 3) -> List[Dict[str, str]]:
    """
    Invokes the search_academic (arXiv) tool via the Model Context Protocol (MCP) server.
    """
    try:
        raw_output = _mcp_academic(query=query, max_results=max_results)
        if isinstance(raw_output, str):
            return json.loads(raw_output)
        return raw_output
    except Exception as err:
        print(f"[MCP Academic Warning] Falling back to direct arXiv search ({err})")
        from tools import search_academic_arxiv
        return search_academic_arxiv(query=query, max_results=max_results)


def mcp_generate_image(prompt: str, width: int = 1024, height: int = 640) -> str:
    """
    Invokes the generate_image (Flux.1) tool via the Model Context Protocol (MCP) server.
    """
    try:
        return _mcp_gen_img(prompt=prompt, width=width, height=height)
    except Exception as err:
        print(f"[MCP Image Warning] Falling back to direct tool ({err})")
        from tools import get_free_image_url
        return get_free_image_url(prompt=prompt, width=width, height=height)


def mcp_fetch_page(url: str) -> str:
    """
    Invokes the fetch_page (HTML scrape) tool via the Model Context Protocol (MCP) server.
    """
    try:
        return _mcp_fetch(url=url)
    except Exception as err:
        return f"Error via MCP: {str(err)}"
