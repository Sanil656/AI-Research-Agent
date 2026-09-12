"""
Model Context Protocol (MCP) Client Bridge for LangGraph.
Dispatches tool invocations through the MCP server tools with automatic fallback.
"""

import json
from typing import List, Dict, Any
from mcp_server import (
    search_web as _mcp_search,
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
        print(f"[MCP Search Warning] Falling back to direct tool ({err})")
        from tools import search_web
        return search_web(query=query, max_results=max_results)


def mcp_generate_image(prompt: str, width: int = 1024, height: int = 640) -> str:
    """
    Invokes the generate_image tool via the Model Context Protocol (MCP) server.
    """
    try:
        return _mcp_gen_img(prompt=prompt, width=width, height=height)
    except Exception as err:
        print(f"[MCP Image Warning] Falling back to direct tool ({err})")
        from tools import get_free_image_url
        return get_free_image_url(prompt=prompt, width=width, height=height)


def mcp_fetch_page(url: str) -> str:
    """
    Invokes the fetch_page tool via the Model Context Protocol (MCP) server.
    """
    try:
        return _mcp_fetch(url=url)
    except Exception as err:
        return f"Error via MCP: {str(err)}"
