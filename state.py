"""Graph State Schema for the LangGraph Research AI Agent."""

import operator
from typing import TypedDict, List, Dict, Any, Annotated, Optional


class ResearchState(TypedDict):
    """Central state contract for multi-turn iterative research."""
    topic: str
    max_iterations: int
    iteration: int
    enable_image: Optional[bool]
    chat_history: Annotated[List[Dict[str, str]], operator.add]
    plan: List[str]
    queries: List[str]
    findings: Annotated[List[Dict[str, Any]], operator.add]
    reflection: str
    is_sufficient: bool
    final_report: str
    image_url: Optional[str]
    image_prompt: Optional[str]
