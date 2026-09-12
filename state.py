"""
State definition for the LangGraph Research AI Agent.
Defines the schema for multi-turn research, live evidence accumulation, and media generation.
"""

import operator
from typing import TypedDict, List, Dict, Any, Annotated, Optional


class ResearchState(TypedDict):
    """Represents the complete state of a research investigation workflow."""

    # --- 1. Target & Depth Settings ---
    topic: str
    max_iterations: int
    iteration: int
    enable_image: Optional[bool]

    # --- 2. Multi-turn Conversation Memory ---
    chat_history: Annotated[List[Dict[str, str]], operator.add]

    # --- 3. Planning & Evidence Gathering ---
    plan: List[str]
    queries: List[str]
    findings: Annotated[List[Dict[str, Any]], operator.add]

    # --- 4. Critical Assessment & Routing ---
    reflection: str
    is_sufficient: bool

    # --- 5. Synthesized Output & Media ---
    final_report: str
    image_url: Optional[str]
    image_prompt: Optional[str]
