"""
State definition for the LangGraph Research AI Agent with Thread, Chat History & Free Image Generation support.
"""

from typing import TypedDict, List, Dict, Any, Annotated, Optional
import operator


class ResearchState(TypedDict):
    # Core inputs and configuration
    topic: str
    max_iterations: int
    iteration: int
    enable_image: Optional[bool]

    # Conversation history across turns in a thread
    chat_history: Annotated[List[Dict[str, str]], operator.add]

    # Research plan & strategy
    plan: List[str]

    # Current queries to search for in this loop
    queries: List[str]

    # Accumulated findings with reducer to easily append new findings across loops
    findings: Annotated[List[Dict[str, Any]], operator.add]

    # Critical reflection, gap identification, and decision on depth
    reflection: str
    is_sufficient: bool

    # Final synthesized output and visuals
    final_report: str
    image_url: Optional[str]
    image_prompt: Optional[str]
