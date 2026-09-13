"""
Graph State Schema for the LangGraph Autonomous Research AI Agent.

Defines the centralized `ResearchState` dictionary schema passed between nodes in the StateGraph.
Uses `Annotated[..., operator.add]` for automatic list accumulation across iterative reflection cycles.
"""

import operator
from typing import TypedDict, List, Dict, Any, Annotated, Optional


class ResearchState(TypedDict):
    """
    Centralized data contract representing the full state of a research cycle.
    
    Data Flow:
    1. Input: `topic`, `max_iterations`, `iteration` initialized by the client.
    2. Node 'plan': Formulates `plan` aspects and targeted `queries`.
    3. Node 'search': Executes queries via MCP and appends new snippets to `findings` (accumulated).
    4. Node 'reflect': Assesses `findings`, populates `reflection`, and sets `is_sufficient`.
    5. Conditional Edge 'should_continue': Loops back to 'search' if false, or proceeds to 'synthesize'.
    6. Node 'synthesize': Synthesizes `final_report`, optionally renders `image_url`, and updates `chat_history`.
    """

    # --- 1. Target & Execution Settings ---
    topic: str
    max_iterations: int
    iteration: int
    enable_image: Optional[bool]

    # --- 2. Multi-turn Conversation Memory (Accumulated) ---
    chat_history: Annotated[List[Dict[str, str]], operator.add]

    # --- 3. Planning & Evidence Accumulation ---
    plan: List[str]
    queries: List[str]
    findings: Annotated[List[Dict[str, Any]], operator.add]

    # --- 4. Critical Assessment & Routing ---
    reflection: str
    is_sufficient: bool

    # --- 5. Final Outputs & Visual Artifacts ---
    final_report: str
    image_url: Optional[str]
    image_prompt: Optional[str]
