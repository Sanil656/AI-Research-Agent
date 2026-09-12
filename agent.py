"""
LangGraph Assembly for the Autonomous Research AI Agent.
Coordinates Planning, Web Evidence Retrieval (MCP), Critical Reflection,
Report Synthesis, and Thread Memory Persistence.
"""

import re
import json
from typing import Dict, Any, List, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage, HumanMessage

from state import ResearchState
from mcp_tools import mcp_search_web, mcp_generate_image
from config import (
    get_llm,
    PLANNER_SYSTEM_PROMPT,
    REFLECTOR_SYSTEM_PROMPT,
    SYNTHESIZER_SYSTEM_PROMPT,
)


def _extract_clean_json(content: str) -> Dict[str, Any]:
    """Safely extracts a JSON dictionary from LLM output text or Markdown code blocks."""
    content = content.strip()

    # 1. Direct JSON parse attempt
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # 2. Extract from markdown code block ```json ... ```
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # 3. Extract between first '{' and last '}'
    start_idx = content.find("{")
    end_idx = content.rfind("}")
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        try:
            return json.loads(content[start_idx:end_idx + 1])
        except json.JSONDecodeError:
            pass

    return {}


# ==========================================
# Graph Node Implementations
# ==========================================

def plan_node(state: ResearchState) -> Dict[str, Any]:
    """Deconstructs the research subject into concrete focus areas and search queries."""
    llm = get_llm()

    # Build conversation context from previous turns in the thread
    history_context = ""
    chat_history = state.get("chat_history", [])
    if chat_history:
        snippets = [
            f"{msg.get('role', 'user').capitalize()}: {msg.get('content', '')[:160]}"
            for msg in chat_history[-4:]
        ]
        history_context = "Prior Conversation Thread Context:\n" + "\n".join(snippets) + "\n\n"

    user_prompt = f"{history_context}Current Question / Topic to investigate:\n\n{state['topic']}"

    response = llm.invoke([
        SystemMessage(content=PLANNER_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt)
    ])

    data = _extract_clean_json(response.content)
    plan = data.get("plan", [f"Key aspects of {state['topic']}"])
    queries = data.get("queries", [state["topic"]])

    return {
        "plan": plan,
        "queries": queries,
        "iteration": 1
    }


def search_node(state: ResearchState) -> Dict[str, Any]:
    """Executes live web searches via MCP tools and collects deduplicated snippets."""
    queries = state.get("queries", [])
    existing_urls = {f.get("url") for f in state.get("findings", [])}
    new_findings = []

    for query in queries:
        results = mcp_search_web(query=query, max_results=3)
        for item in results:
            url = item.get("url", "").strip()
            content = item.get("content", "").strip()

            # Filter duplicates and empty snippets
            if url and url not in existing_urls and len(content) > 30:
                existing_urls.add(url)
                new_findings.append({
                    "query": query,
                    "title": item.get("title", "Web Source").strip(),
                    "url": url,
                    "content": content
                })

    return {
        "findings": new_findings,
        "queries": []
    }


def reflect_node(state: ResearchState) -> Dict[str, Any]:
    """Critically assesses collected evidence and determines if further search cycles are required."""
    llm = get_llm()

    # Format accumulated evidence for reflection
    findings_list = [
        f"[{i}] Source: {f.get('title')} ({f.get('url')})\nEvidence: {f.get('content')}\n"
        for i, f in enumerate(state.get("findings", []), 1)
    ]
    findings_text = "\n".join(findings_list) if findings_list else "No web results found yet."

    user_prompt = (
        f"Research Topic: {state['topic']}\n\n"
        f"Target Aspects:\n" + "\n".join(f"- {p}" for p in state.get("plan", [])) + "\n\n"
        f"Accumulated Evidence:\n{findings_text}\n\n"
        f"Current Cycle: {state['iteration']} of {state['max_iterations']}\n\n"
        "Assess if the core question is already sufficiently answered. If yes, set is_sufficient to true immediately."
    )

    response = llm.invoke([
        SystemMessage(content=REFLECTOR_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt)
    ])

    data = _extract_clean_json(response.content)
    critique = data.get("critique", "Evaluation completed.")
    is_sufficient = data.get("is_sufficient", False)
    follow_up_queries = data.get("follow_up_queries", [])

    return {
        "reflection": critique,
        "is_sufficient": is_sufficient,
        "queries": follow_up_queries,
        "iteration": state["iteration"] + 1
    }


def should_continue(state: ResearchState) -> str:
    """Conditional Edge: Routes to deeper search or proceeds to final synthesis."""
    if state.get("is_sufficient") or state.get("iteration") > state.get("max_iterations"):
        return "synthesize"
    return "search"


def synthesize_node(state: ResearchState) -> Dict[str, Any]:
    """Synthesizes verified evidence into a final publication report and generates a free concept visual."""
    llm = get_llm()

    findings_text = "\n".join(
        f"- [{f.get('title')}]({f.get('url')}): {f.get('content')}"
        for f in state.get("findings", [])
    )

    # Thread context for synthesis
    history_context = ""
    chat_history = state.get("chat_history", [])
    if chat_history:
        snippets = [
            f"{msg.get('role').capitalize()}: {msg.get('content')[:180]}"
            for msg in chat_history[-2:]
        ]
        history_context = "Prior Conversation Context:\n" + "\n".join(snippets) + "\n\n"

    user_prompt = (
        f"{history_context}"
        f"Target Query/Subject: {state['topic']}\n\n"
        f"Key Aspects Investigated:\n" + "\n".join(f"- {p}" for p in state.get("plan", [])) + "\n\n"
        f"Retrieved Verified Evidence:\n{findings_text}\n\n"
        "Instructions: Deliver a direct, highly focused, and actionable answer to the user's prompt. "
        "Strictly avoid conversational filler, generic history, or unnecessary padding. "
        "Lead immediately with the core findings and direct technical details."
    )

    response = llm.invoke([
        SystemMessage(content=SYNTHESIZER_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt)
    ])

    report_content = response.content
    image_url = None
    image_prompt = None

    # Generate 100% free AI concept visual via MCP Flux tool if enabled
    if state.get("enable_image", True):
        # Check if report was a safety refusal or error
        refusal_triggers = ["i'm sorry", "i cannot", "i can't help", "as an ai", "policy violation"]
        is_refusal = any(report_content.lower().strip().startswith(t) for t in refusal_triggers)
        
        if not is_refusal:
            try:
                visual_sys_prompt = (
                    "You are an expert visual designer. Based on the topic and findings, craft a single, "
                    "vivid, 3D technical illustration or photorealistic concept prompt (15-25 words) for a text-to-image generator (Flux). "
                    "Focus on concrete visual elements, lighting, materials, and 8k detail. Output ONLY the raw prompt."
                )
                v_res = llm.invoke([
                    SystemMessage(content=visual_sys_prompt),
                    HumanMessage(content=f"Topic: {state['topic']}\nSummary: {report_content[:250]}")
                ])
                candidate_prompt = v_res.content.strip().strip('"').strip("'")
                
                # Check candidate prompt
                if candidate_prompt and not any(candidate_prompt.lower().startswith(t) for t in refusal_triggers):
                    image_prompt = candidate_prompt
                    image_url = mcp_generate_image(prompt=image_prompt)
            except Exception as err:
                print(f"[Visual Generation Error] {err}")

    new_messages = [
        {"role": "user", "content": state["topic"]},
        {"role": "assistant", "content": report_content}
    ]

    return {
        "final_report": report_content,
        "image_url": image_url,
        "image_prompt": image_prompt,
        "chat_history": new_messages
    }


# ==========================================
# Graph Construction & Compilation
# ==========================================

def create_research_graph(checkpointer: Optional[Any] = None):
    """Assembles and compiles the StateGraph workflow with persistent memory."""
    workflow = StateGraph(ResearchState)

    # Add Nodes
    workflow.add_node("plan", plan_node)
    workflow.add_node("search", search_node)
    workflow.add_node("reflect", reflect_node)
    workflow.add_node("synthesize", synthesize_node)

    # Add Edges
    workflow.add_edge(START, "plan")
    workflow.add_edge("plan", "search")
    workflow.add_edge("search", "reflect")

    workflow.add_conditional_edges(
        "reflect",
        should_continue,
        {
            "search": "search",
            "synthesize": "synthesize"
        }
    )

    workflow.add_edge("synthesize", END)

    # Use default MemorySaver if no checkpointer is supplied
    if checkpointer is None:
        checkpointer = MemorySaver()

    return workflow.compile(checkpointer=checkpointer)
