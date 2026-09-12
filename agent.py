"""
LangGraph assembly for the Research AI Agent.
Coordinates planning, web searching, reflection, report synthesis, and thread memory.
"""

import json
import re
from typing import Dict, Any, List, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage, HumanMessage

from state import ResearchState
from tools import search_web, get_free_image_url
from config import (
    get_llm,
    PLANNER_SYSTEM_PROMPT,
    REFLECTOR_SYSTEM_PROMPT,
    SYNTHESIZER_SYSTEM_PROMPT,
)


def _clean_json_response(content: str) -> Dict[str, Any]:
    """Helper to extract clean JSON from LLM text that might include markdown code fences."""
    content = content.strip()
    # Try direct parse
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Try extracting inside ```json ... ``` or ``` ... ```
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try finding the first '{' and last '}'
    first_brace = content.find("{")
    last_brace = content.rfind("}")
    if first_brace != -1 and last_brace != -1:
        try:
            return json.loads(content[first_brace:last_brace + 1])
        except json.JSONDecodeError:
            pass

    return {}


def plan_node(state: ResearchState) -> Dict[str, Any]:
    """Deconstructs the research topic into analytical sub-dimensions and initial search queries with thread context."""
    llm = get_llm()

    # Incorporate conversation thread context if available
    history_context = ""
    chat_history = state.get("chat_history", [])
    if chat_history:
        history_snippets = []
        for msg in chat_history[-4:]:  # last 2 turns
            role = msg.get("role", "user").capitalize()
            content_preview = msg.get("content", "")[:180]
            history_snippets.append(f"{role}: {content_preview}")
        history_context = "Prior Conversation Thread Context:\n" + "\n".join(history_snippets) + "\n\n"

    user_prompt = f"{history_context}Current Question / Topic to investigate:\n\n{state['topic']}"

    response = llm.invoke([
        SystemMessage(content=PLANNER_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt)
    ])

    data = _clean_json_response(response.content)
    plan = data.get("plan", [f"Key aspects of {state['topic']}"])
    queries = data.get("queries", [state["topic"]])

    return {
        "plan": plan,
        "queries": queries,
        "iteration": 1
    }


def search_node(state: ResearchState) -> Dict[str, Any]:
    """Executes web searches for active queries and collects deduplicated, high-signal snippets."""
    queries = state.get("queries", [])
    existing_urls = {f.get("url") for f in state.get("findings", [])}
    new_findings = []

    for q in queries:
        results = search_web(q, max_results=3)
        for r in results:
            url = r.get("url", "").strip()
            content = r.get("content", "").strip()
            # Filter out duplicates and empty/too-short snippets
            if url and url not in existing_urls and len(content) > 30:
                existing_urls.add(url)
                new_findings.append({
                    "query": q,
                    "title": r.get("title", "Web Source").strip(),
                    "url": url,
                    "content": content
                })

    return {
        "findings": new_findings,
        "queries": []
    }


def reflect_node(state: ResearchState) -> Dict[str, Any]:
    """Critically evaluates current knowledge base, identifies gaps, and decides if more research is required."""
    llm = get_llm()

    # Format findings concisely for LLM assessment
    findings_summary = []
    for idx, f in enumerate(state.get("findings", []), 1):
        findings_summary.append(
            f"[{idx}] Source: {f.get('title')} ({f.get('url')})\nEvidence: {f.get('content')}\n"
        )
    findings_text = "\n".join(findings_summary) if findings_summary else "No web results found yet."

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

    data = _clean_json_response(response.content)
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
    """Routing logic: continue research loop if depth is lacking and under max iterations."""
    if state.get("is_sufficient") or state.get("iteration") > state.get("max_iterations"):
        return "synthesize"
    return "search"


def synthesize_node(state: ResearchState) -> Dict[str, Any]:
    """Generates the final direct research report and an optional free AI visual concept."""
    llm = get_llm()

    findings_text = ""
    for idx, f in enumerate(state.get("findings", []), 1):
        findings_text += f"\n- [{f.get('title')}]({f.get('url')}): {f.get('content')}"

    # Thread context for synthesis
    history_context = ""
    chat_history = state.get("chat_history", [])
    if chat_history:
        history_snippets = []
        for msg in chat_history[-2:]:
            history_snippets.append(f"{msg.get('role').capitalize()}: {msg.get('content')[:200]}")
        history_context = "Prior Conversation Context:\n" + "\n".join(history_snippets) + "\n\n"

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

    # Generate 100% free visual illustration if enabled
    if state.get("enable_image", True):
        try:
            visual_prompt_sys = (
                "You are an expert visual designer. Based on the topic and findings, craft a single, "
                "vivid, 3D technical illustration or photorealistic concept prompt (15-25 words) for a text-to-image generator (Flux). "
                "Focus on concrete visual elements, lighting, materials, and 8k detail. Output ONLY the raw prompt."
            )
            v_res = llm.invoke([
                SystemMessage(content=visual_prompt_sys),
                HumanMessage(content=f"Topic: {state['topic']}\nSummary: {report_content[:250]}")
            ])
            image_prompt = v_res.content.strip().strip('"').strip("'")
            image_url = get_free_image_url(image_prompt)
        except Exception as e:
            print(f"[Image Prompt Error] {e}")

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


def create_research_graph(checkpointer: Optional[Any] = None):
    """Builds and compiles the LangGraph StateGraph with optional MemorySaver checkpointer."""
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

    # Use provided checkpointer or default to in-memory checkpointer
    if checkpointer is None:
        checkpointer = MemorySaver()

    return workflow.compile(checkpointer=checkpointer)
