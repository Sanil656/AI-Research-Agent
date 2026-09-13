"""
LangGraph Assembly for the Autonomous Research AI Agent.

Architecture Overview:
======================
                  ┌───────────────────────────────┐
                  │ 💬 Start: User Research Topic  │
                  └───────────────┬───────────────┘
                                  │
                                  ▼
                     ┌─────────────────────────┐
                     │ 🧠 Node 1: 'plan'       │
                     │  - Deconstructs topic   │
                     │  - Formulates queries   │
                     └────────────┬────────────┘
                                  │
                                  ▼
                     ┌─────────────────────────┐◄──────────────┐
                     │ 🔎 Node 2: 'search'     │               │
                     │  - MCP Live Web Search  │               │
                     │  - MCP Academic (arXiv) │               │
                     └────────────┬────────────┘               │
                                  │                            │ (Gaps Found)
                                  ▼                            │
                     ┌─────────────────────────┐               │
                     │ 🪞 Node 3: 'reflect'    │               │
                     │  - Evaluates coverage   │               │
                     │  - Knowledge gap check  │               │
                     └────────────┬────────────┘               │
                                  │                            │
                                  ▼                            │
                       /─────────────────────\                 │
                      <  is_sufficient?       >────────────────┘
                       \─────────────────────/
                                  │ (Depth Attained)
                                  ▼
                     ┌─────────────────────────┐
                     │ 📝 Node 4: 'synthesize' │
                     │  - Ground-truth report  │
                     │  - MCP Flux Visual Card │
                     └────────────┬────────────┘
                                  │
                                  ▼
                  ┌───────────────────────────────┐
                  │ 🚀 End: Verified Intelligence │
                  └───────────────────────────────┘
"""

import re
import json
from typing import Dict, Any, List, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage, HumanMessage

from state import ResearchState
from mcp_tools import mcp_search_web, mcp_search_academic, mcp_generate_image
from config import (
    get_llm,
    PLANNER_SYSTEM_PROMPT,
    REFLECTOR_SYSTEM_PROMPT,
    SYNTHESIZER_SYSTEM_PROMPT,
)


def _extract_clean_json(content: str) -> Dict[str, Any]:
    """
    Safely extracts and parses a JSON object from raw LLM output text,
    handling Markdown code blocks (```json ... ```) or embedded JSON objects.
    """
    content = content.strip()

    # Strategy 1: Direct JSON parse
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Strategy 2: Extract from markdown code block ```json ... ```
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Strategy 3: Extract between first '{' and last '}'
    start_idx = content.find("{")
    end_idx = content.rfind("}")
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        try:
            return json.loads(content[start_idx:end_idx + 1])
        except json.JSONDecodeError:
            pass

    return {}


# =====================================================================
# Graph Node 1: Research Strategist (Planning)
# =====================================================================

def plan_node(state: ResearchState) -> Dict[str, Any]:
    """
    Formulates a targeted research strategy and decomposes the inquiry into
    2 to 3 high-signal search queries.
    """
    llm = get_llm()

    # Include recent thread history for context in multi-turn conversations
    history_context = ""
    chat_history = state.get("chat_history", [])
    if chat_history:
        snippets = [
            f"{msg.get('role', 'user').capitalize()}: {msg.get('content', '')[:160]}"
            for msg in chat_history[-4:]
        ]
        history_context = "Prior Conversation Context:\n" + "\n".join(snippets) + "\n\n"

    user_prompt = f"{history_context}Research Topic / Inquiry:\n\n{state['topic']}"

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


# =====================================================================
# Graph Node 2: Live Evidence Gathering (MCP Search)
# =====================================================================

def search_node(state: ResearchState) -> Dict[str, Any]:
    """
    Executes live web searches and academic literature lookups via FastMCP tools,
    deduplicating sources and filtering low-signal snippets.
    """
    queries = state.get("queries", [])
    existing_urls = {f.get("url") for f in state.get("findings", [])}
    new_findings: List[Dict[str, Any]] = []

    for query in queries:
        # 1. Real-time web & news search via MCP
        web_results = mcp_search_web(query=query, max_results=3)
        for item in web_results:
            url = item.get("url", "").strip()
            content = item.get("content", "").strip()

            if url and url not in existing_urls and len(content) > 30:
                existing_urls.add(url)
                new_findings.append({
                    "query": query,
                    "title": item.get("title", "Web Source").strip(),
                    "url": url,
                    "content": content
                })

        # 2. Check academic papers on arXiv if query has scientific / technical terms
        academic_keywords = ["algorithm", "quantum", "architecture", "benchmark", "neural", "physics", "model", "theorem"]
        if any(kw in query.lower() for kw in academic_keywords):
            paper_results = mcp_search_academic(query=query, max_results=2)
            for paper in paper_results:
                url = paper.get("url", "").strip()
                if url and url not in existing_urls:
                    existing_urls.add(url)
                    new_findings.append({
                        "query": query,
                        "title": paper.get("title", "Academic Paper"),
                        "url": url,
                        "content": paper.get("content", "")
                    })

    return {
        "findings": new_findings,
        "queries": []
    }


# =====================================================================
# Graph Node 3: Critical Reflection & Knowledge Gap Analysis
# =====================================================================

def reflect_node(state: ResearchState) -> Dict[str, Any]:
    """
    Critically assesses accumulated evidence against the user's inquiry,
    identifies gaps, and decides whether deeper search cycles are needed.
    """
    llm = get_llm()

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


# =====================================================================
# Conditional Edge: Dynamic Termination & Loop Routing
# =====================================================================

def should_continue(state: ResearchState) -> str:
    """
    Evaluates termination conditions:
    - If depth is sufficient OR max cycles exceeded -> Route to 'synthesize'.
    - Otherwise -> Loop back to 'search' with follow-up queries.
    """
    if state.get("is_sufficient") or state.get("iteration") > state.get("max_iterations"):
        return "synthesize"
    return "search"


# =====================================================================
# Graph Node 4: Final Synthesis & Visual Rendering
# =====================================================================

def synthesize_node(state: ResearchState) -> Dict[str, Any]:
    """
    Synthesizes accumulated evidence into a cohesive, publication-grade research report
    with verified citations and generates a 100% free AI concept visual.
    """
    llm = get_llm()

    findings_text = "\n".join(
        f"- [{f.get('title')}]({f.get('url')}): {f.get('content')}"
        for f in state.get("findings", [])
    )

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

    # Generate 100% free AI concept visual via FastMCP Flux tool if enabled
    if state.get("enable_image", True):
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

                if candidate_prompt and not any(candidate_prompt.lower().startswith(t) for t in refusal_triggers):
                    image_prompt = candidate_prompt
                    image_url = mcp_generate_image(prompt=image_prompt)
            except Exception as err:
                print(f"[Visual Generation Notice] {err}")

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


# =====================================================================
# StateGraph Builder & Compiler
# =====================================================================

def create_research_graph(checkpointer: Optional[Any] = None):
    """
    Constructs and compiles the cyclic LangGraph workflow with persistent memory.
    
    Returns:
        A compiled LangGraph executable graph.
    """
    workflow = StateGraph(ResearchState)

    # 1. Register Graph Nodes
    workflow.add_node("plan", plan_node)
    workflow.add_node("search", search_node)
    workflow.add_node("reflect", reflect_node)
    workflow.add_node("synthesize", synthesize_node)

    # 2. Register Graph Edges & Cycles
    workflow.add_edge(START, "plan")
    workflow.add_edge("plan", "search")
    workflow.add_edge("search", "reflect")

    # 3. Dynamic Reflection Loop
    workflow.add_conditional_edges(
        "reflect",
        should_continue,
        {
            "search": "search",
            "synthesize": "synthesize"
        }
    )

    workflow.add_edge("synthesize", END)

    # 4. Attach Checkpointer (Default: MemorySaver)
    if checkpointer is None:
        checkpointer = MemorySaver()

    return workflow.compile(checkpointer=checkpointer)

