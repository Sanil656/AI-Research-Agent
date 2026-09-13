"""LangGraph Assembly: Planning, FastMCP Search/arXiv, Reflection, and Synthesis."""

import re
import json
from typing import Dict, Any, List, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage, HumanMessage

from state import ResearchState
from mcp_tools import mcp_search_web, mcp_search_academic, mcp_generate_image
from config import get_llm, PLANNER_SYSTEM_PROMPT, REFLECTOR_SYSTEM_PROMPT, SYNTHESIZER_SYSTEM_PROMPT


def _extract_clean_json(content: str) -> Dict[str, Any]:
    """Extracts JSON dictionary from raw LLM output or code blocks."""
    content = content.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    if match := re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content):
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    start, end = content.find("{"), content.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(content[start:end + 1])
        except json.JSONDecodeError:
            pass
    return {}


def plan_node(state: ResearchState) -> Dict[str, Any]:
    """Node 1: Formulates search queries and focus aspects."""
    history = "\n".join(f"{m['role'].capitalize()}: {m['content'][:150]}" for m in state.get("chat_history", [])[-4:])
    prompt = f"Context:\n{history}\n\nTopic: {state['topic']}" if history else f"Topic: {state['topic']}"
    res = get_llm().invoke([SystemMessage(content=PLANNER_SYSTEM_PROMPT), HumanMessage(content=prompt)])
    data = _extract_clean_json(res.content)
    return {"plan": data.get("plan", [state["topic"]]), "queries": data.get("queries", [state["topic"]]), "iteration": 1}


def search_node(state: ResearchState) -> Dict[str, Any]:
    """Node 2: Executes web and academic searches via FastMCP."""
    existing_urls = {f.get("url") for f in state.get("findings", [])}
    new_findings = []

    for q in state.get("queries", []):
        for item in mcp_search_web(query=q, max_results=3):
            url = item.get("url", "").strip()
            if url and url not in existing_urls and len(item.get("content", "")) > 30:
                existing_urls.add(url)
                new_findings.append({"query": q, "title": item.get("title", "Web Source").strip(), "url": url, "content": item["content"]})

        if any(k in q.lower() for k in ["algorithm", "quantum", "architecture", "benchmark", "neural", "physics", "model"]):
            for paper in mcp_search_academic(query=q, max_results=2):
                if (url := paper.get("url", "").strip()) and url not in existing_urls:
                    existing_urls.add(url)
                    new_findings.append({"query": q, "title": paper.get("title", "arXiv Paper"), "url": url, "content": paper.get("content", "")})

    return {"findings": new_findings, "queries": []}


def reflect_node(state: ResearchState) -> Dict[str, Any]:
    """Node 3: Evaluates research sufficiency and finds gaps."""
    evidence = "\n".join(f"[{i}] {f['title']}: {f['content']}" for i, f in enumerate(state.get("findings", []), 1)) or "No results."
    prompt = f"Topic: {state['topic']}\nPlan: {state.get('plan')}\nEvidence:\n{evidence}\nCycle: {state['iteration']}/{state['max_iterations']}"
    res = get_llm().invoke([SystemMessage(content=REFLECTOR_SYSTEM_PROMPT), HumanMessage(content=prompt)])
    data = _extract_clean_json(res.content)
    return {"reflection": data.get("critique", "Assessed."), "is_sufficient": data.get("is_sufficient", False), "queries": data.get("follow_up_queries", []), "iteration": state["iteration"] + 1}


def should_continue(state: ResearchState) -> str:
    """Conditional Edge: Loop back or proceed to synthesis."""
    return "synthesize" if (state.get("is_sufficient") or state.get("iteration") > state.get("max_iterations")) else "search"


def synthesize_node(state: ResearchState) -> Dict[str, Any]:
    """Node 4: Synthesizes final report and generates AI concept visual."""
    llm = get_llm()
    findings = "\n".join(f"- [{f.get('title')}]({f.get('url')}): {f.get('content')}" for f in state.get("findings", []))
    prompt = f"Topic: {state['topic']}\nFindings:\n{findings}\n\nDeliver direct findings with citations."
    report = llm.invoke([SystemMessage(content=SYNTHESIZER_SYSTEM_PROMPT), HumanMessage(content=prompt)]).content

    img_url, img_prompt = None, None
    if state.get("enable_image", True) and not any(report.lower().strip().startswith(t) for t in ["i'm sorry", "i cannot", "policy"]):
        try:
            v_res = llm.invoke([SystemMessage(content="Craft a vivid 3D illustration prompt (15-25 words) for Flux. Output ONLY raw prompt."), HumanMessage(content=f"Topic: {state['topic']}\nSummary: {report[:200]}")])
            if (p := v_res.content.strip().strip('"\'')) and not any(p.lower().startswith(t) for t in ["i'm sorry", "cannot"]):
                img_prompt, img_url = p, mcp_generate_image(prompt=p)
        except Exception as e:
            print(f"[Visual Notice] {e}")

    return {"final_report": report, "image_url": img_url, "image_prompt": img_prompt, "chat_history": [{"role": "user", "content": state["topic"]}, {"role": "assistant", "content": report}]}


def create_research_graph(checkpointer: Optional[Any] = None):
    """Compiles the cyclic LangGraph workflow with memory."""
    wf = StateGraph(ResearchState)
    wf.add_node("plan", plan_node)
    wf.add_node("search", search_node)
    wf.add_node("reflect", reflect_node)
    wf.add_node("synthesize", synthesize_node)

    wf.add_edge(START, "plan")
    wf.add_edge("plan", "search")
    wf.add_edge("search", "reflect")
    wf.add_conditional_edges("reflect", should_continue, {"search": "search", "synthesize": "synthesize"})
    wf.add_edge("synthesize", END)

    return wf.compile(checkpointer=checkpointer or MemorySaver())

