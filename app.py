"""
Streamlit Frontend for the LangGraph Autonomous Research AI Agent.
Modular, readable architecture featuring multi-threaded conversation history,
checkpointer persistence, live research step streaming, and free Flux AI image generation.
"""

import os
import uuid
import time
from typing import Dict, Any
import streamlit as st
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver

from agent import create_research_graph

# Load environment configuration
load_dotenv()


# ==========================================
# 1. Page Configuration & Custom CSS
# ==========================================

def setup_page():
    """Initializes page meta settings and styling."""
    st.set_page_config(
        page_title="Autonomous Research AI Agent",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
    <style>
        .main-title {
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 0.1rem;
        }
        .sub-title {
            color: #9ca3af;
            font-size: 0.95rem;
            margin-bottom: 1.2rem;
        }
        .badge {
            background-color: #1e293b;
            color: #94a3b8;
            font-size: 0.75rem;
            padding: 2px 8px;
            border-radius: 12px;
            font-family: monospace;
        }
    </style>
    """, unsafe_allow_html=True)


# ==========================================
# 2. Session State & Thread Initialization
# ==========================================

def init_session_state():
    """Sets up persistent memory, checkpointer, and conversation threads."""
    if "checkpointer" not in st.session_state:
        st.session_state.checkpointer = MemorySaver()

    if "graph" not in st.session_state:
        st.session_state.graph = create_research_graph(checkpointer=st.session_state.checkpointer)

    if "threads" not in st.session_state:
        default_id = "thread_" + str(uuid.uuid4())[:8]
        st.session_state.threads = {
            default_id: {
                "id": default_id,
                "title": "New Research Session",
                "created_at": time.strftime("%Y-%m-%d %H:%M"),
                "messages": []
            }
        }
        st.session_state.current_thread_id = default_id

    if "current_thread_id" not in st.session_state or st.session_state.current_thread_id not in st.session_state.threads:
        st.session_state.current_thread_id = list(st.session_state.threads.keys())[0]


# ==========================================
# 3. Sidebar UI (Settings & Thread Switcher)
# ==========================================

def render_sidebar() -> Dict[str, Any]:
    """Renders the sidebar navigation, settings, and provider options."""
    with st.sidebar:
        st.title("🔬 Research Agent")

        # Create New Thread
        if st.button("➕ New Research Thread", type="primary", use_container_width=True):
            new_id = "thread_" + str(uuid.uuid4())[:8]
            st.session_state.threads[new_id] = {
                "id": new_id,
                "title": "New Research Session",
                "created_at": time.strftime("%Y-%m-%d %H:%M"),
                "messages": []
            }
            st.session_state.current_thread_id = new_id
            st.rerun()

        st.divider()

        # Thread List Navigation
        st.subheader("💬 Research Threads")
        for tid, tinfo in list(st.session_state.threads.items()):
            is_active = (tid == st.session_state.current_thread_id)
            btn_label = f"📌 {tinfo['title']}" if is_active else f"💭 {tinfo['title']}"

            col_t, col_del = st.columns([5, 1])
            with col_t:
                if st.button(btn_label, key=f"btn_{tid}", use_container_width=True):
                    st.session_state.current_thread_id = tid
                    st.rerun()
            with col_del:
                if len(st.session_state.threads) > 1:
                    if st.button("✕", key=f"del_{tid}", help="Delete thread"):
                        del st.session_state.threads[tid]
                        st.session_state.current_thread_id = list(st.session_state.threads.keys())[0]
                        st.rerun()

        st.divider()

        # Model & Depth Settings
        st.subheader("⚙️ Settings")
        provider = st.selectbox(
            "LLM Provider",
            options=["groq", "gemini", "openai", "ollama"],
            index=0
        )

        if provider == "groq":
            model_name = st.text_input("Groq Model", value=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"))
        elif provider == "gemini":
            model_name = st.text_input("Gemini Model", value=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
        elif provider == "openai":
            model_name = st.text_input("OpenAI Model", value=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
        else:
            model_name = st.text_input("Ollama Model", value="llama3.1")

        max_depth = st.slider(
            "Research Depth (Cycles)",
            min_value=1,
            max_value=5,
            value=2,
            help="Number of reflection and live web deepening cycles."
        )

        enable_image = st.checkbox(
            "🎨 Generate Free AI Visual (Flux)",
            value=True,
            help="Automatically generates a 3D technical infographic using Pollinations Flux (100% free, no API key)."
        )

        # Status Indicators
        tavily_key = os.getenv("TAVILY_API_KEY", "").strip()
        if tavily_key:
            st.success("🔎 Search: Tavily API (Active)")
        else:
            st.info("🔎 Search: DuckDuckGo (Live & News)")

        st.success("🔌 Protocol: Model Context Protocol (MCP) Active")

        return {
            "provider": provider,
            "model_name": model_name,
            "max_depth": max_depth,
            "enable_image": enable_image
        }


# ==========================================
# 4. Message Rendering Helpers
# ==========================================

def render_assistant_message(msg: Dict[str, Any], msg_idx: int):
    """Renders an individual assistant message including visual, steps, and report."""
    # 1. Render Image Banner if available
    if msg.get("image_url"):
        st.image(
            msg["image_url"],
            caption=f"🎨 Concept Visual: {msg.get('image_prompt', 'AI Render')}",
            use_container_width=True
        )

    # 2. Render Expandable Research Steps
    steps = msg.get("steps")
    if steps:
        with st.expander("🔍 View Research Steps & Evidence Collected"):
            if steps.get("plan"):
                st.markdown("**🧠 Research Dimensions & Queries:**")
                for p in steps["plan"]:
                    st.write(f"- {p}")
                for q in steps.get("queries", []):
                    st.code(q, language="text")

            if steps.get("findings"):
                st.markdown(f"**🔎 Verified Sources ({len(steps['findings'])}):**")
                for f in steps["findings"][:6]:
                    st.markdown(f"- [{f.get('title')}]({f.get('url')}): *{f.get('content')[:140]}...*")

            if steps.get("reflection"):
                st.markdown(f"**🪞 Critical Reflection:**\n> {steps['reflection']}")

    # 3. Render Synthesized Report
    st.markdown(msg["content"])

    # 4. Download Button
    st.download_button(
        label="⬇ Download Report (.md)",
        data=msg["content"],
        file_name=f"research_report_{msg_idx}.md",
        mime="text/markdown",
        key=f"dl_{msg_idx}"
    )


# ==========================================
# 5. Core Execution Engine
# ==========================================

def execute_research(user_query: str, current_thread: Dict[str, Any], settings: Dict[str, Any]):
    """Executes the LangGraph research workflow and streams live status updates."""
    # Set runtime environment overrides
    os.environ["DEFAULT_LLM_PROVIDER"] = settings["provider"]
    if settings["provider"] == "groq":
        os.environ["GROQ_MODEL"] = settings["model_name"]
    elif settings["provider"] == "gemini":
        os.environ["GEMINI_MODEL"] = settings["model_name"]
    elif settings["provider"] == "openai":
        os.environ["OPENAI_MODEL"] = settings["model_name"]

    # Auto-generate short thread title if first message
    if not current_thread["messages"]:
        short_title = user_query[:32] + ("..." if len(user_query) > 32 else "")
        current_thread["title"] = short_title

    # Append and render user message
    current_thread["messages"].append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Execute assistant turn
    with st.chat_message("assistant"):
        step_collector = {"plan": [], "queries": [], "findings": [], "reflection": ""}
        final_report = ""
        generated_image_url = None
        generated_image_prompt = None

        with st.status("🧠 Agent thinking: Formulating research strategy...", expanded=True) as status_box:
            try:
                # Prepare conversation history state for graph
                chat_history_state = [
                    {"role": m["role"], "content": m["content"]}
                    for m in current_thread["messages"][:-1]
                ]

                initial_state = {
                    "topic": user_query,
                    "max_iterations": settings["max_depth"],
                    "iteration": 0,
                    "enable_image": settings["enable_image"],
                    "chat_history": chat_history_state,
                    "plan": [],
                    "queries": [],
                    "findings": [],
                    "reflection": "",
                    "is_sufficient": False,
                    "final_report": "",
                    "image_url": None,
                    "image_prompt": None
                }

                config = {"configurable": {"thread_id": current_thread["id"]}}

                # Stream updates through LangGraph nodes
                for step in st.session_state.graph.stream(initial_state, config=config, stream_mode="updates"):
                    for node_name, output in step.items():
                        if node_name == "plan":
                            step_collector["plan"] = output.get("plan", [])
                            step_collector["queries"] = output.get("queries", [])
                            status_box.update(label="🧠 Step 1: Research Plan & Queries Formulated")
                            st.write("#### 🧠 Research Strategy")
                            for p in step_collector["plan"]:
                                st.write(f"- {p}")
                            for q in step_collector["queries"]:
                                st.code(q, language="text")

                        elif node_name == "search":
                            new_findings = output.get("findings", [])
                            step_collector["findings"].extend(new_findings)
                            status_box.update(label=f"🔎 Step 2: Retrieved {len(step_collector['findings'])} Sources")
                            st.write(f"**Retrieved {len(new_findings)} evidence snippets.**")

                        elif node_name == "reflect":
                            iter_num = output.get("iteration", 1) - 1
                            is_sufficient = output.get("is_sufficient", False)
                            step_collector["reflection"] = output.get("reflection", "")
                            next_queries = output.get("queries", [])

                            state_label = "Sufficient Depth" if is_sufficient else f"Deepening (Cycle {iter_num}/{settings['max_depth']})"
                            status_box.update(label=f"🪞 Step 3: Reflection ({state_label})")

                            st.write(f"#### 🪞 Reflection Assessment (Cycle {iter_num})")
                            st.info(step_collector["reflection"])

                            if next_queries and not is_sufficient:
                                st.write("**Follow-up Investigation Queries:**")
                                for q in next_queries:
                                    st.code(q, language="text")

                        elif node_name == "synthesize":
                            status_box.update(label="📝 Final Synthesis: Crafting response & visual concept...")
                            final_report = output.get("final_report", "")
                            generated_image_url = output.get("image_url")
                            generated_image_prompt = output.get("image_prompt")

                status_box.update(label="✅ Research & Visual Concept Complete!", state="complete", expanded=False)

            except Exception as e:
                status_box.update(label="⚠️ Error during execution", state="error", expanded=True)
                st.error(f"Execution Error: {str(e)}")
                final_report = f"An error occurred: {str(e)}"

        # Display and persist assistant response
        if final_report:
            if generated_image_url:
                st.image(
                    generated_image_url,
                    caption=f"🎨 Concept Visual: {generated_image_prompt}",
                    use_container_width=True
                )

            st.markdown(final_report)

            current_thread["messages"].append({
                "role": "assistant",
                "content": final_report,
                "steps": step_collector,
                "image_url": generated_image_url,
                "image_prompt": generated_image_prompt
            })

            st.download_button(
                label="⬇ Download Report (.md)",
                data=final_report,
                file_name=f"research_report_{len(current_thread['messages'])}.md",
                mime="text/markdown",
                key=f"dl_live_{len(current_thread['messages'])}"
            )

    st.rerun()


# ==========================================
# 6. Main Application Entrypoint
# ==========================================

def main():
    """Main application loop."""
    setup_page()
    init_session_state()

    settings = render_sidebar()
    current_thread = st.session_state.threads[st.session_state.current_thread_id]

    # Header
    st.markdown(f'<div class="main-title">🔬 {current_thread["title"]}</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="sub-title">Active Thread ID: <span class="badge">{current_thread["id"]}</span> &nbsp;|&nbsp; '
        f'Multi-turn research memory & Model Context Protocol (MCP) enabled</div>',
        unsafe_allow_html=True
    )

    # Quick topic pills for new threads
    if not current_thread["messages"]:
        st.info("💡 Start a new research investigation below, or try one of these topics:")
        col_p1, col_p2, col_p3 = st.columns(3)
        if col_p1.button("🔋 Solid State EV Batteries", use_container_width=True):
            st.session_state.initial_prompt = "What is the commercialization timeline and energy density of solid state EV batteries in 2026?"
        if col_p2.button("🚀 GPT-6 Astra Features", use_container_width=True):
            st.session_state.initial_prompt = "What are the latest announcements and capabilities of OpenAI's GPT-6 Astra?"
        if col_p3.button("🔐 Post-Quantum Cryptography", use_container_width=True):
            st.session_state.initial_prompt = "What are the primary NIST Post-Quantum Cryptography algorithms and migration roadmaps?"

    # Render Chronological Chat History
    for idx, msg in enumerate(current_thread["messages"]):
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.markdown(msg["content"])
            else:
                render_assistant_message(msg, idx)

    # Chat Input Box
    preset_prompt = st.session_state.pop("initial_prompt", None)
    user_query = st.chat_input("Ask a research question or follow-up...") or preset_prompt

    if user_query:
        execute_research(user_query, current_thread, settings)


if __name__ == "__main__":
    main()
