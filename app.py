"""
Streamlit Frontend for the LangGraph Autonomous Research AI Agent.
Professional Chatbot UI featuring Perplexity-style source chips, multi-threaded
conversation history, checkpointer persistence, live research trace, and Flux AI visuals.
"""

import os
import uuid
import time
from typing import Dict, Any
import streamlit as st
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver

from agent import create_research_graph
from config import get_config_val
from database import (
    init_db,
    register_user,
    authenticate_user,
    save_user_thread,
    load_user_threads,
    delete_user_thread
)

# Load environment configuration
load_dotenv()


# ==========================================
# 1. Page Configuration & Professional CSS
# ==========================================

def setup_page():
    """Initializes page meta settings and modern chatbot styling."""
    st.set_page_config(
        page_title="Research AI Assistant",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
    <style>
        /* Main Layout & Typography */
        .main-header-title {
            font-size: 1.8rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            margin-bottom: 0.1rem;
        }
        .main-header-subtitle {
            color: #94a3b8;
            font-size: 0.9rem;
            margin-bottom: 1.2rem;
        }
        
        /* Message Meta Header */
        .msg-meta-bar {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 0.78rem;
            color: #64748b;
            margin-bottom: 10px;
        }
        .meta-pill {
            background-color: #1e293b;
            color: #94a3b8;
            padding: 2px 8px;
            border-radius: 999px;
            font-size: 0.75rem;
            font-family: monospace;
            border: 1px solid #334155;
        }
        .meta-pill-accent {
            background-color: rgba(99, 102, 241, 0.15);
            color: #a5b4fc;
            padding: 2px 8px;
            border-radius: 999px;
            font-size: 0.75rem;
            border: 1px solid rgba(99, 102, 241, 0.3);
        }

        /* Perplexity-style Sources Chips */
        .sources-wrapper {
            margin: 8px 0 16px 0;
            padding: 10px 14px;
            background-color: rgba(15, 23, 42, 0.6);
            border: 1px solid #1e293b;
            border-radius: 10px;
        }
        .sources-title {
            font-size: 0.8rem;
            font-weight: 600;
            color: #94a3b8;
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .source-chips-row {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }
        .source-chip {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background-color: #1e293b;
            color: #cbd5e1;
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 0.78rem;
            text-decoration: none;
            border: 1px solid #334155;
            transition: all 0.2s ease;
        }
        .source-chip:hover {
            background-color: #334155;
            color: #ffffff;
            border-color: #6366f1;
            text-decoration: none;
        }

        /* AI Concept Visual Card */
        .visual-card-container {
            margin: 12px 0 20px 0;
            background-color: #0b101b;
            border: 1px solid #1e293b;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        }
        .visual-card-header {
            padding: 8px 14px;
            background-color: rgba(30, 41, 59, 0.5);
            font-size: 0.78rem;
            font-weight: 600;
            color: #a5b4fc;
            display: flex;
            align-items: center;
            justify-content: space-between;
            border-bottom: 1px solid #1e293b;
        }

        /* Thread Navigation Buttons */
        .thread-btn-active {
            font-weight: 600 !important;
            border-left: 3px solid #6366f1 !important;
        }

        /* Authentication Card */
        .auth-card {
            background-color: #0f172a;
            border: 1px solid #1e293b;
            border-radius: 12px;
            padding: 24px;
            max-width: 480px;
            margin: 40px auto;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
        }
        .auth-title {
            font-size: 1.5rem;
            font-weight: 700;
            text-align: center;
            margin-bottom: 6px;
        }
        .auth-subtitle {
            color: #94a3b8;
            font-size: 0.88rem;
            text-align: center;
            margin-bottom: 20px;
        }
        .user-profile-badge {
            background: linear-gradient(135deg, #1e293b, #0f172a);
            border: 1px solid #334155;
            border-radius: 8px;
            padding: 10px;
            margin-bottom: 12px;
        }
    </style>
    """, unsafe_allow_html=True)


# ==========================================
# 2. Authentication Screen (MySQL Backed)
# ==========================================

def render_auth_screen():
    """Renders the centered Sign In / Sign Up interface."""
    setup_page()

    col_l, col_center, col_r = st.columns([1, 2, 1])
    with col_center:
        st.markdown('<div class="auth-title">🔬 Autonomous Research AI</div>', unsafe_allow_html=True)
        st.markdown('<div class="auth-subtitle">Sign in to access your persistent research history & intelligence agent</div>', unsafe_allow_html=True)

        tab_login, tab_signup = st.tabs(["🔑 Sign In", "📝 Create Account"])

        # 1. Sign In Tab
        with tab_login:
            with st.form("login_form"):
                username_or_email = st.text_input("Username or Email", placeholder="alex or alex@example.com")
                password = st.text_input("Password", type="password", placeholder="••••••••")
                submit_login = st.form_submit_button("Sign In", type="primary", use_container_width=True)

                if submit_login:
                    user_data, msg = authenticate_user(username_or_email, password)
                    if user_data:
                        st.session_state.authenticated_user = user_data
                        # Load user's saved threads from MySQL
                        saved_threads = load_user_threads(user_data["id"])
                        if saved_threads:
                            st.session_state.threads = saved_threads
                            st.session_state.current_thread_id = list(saved_threads.keys())[0]
                        st.success(f"Welcome back, {user_data['username']}! 👋")
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        st.error(msg)

        # 2. Sign Up Tab
        with tab_signup:
            with st.form("signup_form"):
                new_username = st.text_input("Choose Username", placeholder="e.g. datascientist")
                new_email = st.text_input("Email Address", placeholder="name@company.com")
                new_password = st.text_input("Create Password", type="password", placeholder="Min. 6 characters")
                submit_signup = st.form_submit_button("Create Account", type="primary", use_container_width=True)

                if submit_signup:
                    success, msg = register_user(new_username, new_email, new_password)
                    if success:
                        st.success(msg)
                    else:
                        st.error(msg)

        st.divider()
        col_g1, col_g2 = st.columns([1, 1])
        with col_g1:
            if st.button("⚡ Continue as Guest", use_container_width=True, help="Explore without saving history to MySQL"):
                st.session_state.authenticated_user = {
                    "id": "guest_" + str(uuid.uuid4())[:6],
                    "username": "Guest User",
                    "email": "guest@local",
                    "created_at": time.strftime("%Y-%m-%d")
                }
                st.rerun()
        with col_g2:
            st.caption("🔒 Secured with bcrypt salted hashing & MySQL isolated threads.")


# ==========================================
# 3. Session State & Thread Initialization
# ==========================================

def init_session_state():
    """Sets up persistent memory, checkpointer, and conversation threads."""
    if "checkpointer" not in st.session_state:
        st.session_state.checkpointer = MemorySaver()

    if "graph" not in st.session_state:
        st.session_state.graph = create_research_graph(checkpointer=st.session_state.checkpointer)

    if "threads" not in st.session_state:
        user = st.session_state.get("authenticated_user", {})
        saved_threads = {}
        if user and not user.get("id", "").startswith("guest_"):
            saved_threads = load_user_threads(user["id"])

        if saved_threads:
            st.session_state.threads = saved_threads
            st.session_state.current_thread_id = list(saved_threads.keys())[0]
        else:
            default_id = "thread_" + str(uuid.uuid4())[:8]
            st.session_state.threads = {
                default_id: {
                    "id": default_id,
                    "title": "New Research Session",
                    "created_at": time.strftime("%b %d, %H:%M"),
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
        # User Profile Header
        user = st.session_state.get("authenticated_user", {})
        username = user.get("username", "User")
        is_guest = user.get("id", "").startswith("guest_")

        col_u1, col_u2 = st.columns([4, 1])
        with col_u1:
            st.markdown(f'<div class="user-profile-badge">👤 <b>{username}</b><br><span style="font-size:0.75rem; color:#94a3b8;">{"⚡ Guest Account" if is_guest else "🔒 MySQL Synced"}</span></div>', unsafe_allow_html=True)
        with col_u2:
            if st.button("🚪", key="btn_logout", help="Log out of account"):
                st.session_state.pop("authenticated_user", None)
                st.session_state.pop("threads", None)
                st.rerun()

        # New Thread Button
        if st.button("➕ New Research Chat", type="primary", use_container_width=True):
            new_id = "thread_" + str(uuid.uuid4())[:8]
            st.session_state.threads[new_id] = {
                "id": new_id,
                "title": "New Research Session",
                "created_at": time.strftime("%b %d, %H:%M"),
                "messages": []
            }
            st.session_state.current_thread_id = new_id
            st.rerun()

        st.divider()

        # Thread List Navigation
        st.markdown("**💬 Conversations**")
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
                        if not is_guest and "id" in user:
                            delete_user_thread(user["id"], tid)
                        del st.session_state.threads[tid]
                        st.session_state.current_thread_id = list(st.session_state.threads.keys())[0]
                        st.rerun()

        st.divider()

        # Model & Depth Settings
        st.markdown("**⚙️ Configuration**")
        provider = st.selectbox(
            "AI Inference Engine",
            options=["groq", "gemini", "openai", "ollama"],
            index=0
        )

        # Dynamic Key Check & Input
        key_env_map = {
            "groq": "GROQ_API_KEY",
            "gemini": "GEMINI_API_KEY",
            "openai": "OPENAI_API_KEY"
        }
        
        if provider in key_env_map:
            env_var = key_env_map[provider]
            current_key = get_config_val(env_var)
            if not current_key:
                user_key = st.text_input(
                    f"🔑 {provider.upper()} API Key",
                    type="password",
                    help=f"Enter your {provider.capitalize()} API key (saved for this session)."
                )
                if user_key:
                    os.environ[env_var] = user_key.strip()
            else:
                st.caption(f"🔑 {provider.capitalize()} Key: `Connected` ✅")

        if provider == "groq":
            groq_presets = [
                "openai/gpt-oss-120b",
                "openai/gpt-oss-20b",
                "groq/compound",
                "groq/compound-mini",
                "qwen/qwen3.6-27b",
                "Custom..."
            ]
            default_g_model = get_config_val("GROQ_MODEL", "openai/gpt-oss-120b")
            default_idx = groq_presets.index(default_g_model) if default_g_model in groq_presets else 0
            
            selected_preset = st.selectbox(
                "Groq Model",
                options=groq_presets,
                index=default_idx,
                help="openai/gpt-oss-120b is the 120B flagship reasoning model on Groq."
            )
            if selected_preset == "Custom...":
                model_name = st.text_input("Custom Groq Model", value="openai/gpt-oss-120b")
            else:
                model_name = selected_preset
        elif provider == "gemini":
            model_name = st.text_input("Gemini Model", value=get_config_val("GEMINI_MODEL", "gemini-2.5-flash"))
        elif provider == "openai":
            model_name = st.text_input("OpenAI Model", value=get_config_val("OPENAI_MODEL", "gpt-4o-mini"))
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
            "🎨 Generate AI Concept Visuals",
            value=True,
            help="Generates an accompanying 3D visual infographic using the Pollinations Flux engine (100% Free)."
        )

        # Status Indicators
        st.caption("System Status:")
        tavily_key = get_config_val("TAVILY_API_KEY")
        if tavily_key:
            st.success("🔎 Web Search: Tavily API (Active)")
        else:
            st.info("🔎 Web Search: DuckDuckGo (Live + News)")

        st.success("🔌 MCP Protocol: Active")

        return {
            "provider": provider,
            "model_name": model_name,
            "max_depth": max_depth,
            "enable_image": enable_image
        }


# ==========================================
# 4. Professional Message Formatter
# ==========================================

def render_assistant_message(msg: Dict[str, Any], msg_idx: int):
    """Renders a structured, professional chatbot response."""
    steps = msg.get("steps", {})
    findings = steps.get("findings", []) if steps else []

    # 1. Metadata Header Bar
    meta_model = msg.get("model", "Research AI")
    st.markdown(
        f'<div class="msg-meta-bar">'
        f'<span class="meta-pill-accent">⚡ {meta_model}</span>'
        f'<span class="meta-pill">🔍 {len(findings)} Web Sources</span>'
        f'</div>',
        unsafe_allow_html=True
    )

    # 2. Perplexity-Style Verified Source Chips
    if findings:
        chips_html = []
        for f in findings[:6]:
            title = (f.get("title") or "Web Source")[:24] + "..."
            url = f.get("url", "#")
            chips_html.append(f'<a href="{url}" target="_blank" class="source-chip">🌐 {title} ↗</a>')

        st.markdown(
            f'<div class="sources-wrapper">'
            f'<div class="sources-title">Verified Sources & Evidence</div>'
            f'<div class="source-chips-row">{"".join(chips_html)}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    # 3. Hero AI Concept Visual Card (Framed)
    if msg.get("image_url"):
        prompt_preview = msg.get("image_prompt", "AI Concept Diagram")
        st.markdown(
            f'<div class="visual-card-container">'
            f'<div class="visual-card-header">'
            f'<span>🎨 AI Visual Concept & Infographic</span>'
            f'<span style="opacity: 0.7; font-size: 0.7rem;">Flux.1 Engine</span>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True
        )
        st.image(
            msg["image_url"],
            caption=f"Prompt: {prompt_preview}",
            use_container_width=True
        )

    # 4. Synthesized Research Report Content
    st.markdown(msg["content"])

    # 5. Expandable Detailed Research Trace
    if steps:
        with st.expander("🧠 View Agent Thinking Process & Reflection"):
            if steps.get("plan"):
                st.markdown("**1. Formulated Research Strategy:**")
                for p in steps["plan"]:
                    st.write(f"- {p}")
                st.markdown("**Executed Search Queries:**")
                for q in steps.get("queries", []):
                    st.code(q, language="text")

            if steps.get("reflection"):
                st.markdown(f"**2. Critical Reflection:**\n> {steps['reflection']}")

    # 6. Action Toolbar Footer
    col_dl, col_space = st.columns([1, 4])
    with col_dl:
        st.download_button(
            label="⬇ Download Report (.md)",
            data=msg["content"],
            file_name=f"research_report_{msg_idx + 1}.md",
            mime="text/markdown",
            key=f"dl_{msg_idx}",
            use_container_width=True
        )


# ==========================================
# 5. Execution Pipeline
# ==========================================

def execute_research(user_query: str, current_thread: Dict[str, Any], settings: Dict[str, Any]):
    """Executes the research agent and streams updates into the chat UI."""
    os.environ["DEFAULT_LLM_PROVIDER"] = settings["provider"]
    if settings["provider"] == "groq":
        os.environ["GROQ_MODEL"] = settings["model_name"]
    elif settings["provider"] == "gemini":
        os.environ["GEMINI_MODEL"] = settings["model_name"]
    elif settings["provider"] == "openai":
        os.environ["OPENAI_MODEL"] = settings["model_name"]

    # Generate thread title from first question
    if not current_thread["messages"]:
        short_title = user_query[:28] + ("..." if len(user_query) > 28 else "")
        current_thread["title"] = short_title

    # Append user prompt
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

                status_box.update(label="✅ Research & Visual Synthesis Complete!", state="complete", expanded=False)

            except Exception as e:
                status_box.update(label="⚠️ Error during execution", state="error", expanded=True)
                st.error(f"Execution Error: {str(e)}")
                final_report = f"An error occurred: {str(e)}"

        if final_report:
            # Model display string
            model_display = f"{settings['provider'].capitalize()} • {settings['model_name']}"

            # Save assistant message
            assistant_msg = {
                "role": "assistant",
                "content": final_report,
                "model": model_display,
                "steps": step_collector,
                "image_url": generated_image_url,
                "image_prompt": generated_image_prompt
            }
            current_thread["messages"].append(assistant_msg)

            # Persist thread to MySQL for registered users
            user = st.session_state.get("authenticated_user", {})
            if user and not user.get("id", "").startswith("guest_"):
                save_user_thread(user["id"], current_thread["id"], current_thread["title"], current_thread["messages"])

            # Render styled message
            render_assistant_message(assistant_msg, len(current_thread["messages"]) - 1)

    st.rerun()


# ==========================================
# 6. Main Application Loop
# ==========================================

def main():
    """Main application loop."""
    # 1. Initialize MySQL database schema on launch
    if "db_initialized" not in st.session_state:
        init_db()
        st.session_state.db_initialized = True

    # 2. Authentication Gatekeeper
    if "authenticated_user" not in st.session_state:
        render_auth_screen()
        return

    setup_page()
    init_session_state()

    settings = render_sidebar()
    current_thread = st.session_state.threads[st.session_state.current_thread_id]

    # Header
    st.markdown(f'<div class="main-header-title">🔬 {current_thread["title"]}</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="main-header-subtitle">Thread ID: <code>{current_thread["id"]}</code> &nbsp;|&nbsp; '
        f'Autonomous Research Agent with Live Web Citations & Visual Synthesis</div>',
        unsafe_allow_html=True
    )

    # Topic pills for empty conversations
    if not current_thread["messages"]:
        st.info("💡 Start a new research investigation below, or explore one of these subjects:")
        col_p1, col_p2, col_p3 = st.columns(3)
        if col_p1.button("🔋 Solid State EV Batteries", use_container_width=True):
            st.session_state.initial_prompt = "What is the commercialization timeline and energy density of solid state EV batteries in 2026?"
        if col_p2.button("🚀 GPT-6 Astra Features", use_container_width=True):
            st.session_state.initial_prompt = "What are the latest announcements and capabilities of OpenAI's GPT-6 Astra?"
        if col_p3.button("🔐 Post-Quantum Cryptography", use_container_width=True):
            st.session_state.initial_prompt = "What are the primary NIST Post-Quantum Cryptography algorithms and migration roadmaps?"

    # Render Chat History
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
