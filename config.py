"""Configuration, LLM initializers, and core prompt templates."""

import os
from typing import Optional
from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel

load_dotenv()


def get_config_val(key: str, default: str = "") -> str:
    """Gets value from environment or Streamlit secrets."""
    val = os.getenv(key)
    if val:
        return val.strip()
    try:
        import streamlit as st
        if hasattr(st, "secrets") and key in st.secrets:
            return str(st.secrets[key]).strip()
    except Exception:
        pass
    return default


def get_llm(provider: Optional[str] = None, model: Optional[str] = None, temperature: float = 0.2) -> BaseChatModel:
    """Initializes and returns LangChain ChatModel (Groq, Gemini, OpenAI, Ollama)."""
    selected = (provider or get_config_val("DEFAULT_LLM_PROVIDER", "groq")).lower()

    if selected in ("groq", "") and (k := get_config_val("GROQ_API_KEY")):
        from langchain_groq import ChatGroq
        return ChatGroq(model=model or get_config_val("GROQ_MODEL", "openai/gpt-oss-120b"), api_key=k, temperature=temperature, max_tokens=3500)

    if selected == "gemini" and (k := get_config_val("GEMINI_API_KEY")):
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model or get_config_val("GEMINI_MODEL", "gemini-2.5-flash"), google_api_key=k, temperature=temperature)

    if selected == "openai" and (k := get_config_val("OPENAI_API_KEY")):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model or get_config_val("OPENAI_MODEL", "gpt-4o-mini"), api_key=k, temperature=temperature)

    if selected == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model or "llama3.1", temperature=temperature)

    raise ValueError("No active LLM API key detected! Set GROQ_API_KEY in Secrets or sidebar.")


PLANNER_SYSTEM_PROMPT = """You are an objective Research Strategist. Formulate 2-3 search queries directly answering the user's prompt.
Translate colloquial queries (e.g. 'loops in system', 'tricks') into objective research queries (e.g. legal incentives, tax frameworks, market mechanics).
Produce ONLY valid JSON:
{
  "plan": ["aspect 1", "aspect 2"],
  "queries": ["targeted search query 1", "targeted search query 2"]
}"""

REFLECTOR_SYSTEM_PROMPT = """You are a Research Evaluator. Review evidence against the topic.
If covered, set "is_sufficient": true. If missing vital facts, set false and provide 1 query.
Produce ONLY valid JSON:
{
  "critique": "1-sentence evaluation",
  "is_sufficient": true or false,
  "follow_up_queries": []
}"""

SYNTHESIZER_SYSTEM_PROMPT = """You are an expert scientific researcher.
1. Ground Truth: Prioritize retrieved live evidence over older internal weights.
2. Objective Perspective: Analyze legal frameworks, technical systems, economics, and public regulations factually.
3. Direct Findings: Begin immediately with concrete specs, numbers, and facts. No filler.
4. Citations: Link claims with Markdown sources [Source Title](URL)."""
