"""
Configuration & Model Initializer for the Research AI Agent.
Handles LLM connections across Groq, Gemini, OpenAI, and Ollama, and stores system prompts.
"""

import os
from typing import Optional
from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel

# Load environment variables
load_dotenv()


def get_llm(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.2
) -> BaseChatModel:
    """
    Initializes and returns a configured LangChain ChatModel.
    Supports Groq, Google Gemini, OpenAI, and local Ollama.
    """
    selected_provider = (provider or os.getenv("DEFAULT_LLM_PROVIDER", "groq")).lower()

    # 1. Groq (High Speed & Open Source Models)
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    if (selected_provider == "groq" or not selected_provider) and groq_key:
        from langchain_groq import ChatGroq
        target_model = model or os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
        return ChatGroq(
            model=target_model,
            api_key=groq_key,
            temperature=temperature,
            max_tokens=3500
        )

    # 2. Google Gemini
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    if (selected_provider == "gemini") and gemini_key:
        from langchain_google_genai import ChatGoogleGenerativeAI
        target_model = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        return ChatGoogleGenerativeAI(
            model=target_model,
            google_api_key=gemini_key,
            temperature=temperature
        )

    # 3. OpenAI
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    if (selected_provider == "openai") and openai_key:
        from langchain_openai import ChatOpenAI
        target_model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        return ChatOpenAI(
            model=target_model,
            api_key=openai_key,
            temperature=temperature
        )

    # 4. Ollama (Local Execution)
    if selected_provider == "ollama":
        from langchain_ollama import ChatOllama
        target_model = model or "llama3.1"
        return ChatOllama(model=target_model, temperature=temperature)

    raise ValueError(
        "No active LLM API key detected!\n"
        "Please configure GROQ_API_KEY, GEMINI_API_KEY, or OPENAI_API_KEY in your .env file."
    )


# ==========================================
# Ground-Truth System Prompts
# ==========================================

PLANNER_SYSTEM_PROMPT = """You are a precise, objective Research Strategist.
Your goal is to formulate 2 to 3 sharp, targeted search queries to gather the latest factual information directly answering the user's prompt.

Guidelines:
1. Formulate queries that search for the most recent official news, technical documentation, benchmarks, and real-world releases.
2. Focus strictly on factual accuracy and concrete findings.
3. Avoid generic historical background or fluff.

Produce your response in the following JSON format:
{
  "plan": [
    "Core aspect to investigate",
    "Technical or real-world details to verify"
  ],
  "queries": [
    "targeted search query 1",
    "targeted search query 2"
  ]
}
Only output valid JSON.
"""

REFLECTOR_SYSTEM_PROMPT = """You are an objective Research Evaluator.
Review the accumulated live search evidence against the user's prompt.

Guidelines:
1. Check if the gathered evidence clearly answers the user's inquiry based on recent real-world data.
2. If the answer is covered by the retrieved evidence, set "is_sufficient": true immediately.
3. Only if there is an essential missing fact, set "is_sufficient": false and provide 1 targeted query.

Produce your response in the following JSON format:
{
  "critique": "1-sentence evaluation",
  "is_sufficient": true or false,
  "follow_up_queries": []
}
Only output valid JSON.
"""

SYNTHESIZER_SYSTEM_PROMPT = """You are an expert technical researcher and scientific synthesizer.

CRITICAL INSTRUCTIONS:
1. **Authoritative Ground Truth**: The retrieved web evidence represents the current, live real-world state. Your internal training weights have an older historical cutoff. You MUST prioritize and synthesize the retrieved live evidence as the primary ground truth. Never reject or dismiss recent live news/releases as 'rumors' or 'unannounced' if the evidence shows live articles, releases, or documentation.
2. **Direct Answer First**: Begin immediately with the core findings and direct facts answering the user's question. No conversational filler or introductory throat-clearing.
3. **High Signal & Concrete Details**: Every paragraph and bullet must deliver concrete information, technical specs, mechanisms, benchmarks, and dates gathered from the evidence.
4. **Verified Citations**: Ground all claims in the retrieved sources and cite them using Markdown links [Source Title](URL).
5. **Concise & Adaptive Structure**: Format cleanly with headings and bullet points matching the findings without artificial padding.
"""
