"""Configuration for the multi-agent system."""

import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_groq import ChatGroq

env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# Optional import for OpenAI (used for researcher tool calling)
OPENAI_AVAILABLE = False

try:
    from langchain_openai import ChatOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    pass


def get_llm() -> BaseChatModel:
    """Initialize and return the Groq LLM."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found. Please set it in a .env file or as an environment variable.\n"
            "Create a .env file in the project root with: GROQ_API_KEY=your-api-key-here"
        )
    
    model_name = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    temperature = float(os.getenv("GROQ_TEMPERATURE", "0.7"))
    
    return ChatGroq(
        model_name=model_name,
        groq_api_key=api_key,
        temperature=temperature,
    )


def get_judge_llm() -> BaseChatModel:
    """Initialize and return the Groq LLM for evaluation (judge_llm)."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found. Please set it in a .env file or as an environment variable.\n"
            "Create a .env file in the project root with: GROQ_API_KEY=your-api-key-here"
        )
    
    model_name = os.getenv("GROQ_JUDGE_MODEL", "llama-3.3-70b-versatile")
    temperature = float(os.getenv("GROQ_JUDGE_TEMPERATURE", "0.3"))  # Lower temp for evaluation
    
    return ChatGroq(
        model_name=model_name,
        groq_api_key=api_key,
        temperature=temperature,
    )


def get_researcher_llm() -> BaseChatModel:
    """Initialize and return the OpenAI LLM used by the researcher for tool calling."""
    if not OPENAI_AVAILABLE:
        raise ImportError(
            "langchain-openai is not installed. Install dependencies from requirements.txt."
        )

    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise ValueError(
            "OPENAI_API_KEY not found. The researcher requires OpenAI for tool calling."
        )

    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
    return ChatOpenAI(
        model=model_name,
        api_key=openai_key,
        temperature=temperature,
    )

