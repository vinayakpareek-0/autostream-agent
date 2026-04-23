"""Centralized configuration -- loads env vars, initializes LLM and embeddings."""

import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")


def get_llm():
    """Initialize LLM based on LLM_PROVIDER env var (groq or gemini)."""
    if LLM_PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.3,
        )

    from langchain_groq import ChatGroq
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.3,
    )


def get_embeddings() -> HuggingFaceEmbeddings:
    """Initialize local HuggingFace embeddings (no API cost)."""
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
    )

