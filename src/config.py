"""Centralized configuration — loads env vars, initializes LLM and embeddings."""

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()


def get_llm() -> ChatGoogleGenerativeAI:
    """Initialize Gemini 1.5 Flash with sensible defaults."""
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.3,
    )


def get_embeddings() -> HuggingFaceEmbeddings:
    """Initialize local HuggingFace embeddings (no API cost)."""
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
    )
