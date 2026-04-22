"""RAG pipeline — loads knowledge base JSON, builds FAISS index, exposes retriever."""

import json
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever
from src.config import get_embeddings

KNOWLEDGE_BASE_PATH = Path(__file__).parent.parent / "knowledge_base" / "autostream.json"


def _load_documents() -> list[Document]:
    """Convert each knowledge base entry into a LangChain Document."""
    with open(KNOWLEDGE_BASE_PATH, "r") as f:
        data = json.load(f)

    documents: list[Document] = []

    # Pricing plans → one document per plan
    for plan in data["pricing"]:
        content = (
            f"AutoStream {plan['plan']} Plan: {plan['price']}. "
            f"Features: {', '.join(plan['features'])}."
        )
        documents.append(Document(page_content=content, metadata={"source": "pricing"}))

    # Policies → one document per policy
    for policy in data["policies"]:
        content = f"AutoStream Policy — {policy['topic']}: {policy['detail']}"
        documents.append(Document(page_content=content, metadata={"source": "policy"}))

    return documents


def get_retriever(top_k: int = 2) -> VectorStoreRetriever:
    """Build FAISS index from knowledge base and return a retriever."""
    documents = _load_documents()
    vectorstore = FAISS.from_documents(documents, get_embeddings())
    return vectorstore.as_retriever(search_kwargs={"k": top_k})
