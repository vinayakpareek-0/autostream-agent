"""LangGraph agent workflow — state, nodes, edges, and compiled graph."""

import json
from typing import Annotated
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict

from src.config import get_llm
from src.prompts import SYSTEM_PROMPT, INTENT_PROMPT, LEAD_EXTRACTION_PROMPT
from src.rag import get_retriever
from src.tools import mock_lead_capture


# ── State Schema ──────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    """Persisted state across conversation turns."""
    messages: Annotated[list[BaseMessage], add_messages]
    intent: str
    context: str
    lead_info: dict  # {name: str|None, email: str|None, platform: str|None}
    lead_captured: bool


# ── Shared Resources (initialized once) ──────────────────────────────────────

llm = get_llm()
retriever = get_retriever()


# ── Node Functions ────────────────────────────────────────────────────────────

def classify_intent(state: AgentState) -> dict:
    """Classify the latest user message into greeting / inquiry / high_intent."""
    messages = state["messages"]
    response = llm.invoke([
        SystemMessage(content=INTENT_PROMPT),
        *messages[-6:],  # last few turns for context
    ])
    intent = response.content.strip().lower()

    # Normalize to valid categories
    valid = {"greeting", "inquiry", "high_intent"}
    if intent not in valid:
        intent = "inquiry"  # safe fallback

    return {"intent": intent}


def retrieve_knowledge(state: AgentState) -> dict:
    """Fetch relevant chunks from FAISS based on the user's question."""
    latest_msg = state["messages"][-1].content
    docs = retriever.invoke(latest_msg)
    context = "\n".join(doc.page_content for doc in docs)
    return {"context": context}


def handle_lead(state: AgentState) -> dict:
    """Extract name, email, and platform from conversation history."""
    messages = state["messages"]
    response = llm.invoke([
        SystemMessage(content=LEAD_EXTRACTION_PROMPT),
        *messages[-8:],
    ])

    # Parse the LLM's JSON response
    try:
        extracted = json.loads(response.content)
    except json.JSONDecodeError:
        extracted = {"name": None, "email": None, "platform": None}

    # Merge with existing lead_info (keep previously collected fields)
    current = state.get("lead_info", {}) or {}
    lead_info = {
        "name": extracted.get("name") or current.get("name"),
        "email": extracted.get("email") or current.get("email"),
        "platform": extracted.get("platform") or current.get("platform"),
    }
    return {"lead_info": lead_info}


def capture_lead(state: AgentState) -> dict:
    """Call mock_lead_capture once all three fields are collected."""
    info = state["lead_info"]
    result = mock_lead_capture(info["name"], info["email"], info["platform"])
    return {
        "lead_captured": True,
        "context": result,  # pass to generate_response
    }


def generate_response(state: AgentState) -> dict:
    """Generate the agent's reply using full state context."""
    intent = state.get("intent", "greeting")
    context = state.get("context", "")
    lead_info = state.get("lead_info", {}) or {}
    lead_captured = state.get("lead_captured", False)

    # Build context-aware system message
    system_parts = [SYSTEM_PROMPT]

    if context:
        system_parts.append(f"\nRelevant information:\n{context}")

    if intent == "high_intent" and not lead_captured:
        missing = [k for k, v in lead_info.items() if not v]
        if missing:
            system_parts.append(
                f"\nThe user wants to sign up. Still need: {', '.join(missing)}. "
                "Ask for the missing details naturally."
            )

    if lead_captured:
        system_parts.append(
            "\nThe lead has been captured successfully! "
            "Thank the user and let them know the team will reach out soon."
        )

    response = llm.invoke([
        SystemMessage(content="\n".join(system_parts)),
        *state["messages"][-6:],
    ])
    return {"messages": [response]}


# ── Routing Logic ─────────────────────────────────────────────────────────────

def route_by_intent(state: AgentState) -> str:
    """Route to the appropriate handler based on classified intent."""
    return state["intent"]


def should_capture(state: AgentState) -> str:
    """Check if all lead info is collected → capture or ask for more."""
    info = state.get("lead_info", {}) or {}
    if all(info.get(k) for k in ("name", "email", "platform")):
        return "capture_lead"
    return "generate_response"


# ── Graph Assembly ────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """Assemble the LangGraph workflow."""
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("classify_intent", classify_intent)
    workflow.add_node("retrieve_knowledge", retrieve_knowledge)
    workflow.add_node("handle_lead", handle_lead)
    workflow.add_node("capture_lead", capture_lead)
    workflow.add_node("generate_response", generate_response)

    # Entry point
    workflow.set_entry_point("classify_intent")

    # Intent routing
    workflow.add_conditional_edges("classify_intent", route_by_intent, {
        "greeting": "generate_response",
        "inquiry": "retrieve_knowledge",
        "high_intent": "handle_lead",
    })

    # After RAG retrieval → generate response
    workflow.add_edge("retrieve_knowledge", "generate_response")

    # After lead handling → check if complete
    workflow.add_conditional_edges("handle_lead", should_capture, {
        "capture_lead": "capture_lead",
        "generate_response": "generate_response",
    })

    # After lead capture → generate response
    workflow.add_edge("capture_lead", "generate_response")

    # Response is always the final node
    workflow.add_edge("generate_response", END)

    return workflow


# ── Compiled Graph (with memory) ──────────────────────────────────────────────

memory = MemorySaver()
graph = build_graph().compile(checkpointer=memory)
