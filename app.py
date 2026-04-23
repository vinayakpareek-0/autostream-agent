"""Streamlit chat UI for the AutoStream AI agent."""

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessageChunk
from src.graph import graph


# ── Page Config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AutoStream AI Assistant",
    page_icon="🎬",
    layout="centered",
)

st.title("🎬 AutoStream AI Assistant")
st.caption("Your AI-powered video editing companion — ask about plans, pricing, or get started!")


# ── Session State ─────────────────────────────────────────────────────────────

if "thread_id" not in st.session_state:
    st.session_state.thread_id = "streamlit-session"

if "messages" not in st.session_state:
    st.session_state.messages = []


# ── Render Chat History ───────────────────────────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Re-render activity panel for assistant messages
        if msg["role"] == "assistant" and "activity" in msg:
            with st.expander("⚙️ Agent Activity", expanded=False):
                st.markdown(msg["activity"])


# ── Handle User Input ────────────────────────────────────────────────────────

if user_input := st.chat_input("Type your message..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Stream the agent response token by token
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    with st.chat_message("assistant"):
        response_container = st.empty()
        full_response = ""
        nodes_visited = []

        for msg, metadata in graph.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
            stream_mode="messages",
        ):
            # Track which nodes are executing
            node = metadata.get("langgraph_node", "")
            if node and node not in nodes_visited:
                nodes_visited.append(node)

            # Only stream tokens from the final response node
            if (
                isinstance(msg, AIMessageChunk)
                and node == "generate_response"
                and msg.content
            ):
                full_response += msg.content
                response_container.markdown(full_response + "▌")

        response_container.markdown(full_response)

        # Show agent activity panel
        state = graph.get_state(config).values
        intent = state.get("intent", "—")
        lead_info = state.get("lead_info") or {}
        lead_captured = state.get("lead_captured", False)

        intent_label = {
            "greeting": "👋 Greeting",
            "inquiry": "❓ Product Inquiry",
            "high_intent": "🎯 High Intent (Lead)",
        }.get(intent, intent)

        activity_lines = [
            f"**Intent:** {intent_label}",
            f"**Pipeline:** `{'` → `'.join(nodes_visited)}`",
        ]

        if lead_info and any(lead_info.values()):
            info_parts = [f"{k}: {v}" for k, v in lead_info.items() if v]
            activity_lines.append(f"**Lead Info:** {', '.join(info_parts)}")

        if lead_captured:
            activity_lines.append("**Lead Captured:** ✅ Yes")

        activity_md = "\n\n".join(activity_lines)
        with st.expander("⚙️ Agent Activity", expanded=True):
            st.markdown(activity_md)

    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response,
        "activity": activity_md,
    })

