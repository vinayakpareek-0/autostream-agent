# AutoStream — Social-to-Lead AI Agent

A conversational AI agent for **AutoStream** (automated video editing SaaS) that identifies user intent, answers product questions via RAG, and captures high-intent leads.

## Tech Stack

- **Framework:** LangGraph (stateful agent workflow)
- **LLM:** Gemini 1.5 Flash (Google AI Studio — free tier)
- **Embeddings:** HuggingFace `all-MiniLM-L6-v2` (local, free)
- **Vector Store:** FAISS (in-memory, fast)
- **UI:** Streamlit chat interface
- **Language:** Python 3.13

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
cp .env.example .env         # Add your Google API key
```

## Project Structure

```
├── knowledge_base/          # Product data (JSON)
├── src/
│   ├── config.py            # Environment & model initialization
│   ├── prompts.py           # System prompts & templates
│   ├── rag.py               # RAG pipeline (FAISS + HuggingFace)
│   ├── tools.py             # Lead capture tool
│   └── graph.py             # LangGraph workflow (state, nodes, edges)
├── main.py                  # CLI entry point
├── app.py                   # Streamlit chat UI
└── requirements.txt
```
