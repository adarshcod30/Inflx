"""AutoStream Agentic Workflow — Core Agent Package.

This package implements a 6-node LangGraph pipeline for the
Social-to-Lead Agentic Workflow:

  Modules:
    - state.py   : Typed AgentState schema and defaults
    - intent.py  : Intent classification (LLM + rule-based fallback)
    - rag.py     : RAG retrieval from JSON/Markdown knowledge base
    - lead.py    : Lead qualification logic and field tracking
    - tools.py   : Mock lead capture CRM tool
    - graph.py   : LangGraph orchestration and conditional routing
"""
