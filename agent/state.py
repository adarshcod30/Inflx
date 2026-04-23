"""Typed state schema for the AutoStream LangGraph agent.

The AgentState TypedDict defines every field that flows through the graph.
LangGraph nodes read and write specific keys; downstream nodes can
always assume keys exist because `default_state()` initializes them all.

State Categories:
  - Conversation  : message history (multi-turn memory)
  - Intent        : classified user intent for the current turn
  - Workflow      : funnel stage tracking (greeting → inquiry → lead_collect → captured)
  - Lead Data     : incrementally-collected lead fields
  - RAG Context   : retrieved knowledge snippets for grounding
  - Tool Gate     : boolean guard to prevent premature/duplicate tool execution
"""

from typing import TypedDict

from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """Immutable-shape state flowing through every LangGraph node.

    Every key MUST be initialised by `default_state()` before the
    first graph invocation so that nodes can read without KeyError.
    """

    # -- Conversation memory (last N turns) --
    messages: list[BaseMessage]

    # -- Intent classification --
    intent: str                    # GREETING | PRODUCT_QUERY | HIGH_INTENT

    # -- Workflow funnel stage --
    conversation_stage: str        # greeting | inquiry | lead_collect | captured

    # -- Lead qualification --
    is_qualified: bool
    lead_name: str | None
    lead_email: str | None
    lead_platform: str | None
    missing_fields: list[str]      # ordered list of fields still needed

    # -- RAG context (populated by rag_node) --
    rag_context: str

    # -- Tool execution gate --
    is_tool_called: bool


def default_state() -> dict:
    """Return a clean initial state dict with safe defaults.

    Called once per session to seed Streamlit session_state and
    as the baseline for every new graph invocation.
    """
    return {
        "messages": [],
        "intent": "GREETING",
        "conversation_stage": "greeting",
        "is_qualified": False,
        "lead_name": None,
        "lead_email": None,
        "lead_platform": None,
        "missing_fields": [],
        "rag_context": "",
        "is_tool_called": False,
    }
