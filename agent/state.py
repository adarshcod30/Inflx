"""Typed state schema for the AutoStream LangGraph agent.

The AgentState TypedDict defines every field that flows through the graph.
LangGraph nodes read and write specific keys; downstream nodes can
always assume keys exist because `default_state()` initializes them all.
"""

from typing import TypedDict

from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """Immutable-shape state flowing through every LangGraph node."""

    # -- Conversation --
    messages: list[BaseMessage]

    # -- Intent --
    intent: str                    # GREETING | PRODUCT_QUERY | HIGH_INTENT

    # -- Workflow stage --
    conversation_stage: str        # greeting | inquiry | lead_collect | captured

    # -- Lead qualification --
    is_qualified: bool
    lead_name: str | None
    lead_email: str | None
    lead_platform: str | None
    missing_fields: list[str]

    # -- Tool gate --
    is_tool_called: bool


def default_state() -> dict:
    """Return a clean initial state dict with safe defaults."""
    return {
        "messages": [],
        "intent": "GREETING",
        "conversation_stage": "greeting",
        "is_qualified": False,
        "lead_name": None,
        "lead_email": None,
        "lead_platform": None,
        "missing_fields": [],
        "is_tool_called": False,
    }
