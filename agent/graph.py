"""LangGraph agent orchestration — the core agentic pipeline.

Implements a **6-node stateful graph** for the Social-to-Lead workflow:

  1. input_node    — Preprocesses user input and records turn metadata
  2. intent_node   — Classifies user intent via LLM + rule-based fallback
  3. rag_node      — Retrieves relevant knowledge base context (RAG)
  4. lead_node     — Extracts and validates lead fields from conversation
  5. tool_node     — Executes mock_lead_capture ONLY when fully qualified
  6. respond_node  — Generates the final grounded response

Conditional edges route the flow based on detected intent:
  - GREETING      → respond_node  (skip RAG and lead collection)
  - PRODUCT_QUERY → rag_node → respond_node
  - HIGH_INTENT   → rag_node → lead_node → [tool_node] → respond_node

Architecture principles:
  - Every response is grounded in RAG context (no hallucination)
  - Tool execution is gated behind `should_trigger_tool()` checks
  - State is immutable between nodes; each node returns a partial update
  - The graph is channel-agnostic: same `agent_app.invoke()` for UI + webhooks
"""

from __future__ import annotations

import logging
import os
import re

import dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph

from agent.intent import (
    classify_intent_from_text,
    extract_email,
    extract_name,
    extract_platform,
)
from agent.lead import (
    compute_missing_fields,
    is_fully_qualified,
    next_field_prompt,
    should_trigger_tool,
)
from agent.rag import retrieve_knowledge
from agent.state import AgentState
from agent.tools import mock_lead_capture

dotenv.load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM initialisation
# ---------------------------------------------------------------------------

MODEL_NAME = os.getenv("AUTOSTREAM_MODEL", "gemini-3.1-flash-lite-preview")

llm: ChatGoogleGenerativeAI | None = None
try:
    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        temperature=0.3,
        max_output_tokens=1024,
        convert_system_message_to_human=True,
    )
    logger.info("LLM initialised: %s", MODEL_NAME)
except Exception as exc:
    logger.warning("LLM initialisation failed: %s — using deterministic fallback", exc)

# ---------------------------------------------------------------------------
# System prompt (injected into every LLM call for response generation)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are the AutoStream Premium AI Sales Concierge.

AutoStream is a world-class SaaS platform providing AI-driven video automation
for top-tier content creators. Your role is to provide sophisticated, 
accurate, and helpful assistance to potential clients.

OPERATIONAL GUIDELINES:
1. **Persona**: Professional, knowledgeable, and concise. No filler words.
2. **Knowledge**: Use ONLY the provided Knowledge Base Context for product-specific 
   queries (pricing, features, limits, policies). NEVER invent information.
3. **Integrity**: If a user asks something outside AutoStream's scope, 
   politely pivot back to product value.
4. **Lead Capture**: For high-intent users, guide them through collecting 
   Full Name → Professional Email → Primary Creator Platform. One field at a time.
5. **Formatting**: Use Markdown for clarity. Bold key terms. Use bullet points 
   for lists.
6. **Conciseness**: Keep responses under 3 paragraphs unless a detailed 
   comparison is requested.
7. **Grounding**: NEVER fabricate pricing, features, or policy details. 
   If the context doesn't contain the answer, say so honestly.
"""


# =====================================================================
# Node 1: Input Preprocessing
# =====================================================================

def input_node(state: AgentState) -> dict:
    """Preprocess user input — no-op for now but extensible.

    In production this node would handle:
    - Input sanitisation / PII redaction
    - Language detection
    - Spam/abuse filtering
    """
    return {}


# =====================================================================
# Node 2: Intent Classification
# =====================================================================

def intent_node(state: AgentState) -> dict:
    """Classify the user's intent from their latest message.

    Strategy:
    1. If we are in lead_collect stage → maintain HIGH_INTENT
    2. If lead is already captured → use PRODUCT_QUERY for follow-ups
    3. Try LLM classification with structured prompt
    4. Fall back to rule-based classifier
    """
    messages = state["messages"]
    if not messages:
        return {"intent": "GREETING", "conversation_stage": "greeting"}

    last_msg = messages[-1].content

    # Maintain HIGH_INTENT during active lead collection
    if state.get("conversation_stage") == "lead_collect":
        return {"intent": "HIGH_INTENT"}

    # Post-capture: treat all follow-ups as product queries
    if state.get("is_tool_called"):
        return {"intent": "PRODUCT_QUERY"}

    # Build recent conversation context for better classification
    recent_context = ""
    if len(messages) > 1:
        recent_msgs = messages[-4:]  # Last 4 messages for context
        context_lines = []
        for m in recent_msgs:
            role = "User" if isinstance(m, HumanMessage) else "Assistant"
            context_lines.append(f"{role}: {m.content[:100]}")
        recent_context = "\n".join(context_lines)

    # LLM-based classification
    if llm is not None:
        try:
            classification_prompt = f"""Classify this user message into EXACTLY one category.
Reply with ONLY the category name — no explanation, no punctuation, just the word.

Categories:
- GREETING: casual hello, hi, greetings, how are you, general acknowledgment like "yes", "sure", "okay" when no specific product intent is shown
- PRODUCT_QUERY: asking about pricing, features, plans, policies, support, comparisons, or any product-related question
- HIGH_INTENT: EXPLICITLY wants to sign up, purchase, subscribe, buy, start a plan, or CLEARLY states readiness to commit with specific action words

IMPORTANT: Only classify as HIGH_INTENT if the user EXPLICITLY mentions signing up, buying, subscribing, or starting. Generic affirmations like "yes", "sure", "okay" are NOT high intent unless they are directly responding to a subscription offer.

{f"Recent conversation context:{chr(10)}{recent_context}{chr(10)}" if recent_context else ""}
User message to classify: "{last_msg}"

Category:"""
            response = llm.invoke(classification_prompt)
            intent_raw = response.content.strip().upper()

            # Extract valid intent from response
            for valid in ["HIGH_INTENT", "PRODUCT_QUERY", "GREETING"]:
                if valid in intent_raw:
                    intent = valid
                    break
            else:
                intent = classify_intent_from_text(last_msg)

            # Update conversation stage
            stage = _intent_to_stage(intent, state)
            return {"intent": intent, "conversation_stage": stage}

        except Exception as exc:
            logger.warning("LLM intent classification failed: %s", exc)

    # Rule-based fallback
    intent = classify_intent_from_text(last_msg)
    stage = _intent_to_stage(intent, state)
    return {"intent": intent, "conversation_stage": stage}


def _intent_to_stage(intent: str, state: AgentState) -> str:
    """Map an intent to the appropriate conversation funnel stage."""
    if state.get("is_tool_called"):
        return "captured"
    if intent == "GREETING":
        return "greeting"
    if intent == "HIGH_INTENT":
        return "lead_collect"
    return "inquiry"


# =====================================================================
# Node 3: RAG Knowledge Retrieval
# =====================================================================

def rag_node(state: AgentState) -> dict:
    """Retrieve relevant context from the knowledge base.

    Stores the retrieved context in state['rag_context'] so that
    the respond_node can ground its answer without re-querying.
    """
    messages = state["messages"]
    if not messages:
        return {"rag_context": ""}

    query = messages[-1].content
    context = retrieve_knowledge(query)

    logger.debug("RAG retrieved %d chars for query: %s", len(context), query[:80])
    return {"rag_context": context}


# =====================================================================
# Node 4: Lead Field Extraction
# =====================================================================

def lead_node(state: AgentState) -> dict:
    """Extract and validate lead fields from the conversation.

    Processes the latest user message to detect:
    - Platform mentions (YouTube, Instagram, etc.)
    - Email addresses
    - Name patterns ("My name is X", or short text during name collection)

    Updates missing_fields and checks if qualification criteria are met.
    """
    messages = state["messages"]
    if not messages:
        return {}

    last_msg = messages[-1].content
    updates: dict = {}

    current_name = state.get("lead_name")
    current_email = state.get("lead_email")
    current_platform = state.get("lead_platform")
    stage = state.get("conversation_stage", "")
    missing = state.get("missing_fields", [])

    # --- Extract platform ---
    if not current_platform:
        platform = extract_platform(last_msg)
        if platform:
            updates["lead_platform"] = platform
            current_platform = platform

    # --- Extract email ---
    if not current_email:
        email = extract_email(last_msg)
        if email:
            updates["lead_email"] = email
            current_email = email

    # --- Extract name ---
    if not current_name:
        name = extract_name(last_msg, stage=stage, missing=missing)
        if name:
            updates["lead_name"] = name
            current_name = name

    # --- Recompute missing fields ---
    updates["missing_fields"] = compute_missing_fields(
        current_name, current_email, current_platform
    )

    # --- Update qualification status ---
    if is_fully_qualified(current_name, current_email, current_platform):
        updates["is_qualified"] = True
        updates["conversation_stage"] = "lead_collect"  # tool_node will set to captured
    else:
        updates["conversation_stage"] = "lead_collect"

    return updates


# =====================================================================
# Node 5: Tool Execution (Lead Capture)
# =====================================================================

def tool_node(state: AgentState) -> dict:
    """Execute the lead capture tool — ONLY if all conditions are met.

    Gate conditions (all must be True):
    1. lead_name is present
    2. lead_email is a valid email
    3. lead_platform is present
    4. is_tool_called is False (not already captured)

    This is a STRICT gate — premature execution is never allowed.
    """
    if not should_trigger_tool(state):
        return {}

    name = state["lead_name"]
    email = state["lead_email"]
    platform = state["lead_platform"]

    try:
        result = mock_lead_capture(name, email, platform)
        logger.info("Tool executed: %s", result.get("message", ""))
        return {
            "is_tool_called": True,
            "conversation_stage": "captured",
        }
    except Exception as exc:
        logger.error("Tool execution failed: %s", exc)
        return {}


# =====================================================================
# Node 6: Response Generation
# =====================================================================

def respond_node(state: AgentState) -> dict:
    """Generate the agent's response using full context.

    Response strategy (in priority order):
    1. Lead just captured → confirmation message
    2. High intent, collecting fields → prompt for next missing field
    3. LLM generation grounded in RAG context
    4. Deterministic fallback responses
    """
    messages = state["messages"]
    intent = state.get("intent", "GREETING")
    rag_context = state.get("rag_context", "")

    # --- Strategy 1: Lead capture confirmation ---
    if state.get("is_tool_called") and state.get("conversation_stage") == "captured":
        name = state.get("lead_name", "")
        email = state.get("lead_email", "")
        platform = state.get("lead_platform", "")

        # Check if this is the turn where we just captured (not a follow-up)
        missing = state.get("missing_fields", [])
        if not missing and name and email and platform:
            response_text = (
                f"🎉 **Your lead has been captured successfully!**\n\n"
                f"**Lead Summary:**\n"
                f"- **Name:** {name}\n"
                f"- **Email:** {email}\n"
                f"- **Platform:** {platform}\n"
                f"- **Plan Interest:** Pro Plan\n\n"
                f"Welcome aboard, **{name}**! Our team will reach out to you at "
                f"**{email}** within 24 hours to help you get started with "
                f"AutoStream Pro. In the meantime, feel free to ask any other questions!"
            )
            return {"messages": [AIMessage(content=response_text)]}

    # --- Strategy 2: Greetings — always deterministic (NEVER through LLM/RAG) ---
    if intent == "GREETING":
        response_text = (
            "Hello! 👋 I'm the **AutoStream AI Assistant**, powered by Gemini 3.1 Flash-Lite.\n\n"
            "I can help you with:\n"
            "- 📋 **Pricing & Plans** — Compare our Basic and Pro plans\n"
            "- ✨ **Features** — AI captions, 4K export, team collaboration\n"
            "- 📄 **Policies** — Refunds, support, cancellation\n"
            "- 🚀 **Getting Started** — Sign up for AutoStream\n\n"
            "What would you like to know?"
        )
        return {"messages": [AIMessage(content=response_text)]}

    # --- Strategy 3: Prompt for next missing field ---
    if intent == "HIGH_INTENT" and not state.get("is_tool_called"):
        missing = state.get("missing_fields", ["name", "email", "platform"])
        if missing:
            prompt_text = next_field_prompt(missing)
            return {"messages": [AIMessage(content=prompt_text)]}

    # --- Strategy 4: LLM-grounded response (PRODUCT_QUERY only) ---
    if llm is not None and intent == "PRODUCT_QUERY":
        try:
            # Build conversation history for context window
            history_lines = []
            for msg in messages[-10:]:
                role = "User" if isinstance(msg, HumanMessage) else "Assistant"
                history_lines.append(f"{role}: {msg.content}")
            history_text = "\n".join(history_lines)

            # Ensure we have RAG context
            if not rag_context and messages:
                rag_context = retrieve_knowledge(messages[-1].content)

            prompt = f"""{SYSTEM_PROMPT}

KNOWLEDGE BASE CONTEXT (GROUND TRUTH — use ONLY this for product facts):
{rag_context}

CONVERSATION HISTORY:
{history_text}

CURRENT INTENT: {intent}

INSTRUCTIONS:
1. Answer the user's question using ONLY the Knowledge Base Context above.
2. If the context does not contain the answer, say "I don't have that specific information, but I can tell you about our plans, pricing, and policies."
3. NEVER invent pricing, features, or policy details.
4. Always mention BOTH the Basic Plan ($29/mo) and Pro Plan ($79/mo) when discussing pricing.
5. Be professional, friendly, and structured. Use bullet points for lists.
6. Keep the response concise (under 3 paragraphs).

Response:"""
            response = llm.invoke(prompt)
            return {"messages": [AIMessage(content=response.content)]}
        except Exception as exc:
            logger.warning("LLM response generation failed: %s", exc)

    # --- Strategy 5: Deterministic fallback ---
    if intent == "PRODUCT_QUERY":
        if rag_context and "No relevant information" not in rag_context:
            response_text = (
                "Based on our knowledge base:\n\n"
                f"{rag_context}\n\n"
                "Would you like more specific details on any of these points?"
            )
        else:
            response_text = (
                "Great question! AutoStream offers two plans:\n\n"
                "**Basic Plan** — **$29/month**\n"
                "- 10 videos per month\n"
                "- 720p resolution\n"
                "- Email support (business hours)\n"
                "- 10 GB storage\n\n"
                "**Pro Plan** — **$79/month**\n"
                "- Unlimited videos\n"
                "- 4K resolution\n"
                "- AI captions (multi-language)\n"
                "- 24/7 priority support\n"
                "- Unlimited storage\n"
                "- Custom branding & team collaboration\n\n"
                "Which plan would you like to know more about?"
            )
    else:
        response_text = (
            "I'd love to help you get started with AutoStream! We offer:\n\n"
            "- **Basic Plan** ($29/mo) — Great for getting started\n"
            "- **Pro Plan** ($79/mo) — Full-featured for serious creators\n\n"
            "Which one interests you most?"
        )

    return {"messages": [AIMessage(content=response_text)]}


# =====================================================================
# Conditional routing functions
# =====================================================================

def route_after_intent(state: AgentState) -> str:
    """Decide which node to visit after intent classification.

    Routing rules:
    - GREETING      → respond  (skip RAG, no product info needed)
    - PRODUCT_QUERY → rag      (need knowledge base context)
    - HIGH_INTENT   → rag      (retrieve context before lead flow)
    """
    intent = state.get("intent", "GREETING")
    if intent == "GREETING":
        return "respond"
    return "rag"  # Both PRODUCT_QUERY and HIGH_INTENT go through RAG


def route_after_rag(state: AgentState) -> str:
    """Decide whether to go to lead collection or direct response.

    - HIGH_INTENT + not captured → lead (extract/collect fields)
    - Otherwise                  → respond (answer the query)
    """
    intent = state.get("intent", "GREETING")
    if intent == "HIGH_INTENT" and not state.get("is_tool_called"):
        return "lead"
    return "respond"


def route_after_lead(state: AgentState) -> str:
    """Decide whether to execute the tool or go to response.

    - Fully qualified + not captured → tool (execute lead capture)
    - Otherwise                      → respond (ask for missing field)
    """
    if should_trigger_tool(state):
        return "tool"
    return "respond"


# =====================================================================
# Graph assembly
# =====================================================================

def build_agent_graph() -> StateGraph:
    """Construct and compile the LangGraph state machine.

    Node topology:
        input → intent → [rag] → [lead] → [tool] → respond → END
    """
    graph = StateGraph(AgentState)

    # Register all 6 nodes
    graph.add_node("input", input_node)
    graph.add_node("intent", intent_node)
    graph.add_node("rag", rag_node)
    graph.add_node("lead", lead_node)
    graph.add_node("tool", tool_node)
    graph.add_node("respond", respond_node)

    # Entry point
    graph.set_entry_point("input")

    # input → intent (always)
    graph.add_edge("input", "intent")

    # intent → rag OR respond
    graph.add_conditional_edges(
        "intent",
        route_after_intent,
        {
            "respond": "respond",
            "rag": "rag",
        },
    )

    # rag → lead OR respond
    graph.add_conditional_edges(
        "rag",
        route_after_rag,
        {
            "lead": "lead",
            "respond": "respond",
        },
    )

    # lead → tool OR respond
    graph.add_conditional_edges(
        "lead",
        route_after_lead,
        {
            "tool": "tool",
            "respond": "respond",
        },
    )

    # tool → respond (always)
    graph.add_edge("tool", "respond")

    # respond → END
    graph.add_edge("respond", END)

    return graph


# Compile the graph at module level for reuse
agent_app = build_agent_graph().compile()
