"""LangGraph agent orchestration — the core agentic pipeline.

Implements a 4-node stateful graph:
  1. intent_node   — Classifies user intent via LLM + rule fallback
  2. rag_node      — Retrieves relevant knowledge base context
  3. lead_node     — Extracts and validates lead fields from conversation
  4. respond_node  — Generates the final response using full context

Conditional edges route the flow based on detected intent:
  - GREETING     -> respond_node (skip RAG and lead collection)
  - PRODUCT_QUERY -> rag_node -> respond_node
  - HIGH_INTENT  -> rag_node -> lead_node -> respond_node
"""

import json
import logging
import os
import re

import dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph

from agent.intent import (
    classify_intent_from_text,
    extract_email,
    extract_platform,
)
from agent.rag import get_full_knowledge_base, retrieve_knowledge
from agent.state import AgentState
from agent.tools import mock_lead_capture

dotenv.load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM initialization
# ---------------------------------------------------------------------------

MODEL_NAME = os.getenv("AUTOSTREAM_MODEL", "gemini-2.0-flash-lite")

llm: ChatGoogleGenerativeAI | None = None
try:
    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        temperature=0.3,
        max_output_tokens=1024,
        convert_system_message_to_human=True,
    )
except Exception as exc:
    logger.warning("LLM initialization failed: %s — using deterministic fallback", exc)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are the AutoStream Premium AI Sales Concierge.

AutoStream is a world-class SaaS platform providing AI-driven video automation
for top-tier content creators. Your role is to provide sophisticated, 
accurate, and helpful assistance to potential clients.

OPERATIONAL GUIDELINES:
1. **Persona**: Professional, elite, and highly efficient. Avoid filler words.
2. **Knowledge**: Use ONLY the provided Ground Truth context for product-specific 
   queries (pricing, features, limits).
3. **Integrity**: If a user asks something outside the scope of AutoStream, 
   politefully pivot back to product value.
4. **Lead Capture**: For high-intent users, collect 'Full Name', 'Professional Email', 
   and 'Primary Creator Platform'. One field at a time.
5. **Formatting**: Use Markdown for clarity. Bold key terms. Use bullet points 
   for lists.
6. **Conciseness**: Keep responses under 3 paragraphs unless asked for a 
   detailed comparison.
"""


# ---------------------------------------------------------------------------
# Node: Intent Classification
# ---------------------------------------------------------------------------

def intent_node(state: AgentState) -> dict:
    """Classify the user's intent from their latest message."""
    messages = state["messages"]
    if not messages:
        return {"intent": "GREETING"}

    last_msg = messages[-1].content

    # If we are already in lead_collect stage, maintain HIGH_INTENT
    if state.get("conversation_stage") == "lead_collect":
        return {"intent": "HIGH_INTENT"}

    # If lead is already captured, use PRODUCT_QUERY for follow-ups
    if state.get("is_tool_called"):
        return {"intent": "PRODUCT_QUERY"}

    # Try LLM classification first
    if llm is not None:
        try:
            classification_prompt = f"""Classify this user message into exactly one category.
Reply with ONLY the category name, nothing else.

Categories:
- GREETING: casual hello, hi, greetings
- PRODUCT_QUERY: asking about pricing, features, plans, policies, support, or any product question
- HIGH_INTENT: wants to sign up, purchase, subscribe, try, start, buy, or indicates readiness to commit

User message: "{last_msg}"

Category:"""
            response = llm.invoke(classification_prompt)
            intent = response.content.strip().upper()

            # Clean up response
            for valid in ["HIGH_INTENT", "PRODUCT_QUERY", "GREETING"]:
                if valid in intent:
                    intent = valid
                    break
            else:
                intent = classify_intent_from_text(last_msg)

            return {"intent": intent}
        except Exception as exc:
            logger.warning("LLM intent classification failed: %s", exc)

    # Fallback to rule-based
    return {"intent": classify_intent_from_text(last_msg)}


# ---------------------------------------------------------------------------
# Node: RAG Knowledge Retrieval
# ---------------------------------------------------------------------------

def rag_node(state: AgentState) -> dict:
    """Retrieve relevant context from the knowledge base."""
    messages = state["messages"]
    if not messages:
        return {}

    query = messages[-1].content
    _ = retrieve_knowledge(query)

    # We store context retrieval in state for the respond node
    # The respond node will call retrieve_knowledge itself
    return {}


# ---------------------------------------------------------------------------
# Node: Lead Field Extraction
# ---------------------------------------------------------------------------

def lead_node(state: AgentState) -> dict:
    """Extract and validate lead fields from the conversation."""
    messages = state["messages"]
    if not messages:
        return {}

    last_msg = messages[-1].content
    updates: dict = {}

    current_name = state.get("lead_name")
    current_email = state.get("lead_email")
    current_platform = state.get("lead_platform")

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
        name = _extract_name_from_text(last_msg, state)
        if name:
            updates["lead_name"] = name
            current_name = name

    # --- Compute missing fields ---
    missing = []
    if not current_name:
        missing.append("name")
    if not current_email:
        missing.append("email")
    if not current_platform:
        missing.append("platform")
    updates["missing_fields"] = missing

    # --- Check if fully qualified ---
    if current_name and current_email and current_platform:
        updates["is_qualified"] = True
        if not state.get("is_tool_called"):
            result = mock_lead_capture(current_name, current_email, current_platform)
            updates["is_tool_called"] = True
            updates["conversation_stage"] = "captured"
    else:
        updates["conversation_stage"] = "lead_collect"

    return updates


def _extract_name_from_text(text: str, state: AgentState) -> str | None:
    """Heuristic name extraction from user message."""
    text_clean = text.strip()

    # Pattern: "My name is X" or "I am X" or "I'm X"
    patterns = [
        r"(?:my name is|i am|i'm|this is|call me)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
        r"(?:my name is|i am|i'm|this is|call me)\s+(\w+(?:\s+\w+)?)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text_clean, re.IGNORECASE)
        if match:
            name = match.group(1).strip()
            # Filter out non-name words
            noise = {"sure", "yes", "no", "ok", "okay", "here", "it", "the", "a"}
            if name.lower() not in noise and len(name) > 1:
                return name.title()

    # If we're actively collecting name and the message is short (likely just a name)
    if state.get("conversation_stage") == "lead_collect":
        missing = state.get("missing_fields", [])
        if "name" in missing and len(text_clean.split()) <= 3:
            # Check it doesn't look like an email or platform
            if "@" not in text_clean and not extract_platform(text_clean):
                # Filter obvious non-names
                noise = {"sure", "yes", "no", "ok", "okay", "here", "it", "the"}
                if text_clean.lower() not in noise and len(text_clean) > 1:
                    return text_clean.title()

    return None


# ---------------------------------------------------------------------------
# Node: Response Generation
# ---------------------------------------------------------------------------

def respond_node(state: AgentState) -> dict:
    """Generate the agent's response using full context."""
    messages = state["messages"]
    intent = state.get("intent", "GREETING")

    # Build conversation history for context
    history_lines = []
    for msg in messages[-10:]:  # Last 10 messages for context
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        history_lines.append(f"{role}: {msg.content}")
    history_text = "\n".join(history_lines)

    # Get knowledge base context
    if messages:
        kb_context = retrieve_knowledge(messages[-1].content)
    else:
        kb_context = ""

    # --- Response strategy based on state ---

    # Lead was just captured
    if state.get("is_tool_called") and state.get("conversation_stage") == "captured":
        name = state.get("lead_name", "")
        email = state.get("lead_email", "")
        platform = state.get("lead_platform", "")
        response_text = (
            f"Your lead has been captured successfully!\n\n"
            f"**Lead Summary:**\n"
            f"- **Name:** {name}\n"
            f"- **Email:** {email}\n"
            f"- **Platform:** {platform}\n"
            f"- **Plan Interest:** Pro Plan\n\n"
            f"Welcome aboard, {name}! Our team will reach out to you at "
            f"{email} within 24 hours to help you get started with AutoStream Pro. "
            f"In the meantime, feel free to ask any other questions!"
        )
        return {"messages": [AIMessage(content=response_text)]}

    # Collecting lead details
    if intent == "HIGH_INTENT" and not state.get("is_tool_called"):
        missing = state.get("missing_fields", ["name", "email", "platform"])
        if missing:
            field = missing[0]
            prompts = {
                "name": "I'd love to get you started with AutoStream Pro! Could you please share your **full name**?",
                "email": "Thanks! Now, could you please provide your **email address** so we can set up your account?",
                "platform": "Almost there! Which **creator platform** do you primarily use? (e.g., YouTube, Instagram, TikTok, etc.)",
            }
            response_text = prompts.get(field, f"Could you please provide your {field}?")
            return {"messages": [AIMessage(content=response_text)]}

    # Generate LLM response
    if llm is not None:
        try:
            prompt = f"""{SYSTEM_PROMPT}

KNOWLEDGE BASE CONTEXT (GROUND TRUTH):
{kb_context}

CONVERSATION HISTORY:
{history_text}

TASK:
1. If the user asked a question, answer it concisely using ONLY the provided Knowledge Base context.
2. If the context does not contain the answer, politely say you don't have that specific information.
3. If the user is a high-intent lead, guide them through the next steps (Name -> Email -> Platform).
4. Be professional, friendly, and structured. Use bullet points if listing features.

Response:"""
            response = llm.invoke(prompt)
            return {"messages": [AIMessage(content=response.content)]}
        except Exception as exc:
            logger.warning("LLM response generation failed: %s", exc)

    # Deterministic fallback responses (Refined)
    if intent == "GREETING":
        response_text = (
            "Hello! I'm the AutoStream Assistant. I can help you with product details, "
            "pricing, or get you started with a Pro subscription. How can I assist you today?"
        )
    elif intent == "PRODUCT_QUERY":
        response_text = (
            "Based on our knowledge base, here is the relevant information:\n\n"
            f"{kb_context[:500]}...\n\n"
            "Would you like more specific details on any of these points?"
        )
    else:
        response_text = (
            "I'd be happy to help you join AutoStream! We offer a **Basic Plan ($29/mo)** "
            "and a **Pro Plan ($79/mo)**. Which one interests you most?"
        )

    return {"messages": [AIMessage(content=response_text)]}


# ---------------------------------------------------------------------------
# Conditional routing
# ---------------------------------------------------------------------------

def route_after_intent(state: AgentState) -> str:
    """Decide which node to visit after intent classification."""
    intent = state.get("intent", "GREETING")

    if intent == "GREETING":
        return "respond"
    elif intent == "HIGH_INTENT":
        return "rag"
    else:  # PRODUCT_QUERY
        return "rag"


def route_after_rag(state: AgentState) -> str:
    """Decide whether to go to lead collection or response."""
    intent = state.get("intent", "GREETING")
    if intent == "HIGH_INTENT" and not state.get("is_tool_called"):
        return "lead"
    return "respond"


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_agent_graph() -> StateGraph:
    """Construct and compile the LangGraph state machine."""
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("intent", intent_node)
    graph.add_node("rag", rag_node)
    graph.add_node("lead", lead_node)
    graph.add_node("respond", respond_node)

    # Entry point
    graph.set_entry_point("intent")

    # Conditional edges from intent node
    graph.add_conditional_edges(
        "intent",
        route_after_intent,
        {
            "respond": "respond",
            "rag": "rag",
        },
    )

    # Conditional edges from RAG node
    graph.add_conditional_edges(
        "rag",
        route_after_rag,
        {
            "lead": "lead",
            "respond": "respond",
        },
    )

    # Lead node always goes to respond
    graph.add_edge("lead", "respond")

    # Response is terminal
    graph.add_edge("respond", END)

    return graph


# Compile the graph at module level for reuse
agent_app = build_agent_graph().compile()
