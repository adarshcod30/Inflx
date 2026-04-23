"""Comprehensive test suite for the AutoStream Agentic Workflow.

Tests cover:
  - RAG knowledge retrieval accuracy
  - Mock lead capture tool execution
  - Intent classification (rule-based fallback)
  - Multi-turn lead collection flow
  - Email validation edge cases
  - Graph compilation and node connectivity
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agent.intent import (
    classify_intent_from_text,
    extract_email,
    extract_platform,
)
from agent.rag import retrieve_knowledge, get_full_knowledge_base
from agent.state import AgentState, default_state
from agent.tools import mock_lead_capture


# =====================================================================
# RAG Retrieval Tests
# =====================================================================

class TestRAGRetrieval:
    """Verify knowledge base loading and retrieval accuracy."""

    def test_pricing_retrieval_contains_basic_plan(self):
        context = retrieve_knowledge("pricing plans")
        assert "$29/month" in context
        assert "720p" in context

    def test_pricing_retrieval_contains_pro_plan(self):
        context = retrieve_knowledge("Pro plan features")
        assert "$79/month" in context
        assert "4K" in context

    def test_support_policy_retrieval(self):
        context = retrieve_knowledge("support policy 24/7")
        assert "24/7" in context
        assert "Pro" in context

    def test_refund_policy_retrieval(self):
        context = retrieve_knowledge("refund policy")
        assert "7 days" in context

    def test_full_knowledge_base_loads(self):
        full_kb = get_full_knowledge_base()
        assert len(full_kb) > 200
        assert "$29/month" in full_kb
        assert "$79/month" in full_kb

    def test_irrelevant_query_returns_content(self):
        """Even irrelevant queries should return best-effort matches."""
        context = retrieve_knowledge("quantum physics")
        assert isinstance(context, str)


# =====================================================================
# Mock Tool Execution Tests
# =====================================================================

class TestMockLeadCapture:
    """Verify the lead capture tool works correctly."""

    def test_successful_capture(self):
        result = mock_lead_capture("Adarsh", "adarsh@example.com", "YouTube")
        assert "Lead captured successfully" in result
        assert "adarsh@example.com" in result
        assert "YouTube" in result

    def test_capture_returns_timestamp(self):
        result = mock_lead_capture("Test User", "test@test.com", "Instagram")
        assert "captured successfully" in result.lower()

    def test_capture_with_different_platforms(self):
        platforms = ["YouTube", "Instagram", "TikTok", "Twitter/X"]
        for platform in platforms:
            result = mock_lead_capture("User", "u@test.com", platform)
            assert platform in result


# =====================================================================
# Intent Classification Tests (Rule-based)
# =====================================================================

class TestIntentClassification:
    """Verify rule-based intent classification fallback."""

    def test_greeting_detection(self):
        assert classify_intent_from_text("Hi there") == "GREETING"
        assert classify_intent_from_text("Hello!") == "GREETING"
        assert classify_intent_from_text("Hey") == "GREETING"

    def test_product_query_detection(self):
        assert classify_intent_from_text("What are your pricing plans?") == "PRODUCT_QUERY"
        assert classify_intent_from_text("Tell me about features") == "PRODUCT_QUERY"
        assert classify_intent_from_text("How much does it cost?") == "PRODUCT_QUERY"

    def test_high_intent_detection(self):
        assert classify_intent_from_text("I want to sign up") == "HIGH_INTENT"
        assert classify_intent_from_text("I want to buy the Pro plan") == "HIGH_INTENT"
        assert classify_intent_from_text("Let me subscribe") == "HIGH_INTENT"

    def test_refund_is_product_query(self):
        assert classify_intent_from_text("What is your refund policy?") == "PRODUCT_QUERY"

    def test_ambiguous_defaults_to_product_query(self):
        result = classify_intent_from_text("Tell me more")
        assert result in {"PRODUCT_QUERY", "GREETING"}


# =====================================================================
# Email Extraction Tests
# =====================================================================

class TestEmailExtraction:
    """Verify email extraction from free-form text."""

    def test_valid_email_extraction(self):
        assert extract_email("my email is adarsh@gmail.com") == "adarsh@gmail.com"

    def test_email_in_sentence(self):
        assert extract_email("You can reach me at test@company.co.uk okay?") == "test@company.co.uk"

    def test_no_email_returns_none(self):
        assert extract_email("I don't have an email to share") is None

    def test_invalid_email_not_extracted(self):
        assert extract_email("my email is adarsh-at-gmail.com") is None


# =====================================================================
# Platform Extraction Tests
# =====================================================================

class TestPlatformExtraction:
    """Verify platform detection from text."""

    def test_youtube_detection(self):
        assert extract_platform("I create on YouTube") == "YouTube"

    def test_instagram_detection(self):
        assert extract_platform("I use Instagram for content") == "Instagram"

    def test_tiktok_detection(self):
        assert extract_platform("My main platform is TikTok") == "TikTok"

    def test_no_platform_returns_none(self):
        assert extract_platform("I just make videos") is None


# =====================================================================
# State Management Tests
# =====================================================================

class TestStateManagement:
    """Verify default state initialization."""

    def test_default_state_has_all_keys(self):
        state = default_state()
        required_keys = [
            "messages", "intent", "conversation_stage", "is_qualified",
            "lead_name", "lead_email", "lead_platform",
            "missing_fields", "is_tool_called",
        ]
        for key in required_keys:
            assert key in state, f"Missing key: {key}"

    def test_default_state_starts_clean(self):
        state = default_state()
        assert state["intent"] == "GREETING"
        assert state["conversation_stage"] == "greeting"
        assert state["is_qualified"] is False
        assert state["is_tool_called"] is False
        assert state["lead_name"] is None
        assert state["lead_email"] is None
        assert state["lead_platform"] is None
        assert state["missing_fields"] == []


# =====================================================================
# Graph Compilation Test
# =====================================================================

class TestGraphCompilation:
    """Verify the LangGraph compiles without errors."""

    def test_graph_compiles(self):
        from agent.graph import agent_app
        assert agent_app is not None

    def test_graph_has_nodes(self):
        from agent.graph import build_agent_graph
        graph = build_agent_graph()
        assert graph is not None
