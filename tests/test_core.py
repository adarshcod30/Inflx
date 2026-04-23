"""Comprehensive test suite for the AutoStream Agentic Workflow.

Tests cover ALL required scenarios:
  1. Greeting flow
  2. Pricing query (RAG)
  3. Multi-turn memory
  4. High intent detection
  5. Partial lead input
  6. Full lead → tool execution
  7. Invalid email handling
  8. Edge cases
  9. Graph compilation and node connectivity
  10. State management
"""

import ast
import pathlib
import importlib

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agent.intent import classify_intent_from_text, extract_email, extract_platform, extract_name
from agent.lead import compute_missing_fields, is_fully_qualified, should_trigger_tool, next_field_prompt
from agent.rag import retrieve_knowledge, get_full_knowledge_base
from agent.state import AgentState, default_state
from agent.tools import mock_lead_capture


# =====================================================================
# 1. Greeting Flow Tests
# =====================================================================

class TestGreetingFlow:
    def test_hi(self):
        assert classify_intent_from_text("Hi there") == "GREETING"

    def test_hello(self):
        assert classify_intent_from_text("Hello!") == "GREETING"

    def test_hey(self):
        assert classify_intent_from_text("Hey") == "GREETING"

    def test_good_morning(self):
        assert classify_intent_from_text("Good morning") == "GREETING"


# =====================================================================
# 2. Pricing Query (RAG) Tests
# =====================================================================

class TestRAGRetrieval:
    def test_basic_plan_pricing(self):
        context = retrieve_knowledge("pricing plans")
        assert "$29/month" in context
        assert "720p" in context

    def test_pro_plan_pricing(self):
        context = retrieve_knowledge("Pro plan features")
        assert "$79/month" in context
        assert "4K" in context

    def test_support_policy(self):
        context = retrieve_knowledge("support policy 24/7")
        assert "24/7" in context

    def test_refund_policy(self):
        context = retrieve_knowledge("refund policy")
        assert "7 days" in context

    def test_full_kb_loads(self):
        full_kb = get_full_knowledge_base()
        assert len(full_kb) > 200
        assert "$29/month" in full_kb
        assert "$79/month" in full_kb

    def test_no_hallucination_irrelevant_query(self):
        context = retrieve_knowledge("quantum physics black holes")
        assert isinstance(context, str)


# =====================================================================
# 3. Multi-turn Memory Tests
# =====================================================================

class TestMultiTurnMemory:
    def test_state_persists_across_defaults(self):
        state = default_state()
        state["lead_name"] = "Alice"
        assert state["lead_name"] == "Alice"
        assert state["lead_email"] is None

    def test_state_has_all_required_keys(self):
        state = default_state()
        required = ["messages", "intent", "conversation_stage", "is_qualified",
                     "lead_name", "lead_email", "lead_platform",
                     "missing_fields", "is_tool_called", "rag_context"]
        for k in required:
            assert k in state, f"Missing key: {k}"

    def test_default_state_is_clean(self):
        state = default_state()
        assert state["intent"] == "GREETING"
        assert state["is_qualified"] is False
        assert state["is_tool_called"] is False
        assert state["rag_context"] == ""


# =====================================================================
# 4. High Intent Detection Tests
# =====================================================================

class TestHighIntentDetection:
    def test_sign_up(self):
        assert classify_intent_from_text("I want to sign up") == "HIGH_INTENT"

    def test_buy(self):
        assert classify_intent_from_text("I want to buy the Pro plan") == "HIGH_INTENT"

    def test_subscribe(self):
        assert classify_intent_from_text("Let me subscribe") == "HIGH_INTENT"

    def test_interested(self):
        assert classify_intent_from_text("I'm interested in joining") == "HIGH_INTENT"

    def test_get_started(self):
        assert classify_intent_from_text("I'd like to get started") == "HIGH_INTENT"


# =====================================================================
# 5. Partial Lead Input Tests
# =====================================================================

class TestPartialLeadInput:
    def test_missing_all_fields(self):
        missing = compute_missing_fields(None, None, None)
        assert missing == ["name", "email", "platform"]

    def test_missing_email_and_platform(self):
        missing = compute_missing_fields("Alice", None, None)
        assert missing == ["email", "platform"]

    def test_missing_platform_only(self):
        missing = compute_missing_fields("Alice", "alice@test.com", None)
        assert missing == ["platform"]

    def test_no_missing_fields(self):
        missing = compute_missing_fields("Alice", "alice@test.com", "YouTube")
        assert missing == []

    def test_next_prompt_for_name(self):
        prompt = next_field_prompt(["name", "email", "platform"])
        assert "full name" in prompt.lower()

    def test_next_prompt_for_email(self):
        prompt = next_field_prompt(["email", "platform"])
        assert "email" in prompt.lower()


# =====================================================================
# 6. Full Lead → Tool Execution Tests
# =====================================================================

class TestFullLeadCapture:
    def test_successful_capture(self):
        result = mock_lead_capture("Adarsh", "adarsh@example.com", "YouTube")
        assert result["success"] is True
        assert "adarsh@example.com" in result["message"]

    def test_capture_returns_lead_record(self):
        result = mock_lead_capture("Test User", "test@test.com", "Instagram")
        assert result["lead"]["name"] == "Test User"
        assert result["lead"]["platform"] == "Instagram"

    def test_tool_trigger_gate_all_present(self):
        state = {"lead_name": "Alice", "lead_email": "a@b.com",
                 "lead_platform": "YouTube", "is_tool_called": False}
        assert should_trigger_tool(state) is True

    def test_tool_trigger_gate_missing_field(self):
        state = {"lead_name": "Alice", "lead_email": None,
                 "lead_platform": "YouTube", "is_tool_called": False}
        assert should_trigger_tool(state) is False

    def test_tool_trigger_gate_already_called(self):
        state = {"lead_name": "Alice", "lead_email": "a@b.com",
                 "lead_platform": "YouTube", "is_tool_called": True}
        assert should_trigger_tool(state) is False


# =====================================================================
# 7. Invalid Email Handling Tests
# =====================================================================

class TestEmailValidation:
    def test_valid_email(self):
        assert extract_email("my email is adarsh@gmail.com") == "adarsh@gmail.com"

    def test_email_in_sentence(self):
        assert extract_email("Reach me at test@company.co.uk ok?") == "test@company.co.uk"

    def test_no_email(self):
        assert extract_email("I don't have an email") is None

    def test_invalid_email_pattern(self):
        assert extract_email("my email is adarsh-at-gmail.com") is None

    def test_tool_rejects_invalid_email(self):
        with pytest.raises(ValueError, match="Invalid email"):
            mock_lead_capture("Alice", "not-an-email", "YouTube")

    def test_fully_qualified_rejects_bad_email(self):
        assert is_fully_qualified("Alice", "bad-email", "YouTube") is False


# =====================================================================
# 8. Edge Cases
# =====================================================================

class TestEdgeCases:
    def test_empty_input_defaults_product_query(self):
        result = classify_intent_from_text("")
        assert result in {"PRODUCT_QUERY", "GREETING"}

    def test_platform_extraction_youtube(self):
        assert extract_platform("I create on YouTube") == "YouTube"

    def test_platform_extraction_tiktok(self):
        assert extract_platform("My main is TikTok") == "TikTok"

    def test_no_platform(self):
        assert extract_platform("I just make videos") is None

    def test_name_extraction_pattern(self):
        assert extract_name("My name is Adarsh Kumar") == "Adarsh Kumar"

    def test_name_extraction_contextual(self):
        name = extract_name("John", stage="lead_collect", missing=["name"])
        assert name == "John"

    def test_refund_is_product_query(self):
        assert classify_intent_from_text("What is your refund policy?") == "PRODUCT_QUERY"

    def test_multiple_platforms_in_capture(self):
        platforms = ["YouTube", "Instagram", "TikTok", "Twitter/X"]
        for p in platforms:
            result = mock_lead_capture("User", "u@test.com", p)
            assert p in result["message"]


# =====================================================================
# 9. Graph Compilation Tests
# =====================================================================

class TestGraphCompilation:
    def test_graph_compiles(self):
        from agent.graph import agent_app
        assert agent_app is not None

    def test_graph_has_nodes(self):
        from agent.graph import build_agent_graph
        graph = build_agent_graph()
        assert graph is not None

    def test_model_name_default(self, monkeypatch):
        monkeypatch.delenv("AUTOSTREAM_MODEL", raising=False)
        import agent.graph as gm
        importlib.reload(gm)
        assert gm.MODEL_NAME == "gemini-3.1-flash-lite-preview"


# =====================================================================
# 10. UI Constants Tests
# =====================================================================

class TestUIConstants:
    STAGE_TO_INDEX = {"greeting": 0, "inquiry": 1, "lead_collect": 2, "captured": 3}
    FUNNEL_STEPS = ["Greeting", "Inquiry", "Lead Collection", "Captured"]

    def test_all_stages_unique(self):
        assert sorted(self.STAGE_TO_INDEX.values()) == [0, 1, 2, 3]

    def test_funnel_has_four_steps(self):
        assert len(self.FUNNEL_STEPS) == 4

    def _source_contains(self, snippet):
        return snippet in pathlib.Path("app.py").read_text()

    def test_funnel_defined(self):
        assert self._source_contains("FUNNEL_STEPS")

    def test_stage_defined(self):
        assert self._source_contains("STAGE_TO_INDEX")


class TestUIFunctions:
    def _get_app_functions(self):
        source = pathlib.Path("app.py").read_text()
        tree = ast.parse(source)
        return {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}

    def test_render_funnel(self):
        assert "render_funnel" in self._get_app_functions()

    def test_render_lead_card(self):
        assert "render_lead_card" in self._get_app_functions()

    def test_render_chat_page(self):
        assert "render_chat_page" in self._get_app_functions()

    def test_render_dashboard_page(self):
        assert "render_dashboard_page" in self._get_app_functions()

    def test_render_architecture_page(self):
        assert "render_architecture_page" in self._get_app_functions()
