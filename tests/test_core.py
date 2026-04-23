import pytest
from unittest.mock import MagicMock, patch
from src.agent.graph import agent_app
from langchain_core.messages import HumanMessage, AIMessage
from src.agent.state import AgentState

from src.agent.intents import IntentClassification, LeadDetails
from types import SimpleNamespace

# Mock for LLM responses
class MockLLMResponse:
    def __init__(self, content):
        self.content = content

def test_retrieval_logic():
    """Test the RAG retrieval logic."""
    from src.tools.retrieval import retrieve_knowledge
    res = retrieve_knowledge("pricing")
    assert "Basic Plan" in res
    assert "Pro Plan" in res

def test_lead_capture_logic():
    """Test the mock tool execution."""
    from src.tools.lead_capture import mock_lead_capture
    res = mock_lead_capture("Adarsh", "adarsh@example.com", "YouTube")
    assert "Lead Captured" in res
    assert "Adarsh" in res

def test_email_validation_logic():
    """Test the regex used for email validation."""
    import re
    valid_email = "test@example.com"
    invalid_email = "test-at-example.com"
    regex = r"[^@]+@[^@]+\.[^@]+"
    assert re.match(regex, valid_email)
    assert not re.match(regex, invalid_email)

@patch("src.agent.graph.llm")
def test_full_flow_greeting(mock_llm):
    """Scenario 1: Greeting -> Response."""
    # Mocking classifier output
    mock_llm.with_structured_output.return_value.invoke.return_value = SimpleNamespace(intent="greeting")
    # Mocking response node output
    mock_llm.invoke.return_value = MockLLMResponse("Hello! How can I help you with AutoStream today?")
    
    state = {
        "messages": [HumanMessage(content="Hi")], 
        "intent": "N/A", 
        "lead_name": None, 
        "lead_email": None, 
        "lead_platform": None, 
        "is_tool_called": False
    }
    result = agent_app.invoke(state)
    
    assert result["intent"] == "greeting"
    assert "Hello" in result["messages"][-1].content
    assert "AutoStream" in result["messages"][-1].content

@patch("src.agent.graph.llm")
def test_full_flow_lead_collection(mock_llm):
    """Scenarios 3, 4, 5: Multi-turn memory, Partial lead info, Complete lead -> tool call."""
    
    # --- Turn 1: High Intent Detection ---
    mock_llm.with_structured_output.return_value.invoke.side_effect = [
        SimpleNamespace(intent="high_intent_lead"), # classifier
        SimpleNamespace(name=None, email=None, platform=None) # extractor
    ]
    mock_llm.invoke.return_value = MockLLMResponse("I'd be happy to help! What's your name?")
    
    state = {
        "messages": [HumanMessage(content="I want to buy the Pro plan")], 
        "intent": "N/A", 
        "lead_name": None, 
        "lead_email": None, 
        "lead_platform": None, 
        "is_tool_called": False
    }
    result = agent_app.invoke(state)
    
    assert result["intent"] == "high_intent_lead"
    assert "name" in result["messages"][-1].content.lower()
    
    # --- Turn 2: Providing Name (Partial Info) ---
    mock_llm.with_structured_output.return_value.invoke.side_effect = [
        SimpleNamespace(intent="high_intent_lead"), # classifier
        SimpleNamespace(name="Adarsh", email=None, platform=None) # extractor
    ]
    mock_llm.invoke.return_value = MockLLMResponse("Thanks Adarsh! What's your email?")
    
    # Simulate adding the user response to the message list
    result["messages"] = list(result["messages"]) + [HumanMessage(content="My name is Adarsh")]
    result = agent_app.invoke(result)
    
    assert result["lead_name"] == "Adarsh"
    assert result["lead_email"] is None
    assert "email" in result["messages"][-1].content.lower()
    
    # --- Turn 3: Providing Email & Platform (Completion) ---
    mock_llm.with_structured_output.return_value.invoke.side_effect = [
        SimpleNamespace(intent="high_intent_lead"), # classifier
        SimpleNamespace(name=None, email="adarsh@test.com", platform="YouTube") # extractor
    ]
    # No mock_llm.invoke needed here because it should call the mock_lead_capture tool directly in lead_capture_node
    
    result["messages"] = list(result["messages"]) + [HumanMessage(content="adarsh@test.com and I use YouTube")]
    result = agent_app.invoke(result)
    
    assert result["lead_name"] == "Adarsh"
    assert result["lead_email"] == "adarsh@test.com"
    assert result["lead_platform"] == "YouTube"
    assert result["is_tool_called"] is True
    assert "captured" in result["messages"][-1].content.lower()

@patch("src.agent.graph.llm")
def test_invalid_email_handling(mock_llm):
    """Scenario 6: Invalid email handling."""
    
    # Mocking high intent and partial extraction with invalid email
    mock_llm.with_structured_output.return_value.invoke.side_effect = [
        SimpleNamespace(intent="high_intent_lead"), # classifier
        SimpleNamespace(name="Adarsh", email="invalid-email", platform="YouTube") # extractor
    ]
    mock_llm.invoke.return_value = MockLLMResponse("Please provide a valid email address.")
    
    state = {
        "messages": [HumanMessage(content="I'm Adarsh, email is invalid-email, on YouTube")], 
        "intent": "N/A", 
        "lead_name": None, 
        "lead_email": None, 
        "lead_platform": None, 
        "is_tool_called": False
    }
    result = agent_app.invoke(state)
    
    # Lead email should be None because it was invalidated by the regex
    assert result["lead_email"] is None
    assert "valid" in result["messages"][-1].content.lower()

@patch("src.agent.graph.llm")
@patch("src.agent.graph.retrieve_knowledge")
def test_full_flow_pricing(mock_retrieve, mock_llm):
    """Scenario 2: Pricing -> Correct answer (RAG)."""
    # Mocking classifier output
    mock_llm.with_structured_output.return_value.invoke.return_value = SimpleNamespace(intent="product_inquiry")
    # Mocking retrieval output
    mock_retrieve.return_value = "Basic Plan: $29/month. Pro Plan: $79/month."
    # Mocking response node output
    mock_llm.invoke.return_value = MockLLMResponse("Our Basic Plan is $29/month and the Pro Plan is $79/month.")
    
    state = {
        "messages": [HumanMessage(content="How much does it cost?")], 
        "intent": "N/A", 
        "lead_name": None, 
        "lead_email": None, 
        "lead_platform": None, 
        "is_tool_called": False
    }
    result = agent_app.invoke(state)
    
    assert result["intent"] == "product_inquiry"
    assert "$29" in result["messages"][-1].content
    assert "$79" in result["messages"][-1].content
    mock_retrieve.assert_called_once()

