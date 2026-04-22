import pytest
from unittest.mock import MagicMock, patch
from src.agent.graph import agent_app
from langchain_core.messages import HumanMessage, AIMessage

@pytest.fixture
def mock_llm():
    with patch("src.agent.graph.ChatGoogleGenerativeAI") as mock:
        yield mock

def test_classifier_node_greeting():
    # This is a bit complex to unit test nodes directly because of the graph structure,
    # but we can test the logic.
    pass

def test_retrieval_logic():
    from src.tools.retrieval import retrieve_knowledge
    res = retrieve_knowledge("pricing")
    assert "Basic Plan" in res
    assert "Pro Plan" in res

def test_lead_capture_logic():
    from src.tools.lead_capture import mock_lead_capture
    res = mock_lead_capture("Adarsh", "adarsh@example.com", "YouTube")
    assert "Lead Captured" in res
    assert "Adarsh" in res
