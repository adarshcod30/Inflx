import os
from typing import TypedDict, Annotated, Sequence, Optional
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import dotenv

from src.agent.state import AgentState
from src.agent.intents import IntentClassification, LeadDetails
from src.tools.retrieval import retrieve_knowledge
from src.tools.lead_capture import mock_lead_capture

# Load env variables safely
dotenv.load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY", "")

# Initialize LLM
# We use a dummy key if none is provided to allow the graph to be compiled/tested
# In production, the user must provide a valid GOOGLE_API_KEY
llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview", 
    temperature=0.1, 
    google_api_key=os.getenv("GOOGLE_API_KEY", "dummy_key")
)

def _get_chat_history_text(messages, k=6):
    """Format the last k messages for prompt context."""
    recent_messages = messages[-k:]
    return "\n".join([f"{'User' if isinstance(m, HumanMessage) else 'Agent'}: {m.content}" for m in recent_messages])

def classifier_node(state: AgentState):
    """
    Classifies the user intent based on the conversation history.
    """
    messages = state["messages"]
    if not messages:
        return {"intent": "greeting"}

    system_prompt = """You are an intent classifier for AutoStream, a SaaS video editing tool.
Classify the user's latest message into one of three categories:
1. 'greeting': Casual hello, how are you, or introductory pleasantries.
2. 'product_inquiry': Asking about pricing, features, capabilities, or policies.
3. 'high_intent_lead': The user explicitly wants to sign up, try the pro plan, or buy the product.

Conversation history:"""
    
    history_text = _get_chat_history_text(messages, 4)
    prompt = f"{system_prompt}\n{history_text}\n\nClassify the intent of the USER'S LAST MESSAGE."
    
    structured_llm = llm.with_structured_output(IntentClassification)
    result = structured_llm.invoke(prompt)
    
    return {"intent": result.intent}

def greeting_node(state: AgentState):
    """
    Handles casual greetings.
    """
    prompt = """You are a helpful and energetic assistant for AutoStream, an automated video editing SaaS for content creators.
Respond politely to the user's greeting in 1-2 short sentences and ask how you can help them with their video editing workflow today."""
    
    full_prompt = f"{prompt}\n\nUser: {state['messages'][-1].content}\n\nAssistant:"
    response = llm.invoke(full_prompt)
    
    content = response.content
    if isinstance(content, list) and len(content) > 0:
        content = content[0].get("text", str(content))
    elif not content:
        content = "I'm sorry, I couldn't generate a response. How else can I help?"
        
    return {"messages": [AIMessage(content=str(content))]}

def qa_node(state: AgentState):
    """
    Handles product/pricing inquiries using RAG.
    """
    user_msg = state["messages"][-1].content
    
    # Retrieve relevant context from local KB
    context = retrieve_knowledge(user_msg)
    
    system_prompt = f"""You are a knowledgeable sales assistant for AutoStream.
Answer the user's question based strictly on the following context.
If the context does not contain the answer, say "I don't have that information right now, but our support team can help."
Keep your answer clear, persuasive, and friendly.

Context from Knowledge Base:
{context}
"""
    full_prompt = f"{system_prompt}\n\nUser: {state['messages'][-1].content}\n\nAssistant:"
    response = llm.invoke(full_prompt)
    
    content = response.content
    if isinstance(content, list) and len(content) > 0:
        content = content[0].get("text", str(content))
    elif not content:
        content = "I'm sorry, I'm having trouble retrieving that information right now."
        
    return {"messages": [AIMessage(content=str(content))]}

def lead_capture_node(state: AgentState):
    """
    Handles high-intent users, collecting name, email, and platform before executing lead capture.
    """
    system_prompt = """Extract lead details from the conversation if present. 
If the user hasn't provided a specific field yet, return null for it.
Fields to extract: name, email, platform.

Conversation history:"""
    history_text = _get_chat_history_text(state["messages"], 8)
    
    extractor_llm = llm.with_structured_output(LeadDetails)
    details = extractor_llm.invoke(f"{system_prompt}\n\n{history_text}")
    
    # Consolidate state
    new_name = details.name or state.get("lead_name")
    new_email = details.email or state.get("lead_email")
    new_platform = details.platform or state.get("lead_platform")
    
    import re
    missing = []
    if not new_name: missing.append("Name")
    if not new_email: 
        missing.append("Email")
    elif not re.match(r"[^@]+@[^@]+\.[^@]+", new_email):
        missing.append("a valid Email address")
        new_email = None
    if not new_platform: missing.append("Creator Platform (e.g., YouTube, Instagram)")
    
    if not missing:
        # All details collected -> Execute Tool
        capture_result = mock_lead_capture(new_name, new_email, new_platform)
        return {
            "lead_name": new_name,
            "lead_email": new_email,
            "lead_platform": new_platform,
            "is_tool_called": True,
            "messages": [AIMessage(content=f"Excellent! I've captured your details. {capture_result}. Our team will contact you shortly to help you scale your {new_platform} content with AutoStream!")]
        }
    else:
        # Ask for the missing details naturally
        ask_prompt = f"""You are onboarding a high-intent user for AutoStream.
You still need to collect their: {', '.join(missing)}.
Politely and concisely ask them to provide this missing information. 
Mention that this is needed to set up their trial or account."""
        
        response = llm.invoke(ask_prompt)
        content = response.content
        if isinstance(content, list) and len(content) > 0:
            content = content[0].get("text", str(content))
        elif not content:
            content = "Could you please provide your name and email to get started?"
            
        return {
            "lead_name": new_name,
            "lead_email": new_email,
            "lead_platform": new_platform,
            "messages": [AIMessage(content=str(content))]
        }

def route_intent(state: AgentState):
    """
    Conditional routing function based on intent.
    """
    intent = state.get("intent", "greeting")
    if intent == "product_inquiry":
        return "qa"
    elif intent == "high_intent_lead":
        return "lead_capture"
    return "greeting"

# Build the Graph
workflow = StateGraph(AgentState)

workflow.add_node("classifier", classifier_node)
workflow.add_node("greeting", greeting_node)
workflow.add_node("qa", qa_node)
workflow.add_node("lead_capture", lead_capture_node)

workflow.set_entry_point("classifier")

workflow.add_conditional_edges(
    "classifier",
    route_intent,
    {
        "greeting": "greeting",
        "qa": "qa",
        "lead_capture": "lead_capture"
    }
)

workflow.add_edge("greeting", END)
workflow.add_edge("qa", END)
workflow.add_edge("lead_capture", END)

agent_app = workflow.compile()
