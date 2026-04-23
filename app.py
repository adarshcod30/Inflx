import streamlit as st
import os
from langchain_core.messages import HumanMessage, AIMessage
import dotenv
from src.agent.graph import agent_app
from src.agent.state import AgentState

# Load environment variables
dotenv.load_dotenv()

# Page Configuration
st.set_page_config(
    page_title="AutoStream | AI Lead Agent",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Look
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
    
    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: radial-gradient(circle at top right, #1e1e2f, #121212);
        color: #ffffff;
    }
    
    /* Glassmorphism for containers */
    div[data-testid="stVerticalBlock"] > div:has(div.stChatFloatingInputContainer) {
        background: transparent;
    }
    
    .stChatMessage {
        background-color: rgba(255, 255, 255, 0.03);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        margin-bottom: 10px;
        backdrop-filter: blur(10px);
    }
    
    .stButton>button {
        background: linear-gradient(90deg, #6a11cb 0%, #2575fc 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        transition: 0.3s all ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(37, 117, 252, 0.5);
    }
    
    .sidebar .sidebar-content {
        background-color: #161625;
    }
    
    h1 {
        background: linear-gradient(90deg, #fff 0%, #aaa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    
    .status-card {
        background: rgba(37, 117, 252, 0.1);
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 15px;
        border: 1px solid rgba(37, 117, 252, 0.2);
    }
    
    .field-label {
        color: #888;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 4px;
    }
    
    .field-value {
        font-weight: 600;
        color: #fff;
        margin-bottom: 10px;
    }
    
    .badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.7rem;
        font-weight: 800;
        margin-right: 5px;
    }
    
    .badge-blue { background: #2575fc; color: white; }
    .badge-green { background: #00c853; color: white; }
    .badge-red { background: #ff3d00; color: white; }
    </style>
    """, unsafe_allow_html=True)

# App Header
col1, col2 = st.columns([1, 4])
with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/3669/3669911.png", width=80)
with col2:
    st.title("AutoStream Agentic Workflow")
    st.caption("🚀 Powered by LangGraph & Gemini 2 Flash Lite | Social-to-Lead Generation")

st.divider()

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "lead_name" not in st.session_state:
    st.session_state.lead_name = None
if "lead_email" not in st.session_state:
    st.session_state.lead_email = None
if "lead_platform" not in st.session_state:
    st.session_state.lead_platform = None
if "intent" not in st.session_state:
    st.session_state.intent = "N/A"
if "is_tool_called" not in st.session_state:
    st.session_state.is_tool_called = False

# Sidebar: Agent State Monitoring
with st.sidebar:
    st.header("🧠 Agentic Memory")
    st.markdown("Monitor the internal state of the LangGraph agent in real-time.")
    
    # Intent Visualization
    st.subheader("Current Intent")
    intent_color = {
        "greeting": "🔵",
        "product_inquiry": "🟢",
        "high_intent_lead": "🔥",
        "N/A": "⚪"
    }.get(st.session_state.intent, "⚪")
    st.info(f"{intent_color} **{st.session_state.intent.replace('_', ' ').title()}**")
    
    # Lead Profile Progress
    st.subheader("Lead Profile")
    
    def get_status_icon(val):
        return "✅" if val else "❌"
    
    st.markdown(f"""
    <div class="status-card">
        <div class="field-label">Full Name</div>
        <div class="field-value">{st.session_state.lead_name or 'Not Captured'}</div>
        
        <div class="field-label">Email Address</div>
        <div class="field-value">{st.session_state.lead_email or 'Not Captured'}</div>
        
        <div class="field-label">Creator Platform</div>
        <div class="field-value">{st.session_state.lead_platform or 'Not Captured'}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Tool Call Status
    if st.session_state.is_tool_called:
        st.success("🎯 **Lead Captured via Mock API**")
    
    st.divider()
    if st.button("Reset Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.lead_name = None
        st.session_state.lead_email = None
        st.session_state.lead_platform = None
        st.session_state.intent = "N/A"
        st.session_state.is_tool_called = False
        st.rerun()

# Main Chat Interface
chat_container = st.container()

with chat_container:
    for message in st.session_state.messages:
        role = "user" if isinstance(message, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(message.content)

# Input Area
if prompt := st.chat_input("Ask about AutoStream pricing..."):
    # User Message
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    # Invoke Agent
    with st.chat_message("assistant"):
        with st.spinner("Agent thinking..."):
            try:
                # Prepare State
                current_state: AgentState = {
                    "messages": st.session_state.messages,
                    "intent": st.session_state.intent,
                    "lead_name": st.session_state.lead_name,
                    "lead_email": st.session_state.lead_email,
                    "lead_platform": st.session_state.lead_platform,
                    "is_tool_called": st.session_state.is_tool_called
                }
                
                # Run Graph
                final_state = agent_app.invoke(current_state)
                
                # Update State
                st.session_state.intent = final_state.get("intent")
                st.session_state.lead_name = final_state.get("lead_name")
                st.session_state.lead_email = final_state.get("lead_email")
                st.session_state.lead_platform = final_state.get("lead_platform")
                st.session_state.is_tool_called = final_state.get("is_tool_called", False)
                
                # Get Response
                ai_response = final_state["messages"][-1].content
                st.session_state.messages.append(AIMessage(content=ai_response))
                st.markdown(ai_response)
                
                # Small debug toast
                if st.session_state.is_tool_called:
                    st.toast("Success: Lead data synchronized!", icon="🚀")
                    
            except Exception as e:
                import traceback
                print(f"DEBUG ERROR: {str(e)}")
                traceback.print_exc()
                st.error(f"An error occurred: {str(e)}")
                st.info("Ensure GOOGLE_API_KEY is set in your .env file.")

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: #666;'>Built for ServiceHive Internship Assignment | AutoStream Agent</div>", unsafe_allow_html=True)
