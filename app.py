import streamlit as st
import os
import textwrap
import time
from langchain_core.messages import HumanMessage, AIMessage
import dotenv
from src.agent.graph import agent_app
from src.agent.state import AgentState

# Load environment variables
dotenv.load_dotenv()

# Page Configuration
st.set_page_config(
    page_title="AutoStream Core | Agentic Intelligence",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- HYPER-ADVANCED AGENTIC UI/UX ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    :root {
        --primary: #8b5cf6;
        --secondary: #ec4899;
        --accent: #06b6d4;
        --bg-deep: #020617;
        --panel-bg: rgba(15, 23, 42, 0.8);
        --text-bright: #f8fafc;
        --text-muted: #94a3b8;
        --border-glow: rgba(139, 92, 246, 0.3);
    }

    /* Global Body Overrides */
    .stApp {
        background: radial-gradient(circle at 50% 50%, #0f172a 0%, #020617 100%);
        color: var(--text-bright);
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Sidebar - Cyberpunk Panel */
    [data-testid="stSidebar"] {
        background-color: var(--bg-deep) !important;
        border-right: 1px solid rgba(139, 92, 246, 0.2);
        box-shadow: 10px 0 30px rgba(0,0,0,0.5);
    }

    /* Titles & Headlines */
    .agent-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 3.5rem;
        font-weight: 700;
        letter-spacing: -2px;
        background: linear-gradient(135deg, #fff 0%, #8b5cf6 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }

    .agent-status-tag {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 0.8rem;
        color: var(--accent);
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 30px;
    }

    .pulse-dot {
        width: 8px;
        height: 8px;
        background-color: var(--accent);
        border-radius: 50%;
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(6, 182, 212, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 10px rgba(6, 182, 212, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(6, 182, 212, 0); }
    }

    /* Chat Container Customization */
    .stChatMessage {
        background: var(--panel-bg) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 24px !important;
        padding: 24px !important;
        margin-bottom: 20px !important;
        backdrop-filter: blur(12px);
        transition: all 0.3s ease;
    }

    .stChatMessage:hover {
        border-color: var(--border-glow) !important;
        box-shadow: 0 0 20px rgba(139, 92, 246, 0.1);
    }

    /* Sidebar Widgets */
    .sidebar-card {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 24px;
        margin-bottom: 20px;
    }

    .stat-label {
        color: var(--text-muted);
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }

    .stat-value {
        color: #fff;
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 16px;
        font-family: 'Space Grotesk', sans-serif;
    }

    /* Custom Chat Input */
    .stChatInputContainer {
        border-top: 1px solid rgba(255,255,255,0.1) !important;
        background: var(--bg-deep) !important;
    }

    /* Glowing Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #8b5cf6 0%, #ec4899 100%) !important;
        border: none !important;
        border-radius: 14px !important;
        color: white !important;
        font-weight: 700 !important;
        padding: 12px 24px !important;
        box-shadow: 0 4px 15px rgba(139, 92, 246, 0.3) !important;
    }

    .stButton > button:hover {
        transform: scale(1.02) !important;
        box-shadow: 0 4px 25px rgba(139, 92, 246, 0.5) !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- APP HEADER ---
st.markdown('<h1 class="agent-title">AUTOREACH</h1>', unsafe_allow_html=True)
st.markdown('<div class="agent-status-tag"><div class="pulse-dot"></div> Systems Online | Cognitive Agent Active</div>', unsafe_allow_html=True)

# --- SESSION INITIALIZATION ---
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

# --- SIDEBAR: COGNITIVE OVERLAY ---
with st.sidebar:
    st.markdown("### 🧬 COGNITIVE OVERLAY")
    st.caption("Monitoring real-time neural orchestrations.")
    
    # Internal State Card
    st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="stat-label">Active Intent</div><div class="stat-value">{st.session_state.intent.upper().replace("_", " ")}</div>', unsafe_allow_html=True)
    
    # Reasoning Visualization (Simulated for UX)
    st.markdown('<div class="stat-label">Neural Capacity</div>', unsafe_allow_html=True)
    st.progress(0.85)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Lead Profile (Entity Extraction Monitor)
    st.markdown("### 👤 SUBJECT PROFILE")
    profile_html = f"""
    <div class="sidebar-card">
        <div class="stat-label">Identified Name</div>
        <div class="stat-value">{st.session_state.lead_name or "---"}</div>
        <div class="stat-label">Verified Email</div>
        <div class="stat-value">{st.session_state.lead_email or "---"}</div>
        <div class="stat-label">Target Platform</div>
        <div class="stat-value">{st.session_state.lead_platform or "---"}</div>
    </div>
    """
    st.markdown(profile_html, unsafe_allow_html=True)
    
    if st.session_state.is_tool_called:
        st.info("🎯 TARGET DATA SYNCHRONIZED")
        
    st.divider()
    if st.button("TERMINATE SESSION"):
        st.session_state.messages = []
        st.session_state.lead_name = None
        st.session_state.lead_email = None
        st.session_state.lead_platform = None
        st.session_state.intent = "N/A"
        st.session_state.is_tool_called = False
        st.rerun()

# --- INTERACTION LAYER ---
chat_container = st.container()

with chat_container:
    for message in st.session_state.messages:
        role = "user" if isinstance(message, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(message.content)

# --- NEURAL INPUT ---
if prompt := st.chat_input("Initiate interaction..."):
    # User Input
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    # Agent Processing
    with st.chat_message("assistant"):
        thinking_placeholder = st.empty()
        thinking_placeholder.markdown("*Synthesizing response...*")
        
        try:
            # Graph Invocation
            current_state: AgentState = {
                "messages": st.session_state.messages,
                "intent": st.session_state.intent,
                "lead_name": st.session_state.lead_name,
                "lead_email": st.session_state.lead_email,
                "lead_platform": st.session_state.lead_platform,
                "is_tool_called": st.session_state.is_tool_called
            }
            
            final_state = agent_app.invoke(current_state)
            
            # Sync State
            st.session_state.intent = final_state.get("intent", "N/A")
            st.session_state.lead_name = final_state.get("lead_name")
            st.session_state.lead_email = final_state.get("lead_email")
            st.session_state.lead_platform = final_state.get("lead_platform")
            st.session_state.is_tool_called = final_state.get("is_tool_called", False)
            
            # Display Output
            ai_response = final_state["messages"][-1].content
            st.session_state.messages.append(AIMessage(content=ai_response))
            
            thinking_placeholder.empty()
            st.markdown(ai_response)
            
            if st.session_state.is_tool_called:
                st.toast("Protocol Complete: Lead Data Captured.", icon="🎯")
                time.sleep(0.5)
                st.rerun()
                
        except Exception as e:
            thinking_placeholder.empty()
            st.error(f"NEURAL DISRUPTION: {str(e)}")

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("<div style='text-align: center; color: var(--text-muted); font-size: 0.7rem; letter-spacing: 3px;'>CORE ORCHESTRATION ENGINE v2.5.0</div>", unsafe_allow_html=True)
