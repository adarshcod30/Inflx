import streamlit as st
import os
import textwrap
from langchain_core.messages import HumanMessage, AIMessage
import dotenv
from src.agent.graph import agent_app
from src.agent.state import AgentState

# Load environment variables
dotenv.load_dotenv()

# Page Configuration
st.set_page_config(
    page_title="AutoStream | AI Cognitive Sales Agent",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ADVANCED PREMIUM UI/UX ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    :root {
        --primary: #6366f1;
        --primary-glow: rgba(99, 102, 241, 0.5);
        --bg-dark: #0f172a;
        --card-bg: rgba(30, 41, 59, 0.7);
        --text-main: #f8fafc;
        --text-dim: #94a3b8;
    }

    html, body, [class*="st-"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: var(--text-main);
    }

    .stApp {
        background: radial-gradient(circle at 0% 0%, #1e1b4b 0%, #0f172a 100%);
    }

    /* Glassmorphism sidebar */
    [data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.95);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
    }

    /* Chat Bubbles */
    .stChatMessage {
        background-color: var(--card-bg) !important;
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 20px !important;
        padding: 1.5rem !important;
        margin-bottom: 1.2rem !important;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }

    /* Avatar Styling */
    [data-testid="stChatMessageAvatarUser"] {
        background: linear-gradient(135deg, #f472b6, #db2777) !important;
    }
    [data-testid="stChatMessageAvatarAssistant"] {
        background: linear-gradient(135deg, #818cf8, #4f46e5) !important;
    }

    /* Premium Status Card */
    .premium-card {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.8), rgba(15, 23, 42, 0.9));
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 15px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    }

    .label {
        color: var(--text-dim);
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 2px;
    }

    .value {
        color: var(--text-main);
        font-size: 1rem;
        font-weight: 700;
        margin-bottom: 12px;
    }

    .intent-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.7rem;
        font-weight: 800;
        text-transform: uppercase;
        background: rgba(99, 102, 241, 0.2);
        color: #818cf8;
        border: 1px solid rgba(99, 102, 241, 0.4);
    }

    /* Buttons */
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #4f46e5 0%, #7c3aed 100%);
        border: none;
        color: white;
        padding: 0.6rem 1rem;
        border-radius: 12px;
        font-weight: 700;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px -5px var(--primary-glow);
    }

    /* Input area */
    .stChatInputContainer {
        padding-bottom: 2rem !important;
    }

    /* Header styling */
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(to right, #fff, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        color: var(--text-dim);
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

# --- HEADER SECTION ---
with st.container():
    st.markdown('<h1 class="main-title">AutoStream</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Cognitive Social-to-Lead Agentic Workflow</p>', unsafe_allow_html=True)
    st.caption("🚀 Powered by LangGraph & Gemini 3.1 Flash Lite | ServiceHive platform")
    st.divider()

# --- SESSION STATE ---
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

# --- SIDEBAR: AGENT MONITORING ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3669/3669911.png", width=60)
    st.header("🧠 Agentic Memory")
    st.markdown("Real-time monitoring of the agent's internal cognitive state.")
    
    # Intent Visualization
    st.subheader("Current Intent")
    intent_display = st.session_state.intent.replace('_', ' ').title()
    st.markdown(f'<div class="intent-badge">{intent_display}</div>', unsafe_allow_html=True)
    
    st.divider()
    
    # Lead Profile (Slot Filling Monitor)
    st.subheader("Lead Profile")
    
    # Use clean HTML to avoid code-block rendering issues
    profile_html = f"""
    <div class="premium-card">
        <div class="label">Full Name</div>
        <div class="value">{st.session_state.lead_name or '---'}</div>
        <div class="label">Email Address</div>
        <div class="value">{st.session_state.lead_email or '---'}</div>
        <div class="label">Primary Platform</div>
        <div class="value">{st.session_state.lead_platform or '---'}</div>
    </div>
    """
    st.markdown(textwrap.dedent(profile_html), unsafe_allow_html=True)
    
    # Status Toggles
    if st.session_state.is_tool_called:
        st.success("🎯 CRM Sync Successful")
    
    st.divider()
    if st.button("Reset Session"):
        st.session_state.messages = []
        st.session_state.lead_name = None
        st.session_state.lead_email = None
        st.session_state.lead_platform = None
        st.session_state.intent = "N/A"
        st.session_state.is_tool_called = False
        st.rerun()

# --- CHAT INTERFACE ---
chat_area = st.container()

with chat_area:
    for message in st.session_state.messages:
        role = "user" if isinstance(message, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(message.content)

# --- INPUT HANDLING ---
if prompt := st.chat_input("Ask anything about video editing or AutoStream..."):
    # Append User Message
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    # Invoke Agent Logic
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Prepare current state for the graph
                current_state: AgentState = {
                    "messages": st.session_state.messages,
                    "intent": st.session_state.intent,
                    "lead_name": st.session_state.lead_name,
                    "lead_email": st.session_state.lead_email,
                    "lead_platform": st.session_state.lead_platform,
                    "is_tool_called": st.session_state.is_tool_called
                }
                
                # Execute the LangGraph workflow
                final_state = agent_app.invoke(current_state)
                
                # Synchronize application state with graph output
                st.session_state.intent = final_state.get("intent", "N/A")
                st.session_state.lead_name = final_state.get("lead_name")
                st.session_state.lead_email = final_state.get("lead_email")
                st.session_state.lead_platform = final_state.get("lead_platform")
                st.session_state.is_tool_called = final_state.get("is_tool_called", False)
                
                # Display the AI's response
                ai_response = final_state["messages"][-1].content
                st.session_state.messages.append(AIMessage(content=ai_response))
                st.markdown(ai_response)
                
                # Feedback toast for successful tool calls
                if st.session_state.is_tool_called:
                    st.toast("Success: Lead data captured and synchronized with CRM!", icon="🚀")
                    st.rerun() # Refresh to update the UI badges
                    
            except Exception as e:
                st.error(f"Agent Error: {str(e)}")
                st.info("Check your GOOGLE_API_KEY and connection.")

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: #64748b; font-size: 0.8rem;'>Built with Advanced Agentic Orchestration for AutoStream</div>", unsafe_allow_html=True)
