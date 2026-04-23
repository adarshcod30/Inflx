"""AutoStream Agentic Console — Premium Conversational Dashboard.

A high-fidelity, professional interface for the Social-to-Lead workflow.
Redesigned for maximum clarity, focus, and technical observability.
"""

import os
import time

import dotenv
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from agent.graph import agent_app
from agent.state import AgentState, default_state

dotenv.load_dotenv()
MODEL_REFERENCE = os.getenv("AUTOSTREAM_MODEL", "gemini-2.0-flash-lite")

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AutoStream | AI Lead Agent",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Professional CSS (Refined & Clean)
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root {
        --bg-primary: #0a0c10;
        --bg-secondary: #111418;
        --bg-card: rgba(17, 20, 24, 0.95);
        --accent: #3b82f6;
        --accent-glow: rgba(59, 130, 246, 0.3);
        --text-primary: #f8fafc;
        --text-secondary: #94a3b8;
        --border: rgba(255, 255, 255, 0.08);
        --radius: 12px;
    }

    html, body, [class*="st-"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .stApp {
        background-color: var(--bg-primary);
        color: var(--text-primary);
    }

    /* Main Container */
    .main-chat-container {
        max-width: 800px;
        margin: 0 auto;
        padding: 2rem 1rem;
    }

    /* Header */
    .project-header {
        text-align: center;
        margin-bottom: 3rem;
    }

    .project-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #fff 0%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }

    .project-subtitle {
        color: var(--text-secondary);
        font-size: 1rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        font-weight: 500;
    }

    /* Status Bar */
    .status-bar {
        display: flex;
        justify-content: center;
        gap: 1.5rem;
        margin-bottom: 2rem;
        padding: 0.75rem;
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 999px;
        backdrop-filter: blur(10px);
    }

    .status-item {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.85rem;
        color: var(--text-secondary);
    }

    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #475569;
    }

    .status-dot.active {
        background: #22c55e;
        box-shadow: 0 0 8px #22c55e;
    }

    /* Chat Messages */
    .stChatMessage {
        background: transparent !important;
        border: none !important;
        padding: 1.5rem 0 !important;
    }

    .message-bubble {
        padding: 1rem 1.25rem;
        border-radius: var(--radius);
        max-width: 85%;
        line-height: 1.6;
        font-size: 0.95rem;
    }

    .user-bubble {
        background: var(--bg-secondary);
        border: 1px solid var(--border);
        margin-left: auto;
    }

    .assistant-bubble {
        background: rgba(59, 130, 246, 0.05);
        border: 1px solid rgba(59, 130, 246, 0.1);
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: var(--bg-secondary) !important;
        border-right: 1px solid var(--border) !important;
    }

    .sidebar-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 1rem;
        margin-bottom: 1rem;
    }

    .card-label {
        font-size: 0.75rem;
        font-weight: 600;
        color: var(--text-secondary);
        text-transform: uppercase;
        margin-bottom: 0.75rem;
    }

    /* Pipeline Visualization */
    .pipeline-track {
        display: flex;
        align-items: center;
        gap: 0.25rem;
        margin-top: 0.5rem;
    }

    .pipeline-step {
        height: 4px;
        flex: 1;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 2px;
    }

    .pipeline-step.active {
        background: var(--accent);
        box-shadow: 0 0 10px var(--accent-glow);
    }

    /* Input Styling */
    .stChatInputContainer {
        border-top: 1px solid var(--border) !important;
        padding-top: 1rem !important;
        background-color: var(--bg-primary) !important;
    }

    /* Quick Actions */
    .quick-action-btn {
        margin-bottom: 0.5rem;
    }

    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        color: var(--accent);
        font-weight: 600;
    }

    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session Helpers
# ---------------------------------------------------------------------------

def init_session() -> None:
    defaults = default_state()
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    if "queued_prompt" not in st.session_state:
        st.session_state.queued_prompt = None
    if "last_latency_ms" not in st.session_state:
        st.session_state.last_latency_ms = 0.0
    if "turn_count" not in st.session_state:
        st.session_state.turn_count = 0


def reset_session() -> None:
    defaults = default_state()
    for key, value in defaults.items():
        st.session_state[key] = value
    st.session_state.queued_prompt = None
    st.session_state.last_latency_ms = 0.0
    st.session_state.turn_count = 0
    st.rerun()


def apply_final_state(final_state: AgentState) -> None:
    sync_keys = [
        "intent", "conversation_stage", "is_qualified",
        "lead_name", "lead_email", "lead_platform",
        "missing_fields", "is_tool_called",
    ]
    for key in sync_keys:
        if key in final_state:
            st.session_state[key] = final_state[key]


# ---------------------------------------------------------------------------
# UI Components
# ---------------------------------------------------------------------------

def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("<div style='margin-bottom: 2rem;'></div>", unsafe_allow_html=True)
        st.markdown("### 🛠️ Controls")
        
        if st.button("Reset Session", use_container_width=True, type="secondary"):
            reset_session()

        st.markdown("<div style='margin-bottom: 2rem;'></div>", unsafe_allow_html=True)
        
        # System Info Card
        st.markdown("<div class='sidebar-card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-label'>Engine Status</div>", unsafe_allow_html=True)
        st.markdown(f"**Model:** `{MODEL_REFERENCE}`")
        st.markdown(f"**Orchestrator:** `LangGraph v0.2`")
        st.markdown(f"**Latency:** <span class='metric-value'>{st.session_state.last_latency_ms:.0f}ms</span>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Pipeline Card
        st.markdown("<div class='sidebar-card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-label'>Pipeline Execution</div>", unsafe_allow_html=True)
        
        stage_idx = {"greeting": 0, "inquiry": 1, "lead_collect": 2, "captured": 3}.get(st.session_state.conversation_stage, 0)
        
        cols = st.columns(4)
        labels = ["INT", "RAG", "LD", "RES"]
        for i, col in enumerate(cols):
            active = "active" if i <= stage_idx else ""
            col.markdown(f"<div class='pipeline-step {active}'></div>", unsafe_allow_html=True)
            col.caption(labels[i])
        st.markdown("</div>", unsafe_allow_html=True)

        # Lead Telemetry Card
        st.markdown("<div class='sidebar-card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-label'>Lead Profile</div>", unsafe_allow_html=True)
        
        name = st.session_state.lead_name or "Unknown"
        email = st.session_state.lead_email or "Unknown"
        platform = st.session_state.lead_platform or "Unknown"
        
        st.markdown(f"👤 **Name:** `{name}`")
        st.markdown(f"📧 **Email:** `{email}`")
        st.markdown(f"🌐 **Platform:** `{platform}`")
        
        missing = st.session_state.missing_fields
        if missing:
            st.warning(f"Missing: {', '.join(missing)}")
        elif st.session_state.is_tool_called:
            st.success("Lead Captured ✅")
        st.markdown("</div>", unsafe_allow_html=True)


def render_header() -> None:
    st.markdown(
        """
        <div class='project-header'>
            <div class='project-subtitle'>Agentic Workflow</div>
            <div class='project-title'>AutoStream Lead Intelligence</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_bar() -> None:
    intent = st.session_state.intent
    stage = st.session_state.conversation_stage.replace("_", " ").title()
    qualified = "Qualified" if st.session_state.is_qualified else "Analyzing"
    q_active = "active" if st.session_state.is_qualified else ""

    st.markdown(
        f"""
        <div class='status-bar'>
            <div class='status-item'><div class='status-dot active'></div> Intent: {intent}</div>
            <div class='status-item'><div class='status-dot active'></div> Stage: {stage}</div>
            <div class='status-item'><div class='status-dot {q_active}'></div> Lead: {qualified}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Main Logic
# ---------------------------------------------------------------------------

def main() -> None:
    init_session()
    render_sidebar()

    # Main Chat Area
    st.markdown("<div class='main-chat-container'>", unsafe_allow_html=True)
    render_header()
    render_status_bar()

    # Chat history
    for msg in st.session_state.messages:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        bubble_class = "user-bubble" if role == "user" else "assistant-bubble"
        with st.chat_message(role):
            st.markdown(f"<div class='message-bubble {bubble_class}'>{msg.content}</div>", unsafe_allow_html=True)

    # Input handling
    prompt = st.chat_input("Ask about AutoStream pricing or plans...")
    
    if prompt:
        st.session_state.messages.append(HumanMessage(content=prompt))
        st.session_state.turn_count += 1
        
        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(f"<div class='message-bubble user-bubble'>{prompt}</div>", unsafe_allow_html=True)

        # Process with agent
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                start = time.perf_counter()
                
                current_state: AgentState = {
                    "messages": st.session_state.messages,
                    "intent": st.session_state.intent,
                    "conversation_stage": st.session_state.conversation_stage,
                    "is_qualified": st.session_state.is_qualified,
                    "lead_name": st.session_state.lead_name,
                    "lead_email": st.session_state.lead_email,
                    "lead_platform": st.session_state.lead_platform,
                    "missing_fields": st.session_state.missing_fields,
                    "is_tool_called": st.session_state.is_tool_called,
                }

                try:
                    # Invoke LangGraph
                    final_state = agent_app.invoke(current_state)
                    
                    elapsed = (time.perf_counter() - start) * 1000.0
                    st.session_state.last_latency_ms = elapsed
                    
                    apply_final_state(final_state)
                    
                    ai_response = final_state["messages"][-1].content
                    st.session_state.messages.append(AIMessage(content=ai_response))
                    
                    st.markdown(f"<div class='message-bubble assistant-bubble'>{ai_response}</div>", unsafe_allow_html=True)
                    st.rerun()

                except Exception as exc:
                    st.error(f"Execution Error: {exc}")

    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
