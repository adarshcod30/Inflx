"""AutoStream Conversational Agent — Streamlit Dashboard.

A production-grade UI for the Social-to-Lead agentic workflow.
Features real-time agent telemetry, conversation state visualization,
and a premium dark-mode interface.
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
    page_title="AutoStream Agent Console",
    page_icon="https://img.icons8.com/fluency/48/artificial-intelligence.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Premium CSS
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root {
        --bg-primary: #06090f;
        --bg-secondary: #0c1220;
        --bg-card: rgba(12, 18, 32, 0.85);
        --bg-elevated: rgba(20, 30, 55, 0.6);
        --border: rgba(99, 132, 255, 0.15);
        --border-active: rgba(99, 132, 255, 0.35);
        --text-primary: #e8ecf4;
        --text-secondary: #8896b3;
        --text-muted: #5a6a8a;
        --accent-blue: #6384ff;
        --accent-cyan: #22d3ee;
        --accent-emerald: #34d399;
        --accent-amber: #f59e0b;
        --accent-rose: #f43f5e;
        --gradient-primary: linear-gradient(135deg, #6384ff, #22d3ee);
        --gradient-success: linear-gradient(135deg, #34d399, #22d3ee);
        --gradient-warm: linear-gradient(135deg, #f59e0b, #f43f5e);
        --radius-sm: 8px;
        --radius-md: 14px;
        --radius-lg: 20px;
        --radius-xl: 28px;
        --shadow-glow: 0 0 40px rgba(99, 132, 255, 0.08);
    }

    html, body, [class*="st-"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .stApp {
        color: var(--text-primary);
        background:
            radial-gradient(ellipse 1400px 600px at 5% 0%, rgba(99, 132, 255, 0.08), transparent 70%),
            radial-gradient(ellipse 1000px 500px at 95% 0%, rgba(34, 211, 238, 0.06), transparent 65%),
            radial-gradient(ellipse 600px 400px at 50% 100%, rgba(244, 63, 94, 0.04), transparent 60%),
            linear-gradient(160deg, var(--bg-primary), var(--bg-secondary));
        min-height: 100vh;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(6, 9, 15, 0.95) !important;
        border-right: 1px solid var(--border) !important;
        backdrop-filter: blur(20px);
    }

    [data-testid="stSidebar"] .stMarkdown h2 {
        background: var(--gradient-primary);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        letter-spacing: -0.02em;
    }

    /* Hero banner */
    .hero-container {
        position: relative;
        border: 1px solid var(--border);
        background: var(--bg-card);
        border-radius: var(--radius-xl);
        padding: 28px 32px;
        margin-bottom: 20px;
        backdrop-filter: blur(12px);
        box-shadow: var(--shadow-glow);
        overflow: hidden;
    }

    .hero-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: var(--gradient-primary);
        border-radius: var(--radius-xl) var(--radius-xl) 0 0;
    }

    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        line-height: 1.1;
        margin: 0;
        background: linear-gradient(135deg, #ffffff, #c4d0ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-subtitle {
        margin: 8px 0 0 0;
        color: var(--text-secondary);
        font-size: 0.92rem;
        font-weight: 400;
        line-height: 1.5;
    }

    .badge-row {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 16px;
    }

    .badge {
        padding: 5px 12px;
        border-radius: 999px;
        font-size: 0.7rem;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        border: 1px solid var(--border);
        background: var(--bg-elevated);
        color: var(--text-secondary);
        transition: all 0.2s ease;
    }

    .badge-blue { border-color: rgba(99,132,255,0.3); color: var(--accent-blue); }
    .badge-cyan { border-color: rgba(34,211,238,0.3); color: var(--accent-cyan); }
    .badge-emerald { border-color: rgba(52,211,153,0.3); color: var(--accent-emerald); }
    .badge-amber { border-color: rgba(245,158,11,0.3); color: var(--accent-amber); }

    /* Panels */
    .tele-panel {
        border: 1px solid var(--border);
        background: var(--bg-card);
        border-radius: var(--radius-lg);
        padding: 20px;
        margin-bottom: 14px;
        backdrop-filter: blur(8px);
        box-shadow: var(--shadow-glow);
    }

    .tele-panel-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 6px;
    }

    .tele-panel-icon {
        width: 32px;
        height: 32px;
        border-radius: var(--radius-sm);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1rem;
    }

    .tele-panel-title {
        color: var(--text-primary);
        font-weight: 700;
        font-size: 0.95rem;
        letter-spacing: -0.01em;
    }

    .tele-panel-meta {
        color: var(--text-muted);
        font-size: 0.75rem;
        margin-bottom: 14px;
    }

    /* Mini metric cards */
    .metric-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 10px;
    }

    .metric-card {
        border: 1px solid var(--border);
        border-radius: var(--radius-md);
        padding: 12px 14px;
        background: rgba(6, 9, 15, 0.5);
        transition: border-color 0.3s ease, transform 0.2s ease;
    }

    .metric-card:hover {
        border-color: var(--border-active);
        transform: translateY(-1px);
    }

    .metric-label {
        color: var(--text-muted);
        font-size: 0.62rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 600;
        margin-bottom: 4px;
    }

    .metric-value {
        color: var(--text-primary);
        font-size: 0.95rem;
        font-weight: 700;
    }

    .metric-value.active { color: var(--accent-emerald); }
    .metric-value.waiting { color: var(--accent-amber); }
    .metric-value.captured { color: var(--accent-cyan); }
    .metric-value.empty { color: var(--text-muted); }

    /* Progress bar */
    .lead-progress-track {
        width: 100%;
        height: 6px;
        background: rgba(99, 132, 255, 0.1);
        border-radius: 3px;
        margin: 12px 0 6px 0;
        overflow: hidden;
    }

    .lead-progress-fill {
        height: 100%;
        border-radius: 3px;
        background: var(--gradient-success);
        transition: width 0.5s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .lead-progress-label {
        color: var(--text-muted);
        font-size: 0.7rem;
        font-weight: 500;
    }

    /* Pipeline visualization */
    .pipeline-row {
        display: flex;
        align-items: center;
        gap: 4px;
        margin: 16px 0 8px 0;
    }

    .pipe-node {
        flex: 1;
        text-align: center;
        padding: 8px 4px;
        border-radius: var(--radius-sm);
        font-size: 0.62rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        border: 1px solid var(--border);
        background: rgba(6, 9, 15, 0.4);
        color: var(--text-muted);
        transition: all 0.3s ease;
    }

    .pipe-node.done {
        border-color: rgba(52, 211, 153, 0.4);
        color: var(--accent-emerald);
        background: rgba(52, 211, 153, 0.08);
    }

    .pipe-node.active {
        border-color: rgba(99, 132, 255, 0.5);
        color: var(--accent-blue);
        background: rgba(99, 132, 255, 0.1);
        box-shadow: 0 0 12px rgba(99, 132, 255, 0.15);
    }

    .pipe-arrow {
        color: var(--text-muted);
        font-size: 0.6rem;
    }

    /* Chat messages */
    [data-testid="stChatMessage"] {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius-lg) !important;
        padding: 16px !important;
        margin-bottom: 10px !important;
        backdrop-filter: blur(6px);
    }

    /* Quick action buttons */
    .qa-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 8px;
        margin: 12px 0 18px 0;
    }

    /* Streamlit button overrides */
    .stButton > button {
        border-radius: var(--radius-md) !important;
        border: 1px solid var(--border) !important;
        background: var(--bg-elevated) !important;
        color: var(--text-secondary) !important;
        font-weight: 600 !important;
        font-size: 0.82rem !important;
        transition: all 0.25s ease !important;
        padding: 8px 16px !important;
    }

    .stButton > button:hover {
        border-color: var(--accent-blue) !important;
        color: var(--text-primary) !important;
        background: rgba(99, 132, 255, 0.1) !important;
        box-shadow: 0 0 20px rgba(99, 132, 255, 0.1) !important;
    }

    .stChatInputContainer {
        border-top: 1px solid var(--border) !important;
        background: rgba(6, 9, 15, 0.9) !important;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background: var(--bg-elevated) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius-md) !important;
    }

    /* Success/Info boxes */
    .stAlert {
        border-radius: var(--radius-md) !important;
    }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb {
        background: rgba(99, 132, 255, 0.2);
        border-radius: 3px;
    }

    /* Responsive */
    @media (max-width: 920px) {
        .hero-title { font-size: 1.6rem; }
        .metric-grid { grid-template-columns: 1fr; }
        .qa-grid { grid-template-columns: 1fr; }
        .pipeline-row { flex-wrap: wrap; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------

def init_session() -> None:
    """Initialize all session state keys with safe defaults."""
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
    """Clear all agent state and restart the conversation."""
    defaults = default_state()
    for key, value in defaults.items():
        st.session_state[key] = value
    st.session_state.queued_prompt = None
    st.session_state.last_latency_ms = 0.0
    st.session_state.turn_count = 0
    st.rerun()


def queue_prompt(text: str) -> None:
    """Queue a prompt for processing on next rerun."""
    st.session_state.queued_prompt = text
    st.rerun()


def export_transcript() -> str:
    """Export conversation as a plain text transcript."""
    lines: list[str] = []
    for msg in st.session_state.messages:
        role = "User" if isinstance(msg, HumanMessage) else "Agent"
        lines.append(f"{role}: {msg.content}")
    return "\n\n".join(lines)


def apply_final_state(final_state: AgentState) -> None:
    """Merge LangGraph output back into Streamlit session state."""
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
    """Render the sidebar with controls, model info, and quick prompts."""
    st.sidebar.markdown("## Agent Controls")
    st.sidebar.caption("Command center for the AutoStream conversational agent")

    # System info panel
    with st.sidebar.container(border=True):
        st.markdown(f"**Model:** `{MODEL_REFERENCE}`")
        st.markdown("**Runtime:** LangGraph State Machine")
        st.markdown("**Framework:** LangChain + Streamlit")
        gate = "Completed" if st.session_state.is_tool_called else "Waiting"
        gate_color = "green" if st.session_state.is_tool_called else "orange"
        st.markdown(f"**Lead Gate:** :{gate_color}[{gate}]")

    # Quick prompts
    with st.sidebar.container(border=True):
        st.markdown("### Suggested Prompts")
        prompts = [
            ("Show pricing plans", "What are your Basic and Pro pricing plans?"),
            ("Compare Basic vs Pro", "Compare the Basic and Pro plans in detail."),
            ("Start signup flow", "I want to sign up for the Pro plan for my YouTube channel."),
            ("Refund policy", "What is your refund and cancellation policy?"),
            ("Free trial info", "Do you offer a free trial?"),
        ]
        for label, prompt_text in prompts:
            if st.button(label, use_container_width=True, key=f"sb_{label}"):
                queue_prompt(prompt_text)

    # Actions
    with st.sidebar.container(border=True):
        st.markdown("### Session")
        st.download_button(
            label="Export Transcript",
            data=export_transcript(),
            file_name="autostream_transcript.txt",
            mime="text/plain",
            use_container_width=True,
        )
        if st.button("Reset Conversation", use_container_width=True, type="secondary"):
            reset_session()


def render_hero() -> None:
    """Render the hero banner with live status badges."""
    turns = st.session_state.turn_count
    stage = st.session_state.conversation_stage.replace("_", " ").title()
    intent = st.session_state.intent

    st.markdown(
        f"""
        <div class="hero-container">
            <h1 class="hero-title">AutoStream Agent Console</h1>
            <p class="hero-subtitle">
                Intelligent conversational agent powered by LangGraph and Gemini.
                Ask product questions, compare plans, or start your onboarding — all in one place.
            </p>
            <div class="badge-row">
                <span class="badge badge-blue">MODEL: {MODEL_REFERENCE}</span>
                <span class="badge badge-cyan">INTENT: {intent}</span>
                <span class="badge badge-emerald">STAGE: {stage}</span>
                <span class="badge badge-amber">TURNS: {turns}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_pipeline_status() -> None:
    """Render the agentic pipeline node visualization."""
    stage = st.session_state.conversation_stage
    intent = st.session_state.intent
    is_captured = st.session_state.is_tool_called

    # Determine node states
    nodes = {
        "Intent": "done",
        "RAG": "done" if intent != "GREETING" or st.session_state.turn_count > 0 else "active",
        "Lead": "done" if is_captured else ("active" if stage == "lead_collect" else ""),
        "Respond": "done" if st.session_state.turn_count > 0 else "active",
    }

    if st.session_state.turn_count == 0:
        nodes = {k: "" for k in nodes}

    node_html = ""
    for i, (name, status) in enumerate(nodes.items()):
        css_class = f"pipe-node {status}" if status else "pipe-node"
        node_html += f'<div class="{css_class}">{name}</div>'
        if i < len(nodes) - 1:
            node_html += '<div class="pipe-arrow">&#9654;</div>'

    st.markdown(
        f"""
        <div class="tele-panel">
            <div class="tele-panel-header">
                <div class="tele-panel-title">Agent Pipeline</div>
            </div>
            <div class="tele-panel-meta">LangGraph node execution status</div>
            <div class="pipeline-row">{node_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_telemetry() -> None:
    """Render the agent telemetry panel with live metrics."""
    filled = sum(bool(v) for v in [
        st.session_state.lead_name,
        st.session_state.lead_email,
        st.session_state.lead_platform,
    ])
    pct = int((filled / 3) * 100)

    name_val = st.session_state.lead_name or "---"
    email_val = st.session_state.lead_email or "---"
    platform_val = st.session_state.lead_platform or "---"

    qualified_class = "active" if st.session_state.is_qualified else "waiting"
    qualified_text = "Qualified" if st.session_state.is_qualified else "Pending"
    tool_class = "captured" if st.session_state.is_tool_called else "waiting"
    tool_text = "Captured" if st.session_state.is_tool_called else "Waiting"

    latency = f"{st.session_state.last_latency_ms:.0f}ms"

    missing = st.session_state.missing_fields
    missing_text = ", ".join(missing) if missing else "None"

    st.markdown(
        f"""
        <div class="tele-panel">
            <div class="tele-panel-header">
                <div class="tele-panel-title">Agent Telemetry</div>
            </div>
            <div class="tele-panel-meta">Real-time lead qualification and workflow state</div>
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-label">Status</div>
                    <div class="metric-value {qualified_class}">{qualified_text}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Lead Gate</div>
                    <div class="metric-value {tool_class}">{tool_text}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Name</div>
                    <div class="metric-value {'active' if st.session_state.lead_name else 'empty'}">{name_val}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Email</div>
                    <div class="metric-value {'active' if st.session_state.lead_email else 'empty'}">{email_val}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Platform</div>
                    <div class="metric-value {'active' if st.session_state.lead_platform else 'empty'}">{platform_val}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Latency</div>
                    <div class="metric-value">{latency}</div>
                </div>
            </div>
            <div class="lead-progress-track">
                <div class="lead-progress-fill" style="width: {pct}%;"></div>
            </div>
            <div class="lead-progress-label">Lead profile: {filled}/3 fields collected — Missing: {missing_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Raw State Snapshot"):
        st.json({
            "intent": st.session_state.intent,
            "conversation_stage": st.session_state.conversation_stage,
            "is_qualified": st.session_state.is_qualified,
            "lead_name": st.session_state.lead_name,
            "lead_email": st.session_state.lead_email,
            "lead_platform": st.session_state.lead_platform,
            "missing_fields": st.session_state.missing_fields,
            "is_tool_called": st.session_state.is_tool_called,
            "turn_count": st.session_state.turn_count,
            "last_latency_ms": st.session_state.last_latency_ms,
        })


def render_quick_actions() -> None:
    """Render inline quick action buttons above the chat."""
    st.markdown("#### Quick Actions")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if st.button("What is the Pro plan?", use_container_width=True, key="qa_pro"):
            queue_prompt("What does the Pro plan include?")
    with c2:
        if st.button("Refund after 7 days?", use_container_width=True, key="qa_refund"):
            queue_prompt("Can I get a refund after 7 days?")
    with c3:
        if st.button("Free trial?", use_container_width=True, key="qa_trial"):
            queue_prompt("Do you offer a free trial?")
    with c4:
        if st.button("I want to sign up!", use_container_width=True, key="qa_signup"):
            queue_prompt("I want to buy the Pro plan for my YouTube channel.")


# ---------------------------------------------------------------------------
# Core conversation handler
# ---------------------------------------------------------------------------

def process_prompt(prompt: str) -> None:
    """Send a user message through the LangGraph pipeline."""
    st.session_state.messages.append(HumanMessage(content=prompt))
    st.session_state.turn_count += 1

    with st.chat_message("user", avatar="https://img.icons8.com/fluency/48/user-male-circle--v1.png"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="https://img.icons8.com/fluency/48/artificial-intelligence.png"):
        status_box = st.status(
            "Agent is analyzing intent and preparing response...",
            expanded=False,
        )
        start = time.perf_counter()

        try:
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

            final_state = agent_app.invoke(current_state)
            elapsed = (time.perf_counter() - start) * 1000.0
            st.session_state.last_latency_ms = elapsed

            apply_final_state(final_state)

            ai_response = final_state["messages"][-1].content
            st.session_state.messages.append(AIMessage(content=ai_response))

            status_box.update(
                label=f"Response ready ({elapsed:.0f}ms)",
                state="complete",
                expanded=False,
            )
            st.markdown(ai_response)

            if st.session_state.is_tool_called and st.session_state.conversation_stage == "captured":
                st.success("Lead captured and routed to onboarding pipeline.")

        except Exception as exc:
            status_box.update(
                label="Agent encountered an error",
                state="error",
                expanded=True,
            )
            st.error(f"Agent error: {exc}")


# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------

def main() -> None:
    """Application entry point."""
    init_session()
    render_sidebar()
    render_hero()

    left_col, right_col = st.columns([2.2, 1], gap="large")

    with left_col:
        render_quick_actions()

        if not st.session_state.messages:
            st.info(
                "Start a conversation by asking about AutoStream's pricing, "
                "features, or policies. The agent will guide you through the "
                "entire journey — from product discovery to lead capture."
            )

        # Render message history
        for msg in st.session_state.messages:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            avatar = (
                "https://img.icons8.com/fluency/48/user-male-circle--v1.png"
                if role == "user"
                else "https://img.icons8.com/fluency/48/artificial-intelligence.png"
            )
            with st.chat_message(role, avatar=avatar):
                st.markdown(msg.content)

        # Handle input
        queued = st.session_state.queued_prompt
        typed = st.chat_input("Ask about pricing, plans, support, or start signup...")

        prompt = typed or queued
        if prompt:
            st.session_state.queued_prompt = None
            process_prompt(prompt)

    with right_col:
        render_pipeline_status()
        render_telemetry()


if __name__ == "__main__":
    main()
