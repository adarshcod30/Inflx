"""AutoStream AI Agent — Full-Stack Streamlit Dashboard.

Multi-page Streamlit application for the Social-to-Lead Agentic Workflow.
Pages:
  - Chat       : Conversational interface with funnel progress indicator
  - Dashboard  : Live agent state, lead profile, and session analytics
  - Architecture: Pipeline diagram, routing logic, and state schema
"""

import os
import time
from datetime import datetime

import dotenv
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from agent.graph import agent_app
from agent.state import AgentState, default_state

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

dotenv.load_dotenv()
MODEL_REFERENCE = os.getenv("AUTOSTREAM_MODEL", "gemini-3.1-flash-lite-preview")

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AutoStream | AI Lead Intelligence",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FUNNEL_STEPS = ["Greeting", "Inquiry", "Lead Collection", "Captured"]
STAGE_TO_INDEX: dict[str, int] = {
    "greeting": 0,
    "inquiry": 1,
    "lead_collect": 2,
    "captured": 3,
}

# ---------------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Fira+Code:wght@400;500&display=swap');

    :root {
        --bg-primary:   #0a0c10;
        --bg-secondary: #111418;
        --bg-card:      #1c2128;
        --bg-card2:     #161b22;
        --accent:       #2563eb;
        --accent-green: #10b981;
        --accent-amber: #f59e0b;
        --accent-red:   #ef4444;
        --text-primary: #f8fafc;
        --text-secondary: #94a3b8;
        --border:       rgba(255,255,255,0.08);
        --radius:       12px;
    }

    html, body, [class*="st-"] { font-family: 'Outfit', sans-serif; }

    .stApp { background-color: var(--bg-primary); color: var(--text-primary); }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: var(--bg-secondary) !important;
        border-right: 1px solid var(--border) !important;
    }
    [data-testid="stSidebar"] .stRadio label {
        color: var(--text-primary) !important;
        font-weight: 500;
    }

    /* Cards */
    .as-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
    }
    .as-card-label {
        font-size: 0.7rem;
        font-weight: 700;
        color: var(--accent);
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin-bottom: 0.6rem;
    }
    .as-card-value {
        font-size: 0.92rem;
        color: var(--text-primary);
        margin-bottom: 0.35rem;
        font-weight: 500;
    }

    /* Hero */
    .hero-wrap {
        text-align: center;
        padding: 2.5rem 0 1.5rem;
        border-bottom: 1px solid var(--border);
        margin-bottom: 1.5rem;
    }
    .hero-title {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(135deg, #fff 0%, #2563eb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.04em;
        margin-bottom: 0.3rem;
    }
    .hero-sub {
        color: var(--text-secondary);
        font-size: 0.82rem;
        letter-spacing: 0.22em;
        text-transform: uppercase;
        font-weight: 600;
    }

    /* Funnel */
    .funnel-wrap {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0;
        margin: 0 auto 1.8rem;
        max-width: 680px;
    }
    .funnel-step {
        display: flex;
        flex-direction: column;
        align-items: center;
        flex: 1;
        position: relative;
    }
    .funnel-step:not(:last-child)::after {
        content: '';
        position: absolute;
        top: 18px;
        left: 60%;
        width: 80%;
        height: 2px;
        background: var(--border);
        z-index: 0;
    }
    .funnel-step.done::after  { background: var(--accent-green); }
    .funnel-step.active::after { background: var(--border); }
    .funnel-dot {
        width: 36px; height: 36px;
        border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-size: 0.85rem; font-weight: 700;
        background: var(--bg-card2);
        border: 2px solid var(--border);
        color: var(--text-secondary);
        z-index: 1;
        position: relative;
    }
    .funnel-dot.done  { background: var(--accent-green); border-color: var(--accent-green); color: #fff; }
    .funnel-dot.active { background: var(--accent); border-color: var(--accent); color: #fff; box-shadow: 0 0 14px rgba(37,99,235,0.5); }
    .funnel-label {
        font-size: 0.7rem; font-weight: 600;
        color: var(--text-secondary);
        margin-top: 0.4rem;
        text-align: center;
        white-space: nowrap;
    }
    .funnel-label.active { color: var(--accent); }
    .funnel-label.done   { color: var(--accent-green); }

    /* Chat bubbles */
    .msg-wrap { display: flex; margin-bottom: 1.2rem; }
    .msg-wrap.user  { justify-content: flex-end; }
    .msg-wrap.agent { justify-content: flex-start; }
    .msg-bubble {
        max-width: 78%;
        padding: 0.9rem 1.2rem;
        border-radius: 14px;
        font-size: 0.95rem;
        line-height: 1.6;
        border: 1px solid var(--border);
    }
    .msg-bubble.user  { background: var(--accent); color: #fff; border-color: var(--accent); border-bottom-right-radius: 4px; }
    .msg-bubble.agent { background: var(--bg-card); color: var(--text-primary); border-bottom-left-radius: 4px; }
    .msg-role { font-size: 0.68rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.3rem; color: var(--text-secondary); }

    /* Lead capture card */
    .lead-card {
        background: linear-gradient(135deg, #0d2137 0%, #0a1628 100%);
        border: 1px solid var(--accent);
        border-radius: var(--radius);
        padding: 1.5rem;
        margin-top: 1.2rem;
    }
    .lead-card-header {
        font-size: 0.75rem; font-weight: 700;
        color: var(--accent-green);
        text-transform: uppercase; letter-spacing: 0.12em;
        margin-bottom: 1rem;
        display: flex; align-items: center; gap: 0.5rem;
    }
    .lead-card-row {
        display: flex; justify-content: space-between;
        padding: 0.45rem 0;
        border-bottom: 1px solid rgba(255,255,255,0.05);
        font-size: 0.88rem;
    }
    .lead-card-row:last-child { border-bottom: none; }
    .lead-card-key   { color: var(--text-secondary); font-weight: 500; }
    .lead-card-val   { color: var(--text-primary);   font-weight: 600; }

    /* Dashboard metric tiles */
    .metric-tile {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 1rem 1.2rem;
        margin-bottom: 0.75rem;
    }
    .metric-tile-label { font-size: 0.68rem; font-weight: 700; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.1em; }
    .metric-tile-value { font-size: 1.4rem; font-weight: 800; color: var(--text-primary); margin-top: 0.2rem; }
    .metric-tile-value.green  { color: var(--accent-green); }
    .metric-tile-value.blue   { color: var(--accent); }
    .metric-tile-value.amber  { color: var(--accent-amber); }

    /* Architecture page */
    .arch-section { margin-bottom: 2rem; }
    .arch-title {
        font-size: 1rem; font-weight: 700;
        color: var(--accent);
        text-transform: uppercase; letter-spacing: 0.1em;
        margin-bottom: 0.75rem;
        padding-bottom: 0.4rem;
        border-bottom: 1px solid var(--border);
    }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 5px; }
    ::-webkit-scrollbar-track { background: var(--bg-primary); }
    ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

    /* Streamlit overrides */
    .stChatInputContainer { background: transparent !important; padding-bottom: 2rem !important; }
    div[data-testid="stVerticalBlock"] > div { gap: 0 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------

def init_session() -> None:
    """Initialise all session state keys with safe defaults."""
    defaults = default_state()
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    if "messages" not in st.session_state or not st.session_state.messages:
        st.session_state.messages = [
            AIMessage(
                content=(
                    "Hello! I'm your AutoStream AI Agent powered by Gemini 3.1 Flash-Lite. "
                    "I can help you explore our pricing plans, compare features, or guide you "
                    "through the sign-up process. What can I do for you today?"
                )
            )
        ]

    if "last_latency_ms" not in st.session_state:
        st.session_state.last_latency_ms = 0.0

    if "lead_captured_at" not in st.session_state:
        st.session_state.lead_captured_at = None

    if "total_turns" not in st.session_state:
        st.session_state.total_turns = 0


def reset() -> None:
    """Clear all session state and restart."""
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.rerun()


def update_state(results: dict) -> None:
    """Sync LangGraph output back into Streamlit session state."""
    keys = [
        "intent", "conversation_stage", "is_qualified",
        "lead_name", "lead_email", "lead_platform",
        "is_tool_called", "missing_fields",
    ]
    for k in keys:
        if k in results:
            st.session_state[k] = results[k]

    # Stamp lead capture time on first capture
    if results.get("is_tool_called") and st.session_state.lead_captured_at is None:
        st.session_state.lead_captured_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")


# ---------------------------------------------------------------------------
# Shared components
# ---------------------------------------------------------------------------

def render_funnel(stage: str) -> None:
    """Render a 4-step horizontal funnel progress indicator."""
    active_idx = STAGE_TO_INDEX.get(stage, 0)

    steps_html = ""
    for i, label in enumerate(FUNNEL_STEPS):
        if i < active_idx:
            dot_cls = "done"
            lbl_cls = "done"
            icon = "✓"
        elif i == active_idx:
            dot_cls = "active"
            lbl_cls = "active"
            icon = str(i + 1)
        else:
            dot_cls = ""
            lbl_cls = ""
            icon = str(i + 1)

        # connector line class on the step (not last)
        step_cls = dot_cls if i < len(FUNNEL_STEPS) - 1 else ""

        steps_html += f"""
        <div class="funnel-step {step_cls}">
            <div class="funnel-dot {dot_cls}">{icon}</div>
            <div class="funnel-label {lbl_cls}">{label}</div>
        </div>
        """

    st.markdown(
        f"<div class='funnel-wrap'>{steps_html}</div>",
        unsafe_allow_html=True,
    )


def render_lead_card(inline: bool = False) -> None:
    """Render the structured lead capture confirmation card."""
    name = st.session_state.get("lead_name") or "—"
    email = st.session_state.get("lead_email") or "—"
    platform = st.session_state.get("lead_platform") or "—"
    captured_at = st.session_state.get("lead_captured_at") or "—"

    rows = [
        ("Name", name),
        ("Email", email),
        ("Platform", platform),
        ("Plan Interest", "Pro Plan"),
        ("Captured At", captured_at),
        ("Status", "New Lead"),
    ]

    rows_html = "".join(
        f"<div class='lead-card-row'>"
        f"<span class='lead-card-key'>{k}</span>"
        f"<span class='lead-card-val'>{v}</span>"
        f"</div>"
        for k, v in rows
    )

    st.markdown(
        f"""
        <div class='lead-card'>
            <div class='lead-card-header'>&#10003; Lead Captured Successfully</div>
            {rows_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar() -> str:
    """Render sidebar navigation and agent info. Returns selected page name."""
    with st.sidebar:
        st.markdown(
            "<div style='font-size:1.3rem;font-weight:800;margin-bottom:1.5rem;"
            "color:#f8fafc;letter-spacing:-0.02em;'>🎬 AutoStream</div>",
            unsafe_allow_html=True,
        )

        page = st.radio(
            "Navigation",
            ["Chat", "Dashboard", "Architecture"],
            index=0,
            label_visibility="collapsed",
        )

        st.markdown("<hr style='border-color:rgba(255,255,255,0.08);margin:1rem 0;'>", unsafe_allow_html=True)

        if st.button("Reset Conversation", use_container_width=True, type="secondary"):
            reset()

        st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)

        # Engine info card
        st.markdown(
            f"""
            <div class='as-card'>
                <div class='as-card-label'>Agent Engine</div>
                <div class='as-card-value'>Model: <b>{MODEL_REFERENCE}</b></div>
                <div class='as-card-value'>Runtime: LangGraph</div>
                <div class='as-card-value'>Latency: <span style='color:#2563eb;font-weight:700;'>
                    {st.session_state.get('last_latency_ms', 0):.0f} ms</span></div>
                <div class='as-card-value'>Turns: {st.session_state.get('total_turns', 0)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Lead status card
        name = st.session_state.get("lead_name") or "Pending"
        email = st.session_state.get("lead_email") or "Pending"
        platform = st.session_state.get("lead_platform") or "Pending"
        captured = st.session_state.get("is_tool_called", False)
        status_color = "#10b981" if captured else "#f59e0b"
        status_text = "Captured" if captured else "In Progress"

        st.markdown(
            f"""
            <div class='as-card'>
                <div class='as-card-label'>Lead Profile</div>
                <div class='as-card-value'>Name: <b>{name}</b></div>
                <div class='as-card-value'>Email: <b>{email}</b></div>
                <div class='as-card-value'>Platform: <b>{platform}</b></div>
                <div class='as-card-value'>Status: <span style='color:{status_color};font-weight:700;'>{status_text}</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    return page


# ---------------------------------------------------------------------------
# Page: Chat
# ---------------------------------------------------------------------------

def render_chat_page() -> None:
    """Main conversational interface with funnel and lead card."""
    # Hero
    st.markdown(
        """
        <div class='hero-wrap'>
            <div class='hero-sub'>Social-to-Lead Agentic Workflow</div>
            <div class='hero-title'>AutoStream Lead Intelligence</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Funnel progress
    render_funnel(st.session_state.get("conversation_stage", "greeting"))

    # Chat history
    for msg in st.session_state.messages:
        is_human = isinstance(msg, HumanMessage)
        role_cls = "user" if is_human else "agent"
        role_label = "You" if is_human else "AutoStream AI"
        st.markdown(
            f"""
            <div class='msg-wrap {role_cls}'>
                <div class='msg-bubble {role_cls}'>
                    <div class='msg-role'>{role_label}</div>
                    {msg.content}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Agent invocation (triggered after user sends a message)
    if st.session_state.messages and isinstance(st.session_state.messages[-1], HumanMessage):
        with st.spinner("Agent is thinking..."):
            start = time.perf_counter()
            state: AgentState = {
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
                final_results = agent_app.invoke(state)
                st.session_state.last_latency_ms = (time.perf_counter() - start) * 1000.0
                st.session_state.total_turns += 1
                update_state(final_results)
                ai_content = final_results["messages"][-1].content
                st.session_state.messages.append(AIMessage(content=ai_content))
                st.rerun()
            except Exception as exc:
                st.error(f"Agent error: {exc}")

    # Lead capture card (shown below chat when lead is captured)
    if st.session_state.get("is_tool_called"):
        render_lead_card()

    # Chat input
    prompt = st.chat_input("Ask about plans, pricing, or start your subscription...")
    if prompt:
        st.session_state.messages.append(HumanMessage(content=prompt))
        st.rerun()


# ---------------------------------------------------------------------------
# Page: Dashboard
# ---------------------------------------------------------------------------

def render_dashboard_page() -> None:
    """Live analytics dashboard: agent state, session metrics, lead profile."""
    st.markdown(
        """
        <div class='hero-wrap'>
            <div class='hero-sub'>Live Session Analytics</div>
            <div class='hero-title'>Agent Dashboard</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([2, 1.2, 1.2])

    # --- Col 1: Recent conversation ---
    with col1:
        st.markdown("<div class='arch-title'>Recent Conversation</div>", unsafe_allow_html=True)
        msgs = st.session_state.messages[-6:]
        if not msgs:
            st.markdown(
                "<div style='color:#94a3b8;font-size:0.9rem;'>No messages yet. Start chatting on the Chat page.</div>",
                unsafe_allow_html=True,
            )
        else:
            for msg in msgs:
                is_human = isinstance(msg, HumanMessage)
                role_cls = "user" if is_human else "agent"
                role_label = "You" if is_human else "AI"
                content = msg.content[:200] + ("..." if len(msg.content) > 200 else "")
                st.markdown(
                    f"""
                    <div class='msg-wrap {role_cls}' style='margin-bottom:0.6rem;'>
                        <div class='msg-bubble {role_cls}' style='font-size:0.82rem;padding:0.6rem 0.9rem;'>
                            <div class='msg-role'>{role_label}</div>
                            {content}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # --- Col 2: Live agent state ---
    with col2:
        st.markdown("<div class='arch-title'>Agent State</div>", unsafe_allow_html=True)

        intent = st.session_state.get("intent", "GREETING")
        stage = st.session_state.get("conversation_stage", "greeting").replace("_", " ").title()
        qualified = st.session_state.get("is_qualified", False)
        latency = st.session_state.get("last_latency_ms", 0.0)
        turns = st.session_state.get("total_turns", 0)

        intent_color = {"GREETING": "#94a3b8", "PRODUCT_QUERY": "#f59e0b", "HIGH_INTENT": "#10b981"}.get(intent, "#94a3b8")
        q_color = "#10b981" if qualified else "#94a3b8"

        st.markdown(
            f"""
            <div class='metric-tile'>
                <div class='metric-tile-label'>Intent</div>
                <div class='metric-tile-value' style='color:{intent_color};font-size:1rem;'>{intent}</div>
            </div>
            <div class='metric-tile'>
                <div class='metric-tile-label'>Stage</div>
                <div class='metric-tile-value blue' style='font-size:1rem;'>{stage}</div>
            </div>
            <div class='metric-tile'>
                <div class='metric-tile-label'>Qualified</div>
                <div class='metric-tile-value' style='color:{q_color};font-size:1rem;'>{"Yes" if qualified else "No"}</div>
            </div>
            <div class='metric-tile'>
                <div class='metric-tile-label'>Last Latency</div>
                <div class='metric-tile-value amber'>{latency:.0f} ms</div>
            </div>
            <div class='metric-tile'>
                <div class='metric-tile-label'>Total Turns</div>
                <div class='metric-tile-value'>{turns}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --- Col 3: Lead profile ---
    with col3:
        st.markdown("<div class='arch-title'>Lead Profile</div>", unsafe_allow_html=True)

        name = st.session_state.get("lead_name")
        email = st.session_state.get("lead_email")
        platform = st.session_state.get("lead_platform")
        captured = st.session_state.get("is_tool_called", False)

        def field_row(label: str, value, filled: bool) -> str:
            color = "#10b981" if filled else "#94a3b8"
            display = value if filled else "Pending"
            return (
                f"<div class='metric-tile'>"
                f"<div class='metric-tile-label'>{label}</div>"
                f"<div class='metric-tile-value' style='color:{color};font-size:0.9rem;'>{display}</div>"
                f"</div>"
            )

        st.markdown(
            field_row("Name", name, bool(name))
            + field_row("Email", email, bool(email))
            + field_row("Platform", platform, bool(platform)),
            unsafe_allow_html=True,
        )

        if captured:
            render_lead_card()
        else:
            # Show funnel progress inline
            render_funnel(st.session_state.get("conversation_stage", "greeting"))


# ---------------------------------------------------------------------------
# Page: Architecture
# ---------------------------------------------------------------------------

def render_architecture_page() -> None:
    """Pipeline diagram, routing logic table, and state schema."""
    st.markdown(
        """
        <div class='hero-wrap'>
            <div class='hero-sub'>System Design</div>
            <div class='hero-title'>Agent Architecture</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Overview ---
    st.markdown("<div class='arch-title'>Pipeline Overview</div>", unsafe_allow_html=True)
    st.markdown(
        """
        AutoStream uses a **4-node LangGraph state machine** to convert social media conversations
        into qualified business leads. Every user message flows through an intent classifier, an
        optional RAG retrieval step, an optional lead-field extractor, and finally a response
        generator — all sharing a single typed `AgentState` dict that persists across turns.

        The graph is **channel-agnostic**: the same `agent_app.invoke()` call powers both the
        Streamlit UI and any future WhatsApp / Slack integration. State is managed entirely inside
        LangGraph; no external database is required for a single-session deployment.
        """,
        unsafe_allow_html=True,
    )

    # --- Mermaid diagram ---
    st.markdown("<div class='arch-title' style='margin-top:1.5rem;'>Pipeline Flow Diagram</div>", unsafe_allow_html=True)
    st.markdown(
        """
```mermaid
graph TD
    A([User Message]) --> B[intent_node<br/>LLM + rule-based classifier]
    B -->|GREETING| E[respond_node<br/>LLM response generator]
    B -->|PRODUCT_QUERY| C[rag_node<br/>Keyword-overlap retriever]
    B -->|HIGH_INTENT| C
    C -->|PRODUCT_QUERY| E
    C -->|HIGH_INTENT + not captured| D[lead_node<br/>Field extractor + tool gate]
    C -->|HIGH_INTENT + captured| E
    D --> E
    E --> F([END])

    style A fill:#1c2128,stroke:#2563eb,color:#f8fafc
    style B fill:#1c2128,stroke:#2563eb,color:#f8fafc
    style C fill:#1c2128,stroke:#f59e0b,color:#f8fafc
    style D fill:#1c2128,stroke:#10b981,color:#f8fafc
    style E fill:#1c2128,stroke:#2563eb,color:#f8fafc
    style F fill:#1c2128,stroke:#2563eb,color:#f8fafc
```
        """,
        unsafe_allow_html=True,
    )

    # --- Routing logic table ---
    st.markdown("<div class='arch-title' style='margin-top:1.5rem;'>Routing Logic</div>", unsafe_allow_html=True)
    import pandas as pd
    routing_df = pd.DataFrame(
        [
            ["intent_node", "GREETING", "respond_node", "Skip RAG and lead collection"],
            ["intent_node", "PRODUCT_QUERY", "rag_node", "Retrieve knowledge base context"],
            ["intent_node", "HIGH_INTENT", "rag_node", "Retrieve context before lead collection"],
            ["rag_node", "PRODUCT_QUERY", "respond_node", "Context retrieved, generate response"],
            ["rag_node", "HIGH_INTENT (not captured)", "lead_node", "Extract lead fields"],
            ["rag_node", "HIGH_INTENT (captured)", "respond_node", "Lead already captured"],
            ["lead_node", "any", "respond_node", "Always proceeds to response"],
        ],
        columns=["From Node", "Condition", "To Node", "Reason"],
    )
    st.dataframe(routing_df, use_container_width=True, hide_index=True)

    # --- State schema ---
    st.markdown("<div class='arch-title' style='margin-top:1.5rem;'>AgentState Schema</div>", unsafe_allow_html=True)
    st.json(default_state())

    # --- Tech stack ---
    st.markdown("<div class='arch-title' style='margin-top:1.5rem;'>Technology Stack</div>", unsafe_allow_html=True)
    tech_df = pd.DataFrame(
        [
            ["LLM", "Google Gemini 3.1 Flash-Lite Preview", "gemini-3.1-flash-lite-preview"],
            ["Agent Framework", "LangGraph", ">=0.2.0"],
            ["LLM Bindings", "LangChain Google GenAI", ">=2.0.0"],
            ["UI", "Streamlit", ">=1.30.0"],
            ["Language", "Python", "3.9+"],
            ["Validation", "Pydantic", ">=2.0.0"],
            ["RAG", "Custom keyword-overlap (no vector DB)", "—"],
        ],
        columns=["Component", "Technology", "Version"],
    )
    st.dataframe(tech_df, use_container_width=True, hide_index=True)

    # --- WhatsApp deployment ---
    st.markdown("<div class='arch-title' style='margin-top:1.5rem;'>WhatsApp Deployment</div>", unsafe_allow_html=True)
    st.markdown(
        """
        Because the LangGraph backend is **channel-agnostic**, deploying to WhatsApp requires
        only a thin webhook adapter — the core agent logic is unchanged.

        **Architecture:**
        1. Register a **Twilio WhatsApp Sandbox** (or production number) at twilio.com.
        2. Deploy a lightweight **FastAPI** (or Flask) webhook server that exposes `POST /webhook`.
        3. Twilio forwards every incoming WhatsApp message to your webhook as an HTTP POST.
        4. The webhook reconstructs the `AgentState` from a persistent store (Redis / DynamoDB),
           calls `agent_app.invoke(state)`, and sends the AI response back via the Twilio REST API.
        5. The Streamlit UI remains a **separate deployment** — both channels share the same
           `agent_app` Python package.

        ```python
        # webhook.py (FastAPI example)
        from fastapi import FastAPI, Form
        from twilio.rest import Client
        from agent.graph import agent_app
        from agent.state import default_state

        app = FastAPI()
        twilio = Client()

        @app.post("/webhook")
        async def whatsapp_webhook(Body: str = Form(), From: str = Form()):
            state = load_state(From) or default_state()   # load from Redis
            state["messages"].append(HumanMessage(content=Body))
            result = agent_app.invoke(state)
            save_state(From, result)                       # persist to Redis
            reply = result["messages"][-1].content
            twilio.messages.create(body=reply, from_="whatsapp:+14155238886", to=From)
            return {"status": "ok"}
        ```
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    init_session()
    page = render_sidebar()

    if page == "Chat":
        render_chat_page()
    elif page == "Dashboard":
        render_dashboard_page()
    elif page == "Architecture":
        render_architecture_page()


if __name__ == "__main__":
    main()
