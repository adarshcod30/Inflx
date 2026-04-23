"""RAG retrieval module — loads and searches the local knowledge base.

Architecture:
  1. Loads ``data/knowledge_base.json`` at first access (lazy singleton).
  2. Flattens the JSON into titled *chunks* — one per plan, policy, and FAQ entry.
  3. At query time, performs **keyword-overlap scoring** to rank chunks.
  4. Returns the top-K most relevant chunks concatenated as grounding context.

Design rationale:
  - A lightweight keyword retriever avoids external vector-DB dependencies
    while still delivering accurate results for a bounded knowledge domain.
  - The chunking strategy mirrors how a production system would split documents
    before embedding, making migration to a vector store straightforward.

Also maintains the original Markdown knowledge base for backwards compatibility.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_BASE_DIR = Path(__file__).resolve().parent.parent
_KB_JSON_PATH = _BASE_DIR / "data" / "knowledge_base.json"
_KB_MD_PATH = _BASE_DIR / "data" / "knowledge_base.md"

# ---------------------------------------------------------------------------
# Chunk storage (lazy-loaded singleton)
# ---------------------------------------------------------------------------

_chunks: list[dict[str, str]] = []  # [{"title": …, "content": …}, …]


def _flatten_json_kb(data: dict) -> list[dict[str, str]]:
    """Convert structured JSON knowledge base into flat titled chunks.

    Each chunk has a ``title`` and a ``content`` string that can be
    matched against a user query and injected into the LLM prompt.
    """
    chunks: list[dict[str, str]] = []

    # Product overview
    if "product" in data:
        p = data["product"]
        chunks.append({
            "title": "Product Overview",
            "content": (
                f"{p.get('name', 'AutoStream')} — {p.get('tagline', '')}\n"
                f"{p.get('description', '')}"
            ),
        })

    # Plans
    for plan in data.get("plans", []):
        lines = [f"**{plan['name']}** — {plan['price']}"]
        lines.append(f"- Video Limit: {plan.get('video_limit', 'N/A')}")
        lines.append(f"- Max Resolution: {plan.get('max_resolution', 'N/A')}")
        lines.append(f"- Support: {plan.get('support', 'N/A')}")
        lines.append(f"- Export Formats: {', '.join(plan.get('export_formats', []))}")
        lines.append(f"- Storage: {plan.get('storage', 'N/A')}")
        if plan.get("ai_captions"):
            lines.append(f"- AI Captions: {plan.get('ai_captions_detail', 'Included')}")
        if plan.get("custom_branding"):
            lines.append("- Custom Branding: Included")
        if plan.get("team_collaboration"):
            lines.append(f"- Team Collaboration: Up to {plan.get('team_seats', 'N/A')} seats")
        chunks.append({
            "title": f"Pricing — {plan['name']}",
            "content": "\n".join(lines),
        })

    # Policies
    for key, policy in data.get("policies", {}).items():
        chunks.append({
            "title": policy.get("title", key.title()),
            "content": "\n".join(f"- {d}" for d in policy.get("details", [])),
        })

    # FAQs
    for faq in data.get("faq", []):
        chunks.append({
            "title": f"FAQ — {faq['question']}",
            "content": faq["answer"],
        })

    return chunks


def _load_md_kb() -> list[dict[str, str]]:
    """Fallback: parse the Markdown knowledge base into titled sections."""
    if not _KB_MD_PATH.exists():
        return []

    raw = _KB_MD_PATH.read_text(encoding="utf-8")
    sections: list[dict[str, str]] = []
    current_title = "General"
    current_body: list[str] = []

    for line in raw.splitlines():
        if line.startswith("## ") and not line.startswith("### "):
            if current_body:
                sections.append({
                    "title": current_title,
                    "content": "\n".join(current_body).strip(),
                })
            current_title = line.lstrip("#").strip()
            current_body = []
        else:
            current_body.append(line)

    if current_body:
        sections.append({
            "title": current_title,
            "content": "\n".join(current_body).strip(),
        })

    return sections


def _load_knowledge_base() -> list[dict[str, str]]:
    """Load knowledge base — prefer JSON, fall back to Markdown."""
    if _KB_JSON_PATH.exists():
        try:
            data = json.loads(_KB_JSON_PATH.read_text(encoding="utf-8"))
            chunks = _flatten_json_kb(data)
            logger.info("Loaded %d chunks from JSON knowledge base", len(chunks))
            return chunks
        except Exception as exc:
            logger.warning("JSON KB load failed (%s), falling back to Markdown", exc)

    md_chunks = _load_md_kb()
    if md_chunks:
        logger.info("Loaded %d sections from Markdown knowledge base", len(md_chunks))
        return md_chunks

    raise FileNotFoundError(
        f"No knowledge base found at {_KB_JSON_PATH} or {_KB_MD_PATH}"
    )


def _ensure_loaded() -> None:
    """Lazy-load knowledge base on first access."""
    global _chunks
    if not _chunks:
        _chunks = _load_knowledge_base()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def retrieve_knowledge(query: str, top_k: int = 3) -> str:
    """Return relevant knowledge base content for the given query.

    Uses keyword overlap scoring to rank chunks by relevance.
    Short queries are enriched with synonyms for better matching.

    Args:
        query: The user's natural-language question.
        top_k: Number of top-scoring chunks to return.

    Returns:
        A formatted string of the most relevant knowledge chunks,
        or a "not found" message if no chunk scores above zero.
    """
    _ensure_loaded()

    # --- Query enrichment: expand short queries with related terms ---
    _SYNONYMS = {
        "features": ["feature", "resolution", "captions", "branding", "collaboration", "export", "storage", "video", "support", "4k", "720p"],
        "pricing": ["price", "plan", "cost", "month", "29", "79", "basic", "pro"],
        "plans": ["plan", "basic", "pro", "price", "pricing", "cost", "29", "79", "month"],
        "policies": ["policy", "refund", "support", "cancellation", "cancel"],
        "refund": ["refund", "money", "back", "7", "days", "policy"],
        "support": ["support", "email", "chat", "priority", "24", "hours", "response"],
        "cancellation": ["cancel", "cancellation", "policy", "billing", "cycle"],
        "pro": ["pro", "plan", "79", "unlimited", "4k", "captions", "branding", "collaboration"],
        "basic": ["basic", "plan", "29", "720p", "10", "videos"],
        "trial": ["trial", "free", "7", "day", "credit"],
        "upgrade": ["upgrade", "basic", "pro", "prorated", "credit"],
    }

    enriched_query = query.lower()
    query_words = set(re.findall(r"\w+", enriched_query))

    # Expand short queries with synonyms
    if len(query_words) <= 3:
        extra_tokens = set()
        for word in list(query_words):
            if word in _SYNONYMS:
                extra_tokens.update(_SYNONYMS[word])
        query_words = query_words | extra_tokens

    if not query_words:
        return "No relevant information found in the knowledge base."

    scored: list[tuple[float, dict[str, str]]] = []
    for chunk in _chunks:
        chunk_text = (chunk["title"] + " " + chunk["content"]).lower()
        chunk_tokens = set(re.findall(r"\w+", chunk_text))
        overlap = len(query_words & chunk_tokens)
        # Use raw overlap count for ranking (not normalised by query size)
        # This prevents single-word queries from always scoring 1.0 on every chunk
        score = overlap
        scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)

    results: list[str] = []
    for score, chunk in scored[:top_k]:
        if score > 0:
            results.append(f"### {chunk['title']}\n{chunk['content']}")

    if not results:
        return "No relevant information found in the knowledge base."

    return "\n\n".join(results)


def get_full_knowledge_base() -> str:
    """Return the entire knowledge base as a single string."""
    _ensure_loaded()
    parts = [f"### {c['title']}\n{c['content']}" for c in _chunks]
    return "\n\n".join(parts)
