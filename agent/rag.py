"""RAG retrieval module — loads and searches the local knowledge base.

The retriever reads `data/knowledge_base.md` at import time and provides
a `retrieve_knowledge()` function that performs keyword-based semantic
matching to find the most relevant sections for a user query.
"""

import os
import re
from pathlib import Path


_KB_PATH = Path(__file__).resolve().parent.parent / "data" / "knowledge_base.md"

_knowledge_sections: list[dict[str, str]] = []


def _load_knowledge_base() -> list[dict[str, str]]:
    """Parse the Markdown knowledge base into titled sections."""
    if not _KB_PATH.exists():
        raise FileNotFoundError(f"Knowledge base not found at {_KB_PATH}")

    raw = _KB_PATH.read_text(encoding="utf-8")

    sections: list[dict[str, str]] = []
    current_title = "General"
    current_body: list[str] = []

    for line in raw.splitlines():
        if line.startswith("##"):
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


def _ensure_loaded() -> None:
    """Lazy-load knowledge base on first access."""
    global _knowledge_sections
    if not _knowledge_sections:
        _knowledge_sections = _load_knowledge_base()


def retrieve_knowledge(query: str) -> str:
    """Return relevant knowledge base content for the given query.

    Uses keyword overlap scoring to rank sections by relevance.
    Returns the top-3 matching sections concatenated as context.
    """
    _ensure_loaded()

    query_tokens = set(re.findall(r"\w+", query.lower()))

    scored: list[tuple[float, dict]] = []
    for section in _knowledge_sections:
        section_text = (section["title"] + " " + section["content"]).lower()
        section_tokens = set(re.findall(r"\w+", section_text))

        if not query_tokens:
            continue

        overlap = len(query_tokens & section_tokens)
        score = overlap / len(query_tokens)
        scored.append((score, section))

    scored.sort(key=lambda x: x[0], reverse=True)

    top_sections = scored[:3]
    results: list[str] = []
    for score, section in top_sections:
        if score > 0:
            results.append(f"### {section['title']}\n{section['content']}")

    if not results:
        return "No relevant information found in the knowledge base."

    return "\n\n".join(results)


def get_full_knowledge_base() -> str:
    """Return the entire knowledge base as a single string."""
    _ensure_loaded()
    parts = [f"### {s['title']}\n{s['content']}" for s in _knowledge_sections]
    return "\n\n".join(parts)
