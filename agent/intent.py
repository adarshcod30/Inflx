"""Intent classification and entity extraction for the AutoStream agent.

Provides:
  - Pydantic models for structured intent output and lead detail extraction
  - Rule-based fallback classifier (`classify_intent_from_text`)
  - Entity extractors for email addresses and creator platforms
  - Name extraction heuristic for conversational lead capture

The rule-based classifier serves as both a standalone fallback when the LLM
is unavailable and a validation layer that cross-checks LLM outputs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from pydantic import BaseModel, field_validator


# =====================================================================
# Structured output models
# =====================================================================

class IntentClassification(BaseModel):
    """Structured output for intent classification."""

    intent: str        # GREETING | PRODUCT_QUERY | HIGH_INTENT
    confidence: float  # 0.0 to 1.0

    @field_validator("intent")
    @classmethod
    def validate_intent(cls, v: str) -> str:
        allowed = {"GREETING", "PRODUCT_QUERY", "HIGH_INTENT"}
        v = v.upper().strip()
        if v not in allowed:
            return "GREETING"
        return v


class LeadDetails(BaseModel):
    """Structured extraction of lead contact information."""

    name: str | None = None
    email: str | None = None
    platform: str | None = None

    @field_validator("email")
    @classmethod
    def validate_email_format(cls, v: str | None) -> str | None:
        if v is None:
            return None
        pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        if not re.match(pattern, v.strip()):
            return None
        return v.strip().lower()

    @field_validator("platform")
    @classmethod
    def normalize_platform(cls, v: str | None) -> str | None:
        if v is None:
            return None
        known = {
            "youtube": "YouTube",
            "instagram": "Instagram",
            "tiktok": "TikTok",
            "twitter": "Twitter/X",
            "x": "Twitter/X",
            "facebook": "Facebook",
            "twitch": "Twitch",
            "linkedin": "LinkedIn",
        }
        return known.get(v.strip().lower(), v.strip().title())


# =====================================================================
# Rule-based intent classifier (fallback)
# =====================================================================

# Signal keyword banks — ordered by priority
_HIGH_INTENT_SIGNALS = [
    "sign up", "signup", "subscribe", "buy", "purchase", "start",
    "get started", "try", "interested", "want to", "i want", "join",
    "register", "enroll", "ready", "let's go", "count me in",
    "i'd like to", "i would like", "take my money", "sign me up",
    "i'm in", "im in", "let's do it",
]

_PRODUCT_SIGNALS = [
    "price", "pricing", "cost", "plan", "feature", "basic", "pro",
    "compare", "comparison", "difference", "refund", "support", "cancel",
    "trial", "how much", "what do", "what does", "tell me about",
    "details", "resolution", "video", "caption", "storage", "export",
    "upgrade", "downgrade", "free trial", "team", "enterprise",
    "policy", "policies",
]

_GREETING_SIGNALS = [
    "hi", "hello", "hey", "good morning", "good afternoon",
    "good evening", "howdy", "greetings", "what's up", "sup",
    "yo", "hiya", "namaste",
]


def classify_intent_from_text(text: str) -> str:
    """Rule-based fallback for intent classification.

    Priority: HIGH_INTENT > PRODUCT_QUERY > GREETING > default PRODUCT_QUERY

    Used when the LLM-based classification is unavailable or as
    a secondary validation layer.
    """
    text_lower = text.lower().strip()

    # 1. High-intent signals (strongest)
    if any(signal in text_lower for signal in _HIGH_INTENT_SIGNALS):
        return "HIGH_INTENT"

    # 2. Product/feature query signals
    if any(signal in text_lower for signal in _PRODUCT_SIGNALS):
        return "PRODUCT_QUERY"

    # 3. Greeting signals (must start with the phrase)
    if any(text_lower.startswith(s) for s in _GREETING_SIGNALS):
        return "GREETING"

    # 4. Default to PRODUCT_QUERY (safer than GREETING for unknown inputs)
    return "PRODUCT_QUERY"


# =====================================================================
# Entity extractors
# =====================================================================

_EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")

_PLATFORM_MAP = {
    "youtube": "YouTube",
    "instagram": "Instagram",
    "tiktok": "TikTok",
    "twitter": "Twitter/X",
    "x": "Twitter/X",
    "facebook": "Facebook",
    "twitch": "Twitch",
    "linkedin": "LinkedIn",
}


def extract_email(text: str) -> str | None:
    """Extract the first valid email address from free-form text."""
    match = _EMAIL_PATTERN.search(text)
    return match.group(0).lower() if match else None


def extract_platform(text: str) -> str | None:
    """Extract a creator platform mention from text.

    Returns the normalised platform name (e.g. 'YouTube') or None.
    """
    text_lower = text.lower()
    for key, value in _PLATFORM_MAP.items():
        if key in text_lower:
            return value
    return None


def extract_name(text: str, stage: str | None = None, missing: list[str] | None = None) -> str | None:
    """Heuristic name extraction from user messages.

    Strategy:
      1. Pattern match on 'My name is X', 'I am X', 'I'm X', etc.
      2. If we are in lead_collect stage and name is the expected field,
         treat short messages (≤3 words) as name input.
    """
    text_clean = text.strip()

    # Noise words that should not be accepted as names
    _NOISE = {
        "sure", "yes", "no", "ok", "okay", "here", "it", "the", "a",
        "thanks", "thank", "please", "yep", "yeah", "nah", "nope",
    }

    # Pattern 1: explicit name statement
    patterns = [
        r"(?:my name is|i am|i'm|this is|call me|it's|its)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
        r"(?:my name is|i am|i'm|this is|call me|it's|its)\s+(\w+(?:\s+\w+)?)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text_clean, re.IGNORECASE)
        if match:
            name = match.group(1).strip()
            if name.lower() not in _NOISE and len(name) > 1:
                return name.title()

    # Pattern 2: contextual — we're actively asking for name
    if stage == "lead_collect" and missing and "name" in missing:
        words = text_clean.split()
        if len(words) <= 3:
            # Reject if it looks like an email or platform
            if "@" not in text_clean and not extract_platform(text_clean):
                if text_clean.lower() not in _NOISE and len(text_clean) > 1:
                    return text_clean.title()

    return None
