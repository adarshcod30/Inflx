"""Intent classification and lead detail extraction.

Uses structured output parsing from LLM responses to identify user intent
and extract lead information (name, email, platform) when provided.
"""

import re
from dataclasses import dataclass

from pydantic import BaseModel, EmailStr, field_validator


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
        # Basic email pattern check
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


def classify_intent_from_text(text: str) -> str:
    """Rule-based fallback for intent classification.

    Used when the LLM-based classification is unavailable or as
    a secondary validation layer.
    """
    text_lower = text.lower()

    high_intent_signals = [
        "sign up", "signup", "subscribe", "buy", "purchase", "start",
        "get started", "try", "interested", "want to", "i want", "join",
        "register", "enroll", "ready", "let's go", "count me in",
        "i'd like to", "i would like",
    ]

    product_signals = [
        "price", "pricing", "cost", "plan", "feature", "basic", "pro",
        "compare", "difference", "refund", "support", "cancel", "trial",
        "how much", "what do", "what does", "tell me about", "details",
        "resolution", "video", "caption", "storage", "export",
    ]

    greeting_signals = [
        "hi", "hello", "hey", "good morning", "good afternoon",
        "good evening", "howdy", "greetings", "what's up", "sup",
    ]

    if any(signal in text_lower for signal in high_intent_signals):
        return "HIGH_INTENT"

    if any(signal in text_lower for signal in product_signals):
        return "PRODUCT_QUERY"

    if any(text_lower.strip().startswith(s) for s in greeting_signals):
        return "GREETING"

    return "PRODUCT_QUERY"


def extract_email(text: str) -> str | None:
    """Extract first valid email address from text."""
    pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    match = re.search(pattern, text)
    return match.group(0).lower() if match else None


def extract_platform(text: str) -> str | None:
    """Extract creator platform mention from text."""
    platforms = {
        "youtube": "YouTube",
        "instagram": "Instagram",
        "tiktok": "TikTok",
        "twitter": "Twitter/X",
        "x": "Twitter/X",
        "facebook": "Facebook",
        "twitch": "Twitch",
        "linkedin": "LinkedIn",
    }
    text_lower = text.lower()
    for key, value in platforms.items():
        if key in text_lower:
            return value
    return None
