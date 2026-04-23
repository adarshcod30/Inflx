"""Lead qualification engine for the AutoStream agent.

Encapsulates the logic for:
  - Detecting when a user transitions to HIGH_INTENT
  - Determining which lead fields are still missing
  - Generating the appropriate prompt for the next missing field
  - Deciding whether all criteria are met to trigger tool execution

This module is intentionally *stateless* — it reads from and writes to
the AgentState dict, but holds no internal mutable state.
"""

from __future__ import annotations

import logging
from typing import Any

from utils.helpers import is_valid_email

logger = logging.getLogger(__name__)

# Ordered list of fields the agent must collect before tool execution.
REQUIRED_FIELDS = ["name", "email", "platform"]

# Human-readable prompts for each missing field (used by respond_node).
FIELD_PROMPTS: dict[str, str] = {
    "name": (
        "I'd love to get you started with AutoStream Pro! "
        "Could you please share your **full name**?"
    ),
    "email": (
        "Thanks! Now, could you please provide your **professional email address** "
        "so our team can set up your account?"
    ),
    "platform": (
        "Almost there! Which **content creator platform** do you primarily use? "
        "(e.g., YouTube, Instagram, TikTok, Twitch, etc.)"
    ),
}


def compute_missing_fields(
    lead_name: str | None,
    lead_email: str | None,
    lead_platform: str | None,
) -> list[str]:
    """Return the ordered list of fields that have not yet been collected."""
    missing: list[str] = []
    if not lead_name:
        missing.append("name")
    if not lead_email:
        missing.append("email")
    if not lead_platform:
        missing.append("platform")
    return missing


def is_fully_qualified(
    lead_name: str | None,
    lead_email: str | None,
    lead_platform: str | None,
) -> bool:
    """Return True only when ALL three lead fields are present and valid."""
    if not lead_name or not lead_email or not lead_platform:
        return False
    if not is_valid_email(lead_email):
        return False
    return True


def next_field_prompt(missing_fields: list[str]) -> str:
    """Return the prompt string for the first missing field."""
    if not missing_fields:
        return ""
    field = missing_fields[0]
    return FIELD_PROMPTS.get(field, f"Could you please provide your {field}?")


def should_trigger_tool(state: dict[str, Any]) -> bool:
    """Gate check — returns True ONLY when it is safe to execute the lead capture tool.

    Conditions (ALL must be True):
      1. lead_name is not None/empty
      2. lead_email is a valid email
      3. lead_platform is not None/empty
      4. is_tool_called is False (hasn't been captured yet)
    """
    if state.get("is_tool_called"):
        return False

    name = state.get("lead_name")
    email = state.get("lead_email")
    platform = state.get("lead_platform")

    return is_fully_qualified(name, email, platform)
