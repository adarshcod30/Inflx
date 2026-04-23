"""Mock lead capture tool for the AutoStream agent.

This module simulates a CRM API call that would, in production,
push qualified leads into a system like HubSpot or Salesforce.

STRICT EXECUTION RULES:
  - The function is ONLY called after ALL three required fields
    (name, email, platform) have been collected and validated.
  - The caller (lead_node in graph.py) checks `should_trigger_tool()`
    from the lead module before invoking this function.
  - Duplicate execution is prevented by the `is_tool_called` gate.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from utils.helpers import is_valid_email

logger = logging.getLogger(__name__)


def mock_lead_capture(name: str, email: str, platform: str) -> dict[str, Any]:
    """Simulate capturing a qualified lead via a CRM API.

    In a production deployment this would POST to a CRM webhook
    (HubSpot, Salesforce, etc.). The mock version logs the record
    and returns a structured confirmation payload.

    Args:
        name: Full name of the lead.
        email: Validated email address.
        platform: Creator platform (YouTube, Instagram, etc.).

    Returns:
        dict with the lead record and a success flag.

    Raises:
        ValueError: If email validation fails (defence-in-depth).
    """
    # Defence-in-depth: validate email even though upstream should have checked
    if not is_valid_email(email):
        raise ValueError(f"Invalid email passed to lead capture tool: {email}")

    timestamp = datetime.now(timezone.utc).isoformat()

    lead_record: dict[str, Any] = {
        "name": name,
        "email": email,
        "platform": platform,
        "plan_interest": "Pro",
        "captured_at": timestamp,
        "source": "autostream_agent",
        "status": "new_lead",
    }

    # Structured log line (machine-parseable)
    logger.info(
        "LEAD_CAPTURED | %s",
        json.dumps(lead_record, ensure_ascii=False),
    )

    # Human-readable console output (required by spec)
    print(f"Lead captured successfully: {name}, {email}, {platform}")

    return {
        "success": True,
        "lead": lead_record,
        "message": (
            f"Lead captured successfully: {name} ({email}) "
            f"on {platform} at {timestamp}"
        ),
    }
