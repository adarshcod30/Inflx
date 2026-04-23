"""Mock lead capture tool for the AutoStream agent.

This module simulates a CRM API call that would, in production,
push qualified leads into a system like HubSpot or Salesforce.
The function is only called after all three required fields
(name, email, platform) have been collected and validated.
"""

import json
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def mock_lead_capture(name: str, email: str, platform: str) -> str:
    """Simulate capturing a qualified lead via a CRM API.

    Args:
        name: Full name of the lead.
        email: Validated email address.
        platform: Creator platform (YouTube, Instagram, etc.).

    Returns:
        Confirmation string with captured lead details.
    """
    timestamp = datetime.now(timezone.utc).isoformat()

    lead_record = {
        "name": name,
        "email": email,
        "platform": platform,
        "plan_interest": "Pro",
        "captured_at": timestamp,
        "source": "autostream_agent",
        "status": "new_lead",
    }

    logger.info("Lead captured successfully: %s", json.dumps(lead_record))

    print(f"Lead captured successfully: {name}, {email}, {platform}")

    return (
        f"Lead captured successfully: {name} ({email}) "
        f"on {platform} at {timestamp}"
    )
