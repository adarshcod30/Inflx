"""Shared utility functions for the AutoStream Agentic Workflow.

Centralises reusable helpers — logging setup, timestamp formatting,
text sanitisation, and email validation — so that every module imports
them from a single source of truth.
"""

import logging
import re
import sys
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def configure_logging(level: int = logging.INFO) -> logging.Logger:
    """Return a project-wide logger with a clean console handler.

    Call once at application entry to set up logging for every module
    under the ``agent`` and ``utils`` packages.
    """
    root = logging.getLogger("autostream")
    if root.handlers:
        return root

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s] %(levelname)-8s %(name)s — %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root.addHandler(handler)
    root.setLevel(level)
    return root


# ---------------------------------------------------------------------------
# Timestamps
# ---------------------------------------------------------------------------

def utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def utc_now_display() -> str:
    """Return the current UTC time in a human-friendly format."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

def sanitise_text(text: str) -> str:
    """Strip leading/trailing whitespace and collapse internal whitespace."""
    return re.sub(r"\s+", " ", text.strip())


def truncate(text: str, max_len: int = 200) -> str:
    """Truncate *text* to *max_len* characters, appending '…' if trimmed."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

EMAIL_REGEX = re.compile(
    r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$"
)


def is_valid_email(email: str) -> bool:
    """Return True if *email* passes a basic RFC-style pattern check."""
    return bool(EMAIL_REGEX.match(email.strip()))


def mask_email(email: str) -> str:
    """Mask an email for privacy display, e.g. 'a***h@gmail.com'."""
    if "@" not in email:
        return email
    local, domain = email.split("@", 1)
    if len(local) <= 2:
        masked_local = local[0] + "***"
    else:
        masked_local = local[0] + "***" + local[-1]
    return f"{masked_local}@{domain}"
