"""Generic helper utilities."""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from typing import Optional


def generate_correlation_id() -> str:
    """Return a new UUID4 correlation identifier."""
    return str(uuid.uuid4())


def get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    """Fetch an environment variable."""
    return os.getenv(name, default)


def utc_now() -> datetime:
    """Current UTC time."""
    return datetime.now(timezone.utc)