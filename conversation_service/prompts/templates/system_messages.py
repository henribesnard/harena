"""Prompt templates for system messages.

This module exposes versioned system message templates and utilities for
selecting between variants to support A/B testing.
"""
from __future__ import annotations

from typing import Dict
import random

DEFAULT_VERSION = "v1"
DEFAULT_VARIANT = "A"

_SYSTEM_MESSAGES: Dict[str, Dict[str, str]] = {
    "v1": {
        "A": "You are an AI assistant.",
        "B": "You are a helpful AI assistant.",
    },
    "v2": {
        "A": "You are an advanced AI assistant.",
    },
}


def get_system_message(version: str = DEFAULT_VERSION, variant: str | None = None, **kwargs) -> str:
    """Return a formatted system message for the given version and variant.

    When ``variant`` is ``None`` a random variant for the requested ``version``
    is selected. Additional ``kwargs`` are interpolated into the template using
    :py:meth:`str.format`.
    """

    version_messages = _SYSTEM_MESSAGES.get(version)
    if not version_messages:
        raise ValueError(f"Unknown system message version: {version}")

    if variant is None:
        variant = random.choice(list(version_messages))

    template = version_messages.get(variant)
    if template is None:
        raise ValueError(f"Unknown variant '{variant}' for version '{version}'")

    return template.format(**kwargs)


def available_system_message_versions() -> Dict[str, Dict[str, str]]:
    """Return the mapping of available versions and their variants."""

    return _SYSTEM_MESSAGES


__all__ = [
    "DEFAULT_VERSION",
    "DEFAULT_VARIANT",
    "get_system_message",
    "available_system_message_versions",
]
