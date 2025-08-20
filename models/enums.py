"""Enumeration classes used by shared models."""
from __future__ import annotations

from enum import StrEnum


class AgentType(StrEnum):
    """Type of agent used in conversations."""

    ASSISTANT = "assistant"
    USER_PROXY = "user_proxy"
    CONVERSABLE = "conversable"


class MessageRole(StrEnum):
    """Role of a message within a conversation."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Currency(StrEnum):
    """Supported currency codes."""

    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"


__all__ = ["AgentType", "MessageRole", "Currency"]
