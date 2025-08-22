"""Local stub implementations of chat agents.

These classes provide the minimal surface required by the test suite.
They do not implement any real chat functionality but allow the codebase
to run in environments where the `autogen-agentchat` package is absent.
"""
from __future__ import annotations

from typing import Any


class BaseChatAgent:
    """Simplified base chat agent.

    The real package exposes a richer API; this stub only stores provided
    arguments for compatibility purposes.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - trivial
        self.args = args
        self.kwargs = kwargs


class AssistantAgent(BaseChatAgent):
    """Stub assistant agent derived from :class:`BaseChatAgent`."""

    pass
