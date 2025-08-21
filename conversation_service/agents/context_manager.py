"""Context management utilities for agent collaboration.

This module defines a lightweight :class:`ContextManager` used to share
intermediate results between multiple conversational agents.  Each agent can
read from and write to the shared context, allowing downstream agents to enrich
their processing with information produced by upstream agents.

The manager simply wraps an internal dictionary and exposes convenience
methods for updating and retrieving the conversation context.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class ContextManager:
    """Shared context store for agent teams.

    The manager keeps a mutable dictionary that can be used by a team of agents
    to pass information such as detected intent, extracted entities, generated
    queries or final responses.
    """

    def __init__(self) -> None:
        self._context: Dict[str, Any] = {}

    def update(self, **kwargs: Any) -> None:
        """Update the context with the provided keyword arguments."""

        self._context.update(kwargs)

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """Retrieve a value from the context."""

        return self._context.get(key, default)

    def get_context(self) -> Dict[str, Any]:
        """Return a shallow copy of the current context."""

        return dict(self._context)

    def clear(self) -> None:
        """Remove all stored context entries."""

        self._context.clear()
