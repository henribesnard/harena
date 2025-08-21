from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence


@dataclass
class ChatMessage:
    """Simple chat message container."""

    content: str
    source: str


@dataclass
class Response:
    """Wrapper for an agent response message."""

    chat_message: ChatMessage


@dataclass
class TaskResult:
    """Collection of messages produced during a run."""

    messages: Sequence[Any]


class AssistantAgent(Protocol):
    """Protocol for assistant agents used by the orchestrator."""

    name: str

    async def on_messages(
        self, messages: Sequence[ChatMessage], cancellation_token: Any
    ) -> Response:
        """Handle a sequence of messages and produce a response."""

    async def on_reset(self, cancellation_token: Any) -> None:
        """Reset any internal state for a fresh conversation."""
