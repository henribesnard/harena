from __future__ import annotations

"""Simple in-memory team orchestrator for conversation agents."""

import time
from typing import Dict, List, Optional
from uuid import uuid4

from models.conversation_models import ConversationMessage
from conversation_service.core.metrics_collector import (
    MetricsCollector,
    metrics_collector,
)


class TeamOrchestrator:
    """Coordinate agents and manage conversation history."""

    def __init__(self, metrics: Optional[MetricsCollector] = None) -> None:
        self._conversations: Dict[str, List[ConversationMessage]] = {}
        self._metrics = metrics or metrics_collector

    def start_conversation(self, user_id: Optional[int] = None) -> str:
        """Create a new conversation and return its identifier."""
        start = time.time()
        conv_id = str(uuid4())
        self._conversations[conv_id] = []
        duration = (time.time() - start) * 1000
        self._metrics.record_orchestrator_call(
            operation="start_conversation", success=True, processing_time_ms=duration
        )
        return conv_id

    def get_history(self, conversation_id: str) -> Optional[List[ConversationMessage]]:
        """Return history for a conversation or ``None`` if not found."""
        start = time.time()
        history = self._conversations.get(conversation_id)
        duration = (time.time() - start) * 1000
        self._metrics.record_orchestrator_call(
            operation="get_history",
            success=history is not None,
            processing_time_ms=duration,
        )
        return history

    async def query_agents(self, conversation_id: str, message: str) -> str:
        """Process a message through the agent team.

        The current implementation is a placeholder that simply echoes the
        user's message. It also stores the user and assistant turns in memory.
        """
        start = time.time()
        history = self._conversations.setdefault(conversation_id, [])
        history.append(ConversationMessage(role="user", content=message))

        # Placeholder agent behaviour: echo the message
        reply = f"Echo: {message}"
        history.append(ConversationMessage(role="assistant", content=reply))

        duration = (time.time() - start) * 1000
        self._metrics.record_orchestrator_call(
            operation="query_agents", success=True, processing_time_ms=duration
        )
        return reply
