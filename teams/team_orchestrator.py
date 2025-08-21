from __future__ import annotations

"""Simple in-memory team orchestrator for conversation agents."""

import asyncio
import logging
from typing import Dict, List, Optional
from uuid import uuid4

import httpx

from models.conversation_models import ConversationMessage

logger = logging.getLogger(__name__)


class TeamOrchestrator:
    """Coordinate agents and manage conversation history."""

    def __init__(self) -> None:
        self._conversations: Dict[str, List[ConversationMessage]] = {}
        self._total_calls: int = 0
        self._error_calls: int = 0

    def start_conversation(self, user_id: Optional[int] = None) -> str:
        """Create a new conversation and return its identifier."""
        try:
            conv_id = str(uuid4())
            self._conversations[conv_id] = []
            return conv_id
        except Exception:  # pragma: no cover - unexpected failure
            logger.exception("Failed to start conversation")
            raise RuntimeError("Unable to start conversation at the moment")

    def get_history(self, conversation_id: str) -> Optional[List[ConversationMessage]]:
        """Return history for a conversation or ``None`` if not found."""
        try:
            return self._conversations.get(conversation_id)
        except Exception:  # pragma: no cover - guard against corrupt state
            logger.exception("Failed to retrieve conversation history")
            return None

    async def _call_agent(self, message: str, max_retries: int = 3) -> str:
        """Placeholder agent call with retry logic for LLM/HTTP operations."""
        last_error: Optional[Exception] = None
        for attempt in range(1, max_retries + 1):
            try:
                # In a real implementation, an HTTP/LLM request would be made here.
                async with httpx.AsyncClient() as _client:
                    await asyncio.sleep(0)  # Placeholder for external call
                return f"Echo: {message}"
            except Exception as exc:  # pragma: no cover - network path
                last_error = exc
                logger.warning(
                    "Agent call failed (attempt %s/%s): %s",
                    attempt,
                    max_retries,
                    exc,
                )
                if attempt < max_retries:
                    await asyncio.sleep(2 ** (attempt - 1))
        assert last_error is not None
        raise last_error

    async def query_agents(self, conversation_id: str, message: str) -> str:
        """Process a message through the agent team.

        Errors are logged and surfaced with user-friendly messages. The current
        behaviour echoes the user's message as a placeholder.
        """
        self._total_calls += 1
        try:
            history = self._conversations.setdefault(conversation_id, [])
            history.append(ConversationMessage(role="user", content=message))
        except Exception:
            self._error_calls += 1
            logger.exception("Failed to record user message")
            return "Une erreur est survenue lors de l'enregistrement du message."

        try:
            reply = await self._call_agent(message)
        except Exception:
            self._error_calls += 1
            logger.exception("Agent processing failed")
            reply = "Désolé, une erreur est survenue lors du traitement de votre demande."
        history.append(ConversationMessage(role="assistant", content=reply))
        return reply

    def get_error_metrics(self) -> Dict[str, float]:
        """Return basic error rate metrics."""
        return {
            "total_calls": float(self._total_calls),
            "error_calls": float(self._error_calls),
            "error_rate":
                self._error_calls / self._total_calls if self._total_calls else 0.0,
        }
