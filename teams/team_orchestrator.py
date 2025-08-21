from __future__ import annotations

"""Simple in-memory team orchestrator for conversation agents."""

from typing import Dict, List, Optional
from uuid import uuid4

from sqlalchemy.orm import Session

from models.conversation_models import ConversationMessage
from conversation_service.repository import ConversationRepository


class TeamOrchestrator:
    """Coordinate agents and manage conversation history."""

    def __init__(self) -> None:
        self._conversations: Dict[str, List[ConversationMessage]] = {}

    def start_conversation(self, user_id: int, db: Session) -> str:
        """Create a new conversation, persist it and return its identifier."""
        conv_id = str(uuid4())
        self._conversations[conv_id] = []
        ConversationRepository(db).create(
            user_id=user_id, conversation_id=conv_id
        )
        return conv_id

    def get_history(
        self, conversation_id: str
    ) -> Optional[List[ConversationMessage]]:
        """Return history for a conversation or ``None`` if not found."""
        return self._conversations.get(conversation_id)

    async def query_agents(self, conversation_id: str, message: str) -> str:
        """Process a message through the agent team.

        The current implementation is a placeholder that simply echoes the
        user's message. It also stores the user and assistant turns in memory.
        """
        history = self._conversations.setdefault(conversation_id, [])
        history.append(ConversationMessage(role="user", content=message))

        # Placeholder agent behaviour: echo the message
        reply = f"Echo: {message}"
        history.append(ConversationMessage(role="assistant", content=reply))
        return reply
