from __future__ import annotations

from typing import Sequence

from conversation_service.message_repository import ConversationMessageRepository
from conversation_service.models.conversation_models import MessageCreate


class ConversationService:
    """High level operations for conversations."""

    def __init__(self, repo: ConversationMessageRepository) -> None:
        self._repo = repo

    def save_conversation_turn(
        self,
        *,
        conversation_db_id: int,
        user_id: int,
        messages: Sequence[MessageCreate],
    ) -> None:
        """Persist a full conversation turn atomically.

        Parameters
        ----------
        conversation_db_id:
            Database identifier of the conversation.
        user_id:
            Identifier of the user owning the conversation.
        messages:
            Sequence of messages belonging to the turn, typically a user
            message followed by the assistant response.
        """

        self._repo.add_batch(
            conversation_db_id=conversation_db_id,
            user_id=user_id,
            messages=messages,
        )
"""Core operations for conversation persistence."""

import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session

from db_service.models.conversation import Conversation, ConversationTurn
from .repository import ConversationRepository


class ConversationService:
    """High level conversation operations."""

    def __init__(self, db: Session) -> None:
        self._db = db
        self._repo = ConversationRepository(db)

    def get_for_user(
        self, conversation_id: str, user_id: int
    ) -> Optional[Conversation]:
        """Return the conversation if owned by ``user_id``."""

        conv = self._repo.get_by_conversation_id(conversation_id)
        if conv is None or conv.user_id != user_id:
            return None
        return conv

    def save_conversation_turn(
        self,
        conversation: Conversation,
        user_message: str,
        assistant_response: str,
    ) -> ConversationTurn:
        """Persist a complete conversation turn atomically."""

        turn_number = conversation.total_turns + 1
        turn = ConversationTurn(
            turn_id=uuid.uuid4().hex,
            conversation_id=conversation.id,
            turn_number=turn_number,
            user_message=user_message,
            assistant_response=assistant_response,
        )
        conversation.total_turns = turn_number
        conversation.last_activity_at = datetime.now(timezone.utc)
        self._db.add(turn)
        self._db.add(conversation)
        self._db.commit()
        self._db.refresh(turn)
        return turn


__all__ = ["ConversationService"]
