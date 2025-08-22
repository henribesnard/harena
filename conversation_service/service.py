from __future__ import annotations

"""High level operations for conversations and message persistence."""

from typing import List, Optional, Sequence

import sqlalchemy
from sqlalchemy.orm import Session

from db_service.models.conversation import Conversation
from conversation_service.message_repository import ConversationMessageRepository
from conversation_service.models.conversation_models import (
    ConversationMessage,
    MessageCreate,
)
from conversation_service.repository import ConversationRepository


class ConversationService:
    """Provide conversation helpers and atomic turn persistence."""

    def __init__(self, db: Session) -> None:
        self._db = db
        self._conv_repo = ConversationRepository(db)
        self._msg_repo = ConversationMessageRepository(db)

    # ------------------------------------------------------------------
    # Conversation helpers
    # ------------------------------------------------------------------
    def create_conversation(self, user_id: int, conversation_id: str) -> Conversation:
        """Create a new conversation instance and persist it."""
        conv = self._conv_repo.create(user_id, conversation_id)
        self._db.commit()
        return conv

    def get_for_user(self, conversation_id: str, user_id: int) -> Optional[Conversation]:
        """Return the conversation if owned by ``user_id``."""
        conv = self._conv_repo.get_by_conversation_id(conversation_id)
        if conv is None or conv.user_id != user_id:
            return None
        return conv

    def list_history(self, conversation_id: str) -> List[ConversationMessage]:
        """Return persisted messages for ``conversation_id``.

        If the messages table has not been created yet, an empty list is
        returned instead of raising an error.
        """
        try:
            return self._msg_repo.list_models(conversation_id)
        except sqlalchemy.exc.ProgrammingError:
            return []

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def save_conversation_turn_atomic(
        self,
        *,
        conversation_db_id: int,
        user_id: int,
        messages: Sequence[MessageCreate],
    ) -> None:
        """Persist a full conversation turn atomically."""
        self._msg_repo.add_batch(
            conversation_db_id=conversation_db_id,
            user_id=user_id,
            messages=messages,
        )


__all__ = ["ConversationService"]
