"""Repository for persisting and retrieving conversation messages."""
from __future__ import annotations

import logging
from typing import List, Sequence

from sqlalchemy.orm import Session

from db_service.models.conversation import (
    Conversation,
    ConversationMessage as ConversationMessageDB,
)
from conversation_service.models.conversation_models import (
    ConversationMessage,
    MessageCreate,
)


logger = logging.getLogger(__name__)


class ConversationMessageRepository:
    """Handle CRUD operations for :class:`ConversationMessage`."""

    def __init__(self, db: Session) -> None:
        self._db = db

    def add_batch(
        self,
        *,
        conversation_db_id: int,
        user_id: int,
        messages: Sequence[MessageCreate],
    ) -> List[ConversationMessageDB]:
        """Persist multiple messages.

        Messages are added to the current session and flushed so that their
        identifiers are populated. Transaction management is handled by the
        caller; this method only validates and inserts records.
        """

        instances: List[ConversationMessageDB] = []
        for m in messages:
            self._validate(
                conversation_db_id=conversation_db_id,
                user_id=user_id,
                content=m.content,
            )
            msg = ConversationMessageDB(
                conversation_id=conversation_db_id,
                user_id=user_id,
                role=m.role,
                content=m.content,
            )
            self._db.add(msg)
            self._db.flush()
            self._db.refresh(msg)
            instances.append(msg)

        return instances

    def list_by_conversation(self, conversation_id: str) -> List[ConversationMessageDB]:
        """Return ORM messages for ``conversation_id`` ordered chronologically."""

        return (
            self._db.query(ConversationMessageDB)
            .join(Conversation, Conversation.id == ConversationMessageDB.conversation_id)
            .filter(Conversation.conversation_id == conversation_id)
            .order_by(ConversationMessageDB.created_at)
            .all()
        )

    def list_models(self, conversation_id: str) -> List[ConversationMessage]:
        """Return user/assistant messages as pydantic models."""

        return [
            ConversationMessage(
                user_id=m.user_id,
                conversation_id=conversation_id,
                role=m.role,
                content=m.content,
                timestamp=m.created_at,
            )
            for m in self.list_by_conversation(conversation_id)
            if m.role in {"user", "assistant"}
        ]

    def _validate(
        self,
        *,
        conversation_db_id: int,
        user_id: int,
        content: str,
    ) -> None:
        """Validate message content and identifiers.

        Currently ensures that the message content is not empty. Additional
        domain-specific validation can be added here.
        """
        if not content.strip():
            raise ValueError("content must not be empty")


__all__ = ["ConversationMessageRepository"]
