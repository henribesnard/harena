"""Repository for persisting and retrieving conversation messages."""

from __future__ import annotations

from typing import Iterable, List, Sequence

from sqlalchemy.orm import Session

from db_service.models.conversation import (
    Conversation,
    ConversationMessage as ConversationMessageDB,
)
from conversation_service.models.conversation_models import (
    ConversationMessage,
    MessageCreate,
)


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
        """Persist multiple messages without committing the transaction."""

        objs = [
            ConversationMessageDB(
                conversation_id=conversation_db_id,
                user_id=user_id,
                role=m.role,
                content=m.content,
            )
            for m in messages
        ]
        self._db.add_all(objs)
        self._db.flush()
        for obj in objs:
            self._db.refresh(obj)
        return objs

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


__all__ = ["ConversationMessageRepository"]
