"""Repository for persisting and retrieving conversation messages."""
from __future__ import annotations

import logging
from contextlib import contextmanager
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
        """Persist multiple messages atomically.

        All messages are inserted in a single transaction. If any insertion
        fails, the transaction is rolled back and the exception propagated.
        """

        instances: List[ConversationMessageDB] = []
        with self.transaction():
            for m in messages:
                # Validate message fields and shared identifiers
                MessageCreate(role=m.role, content=m.content)
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

    def _validate(
        self,
        *,
        conversation_db_id: int,
        user_id: int,
        content: str,
    ) -> None:
        """Basic sanity checks for messages.

        Currently only ensures that the ``content`` field is not empty.
        """

        if not content.strip():
            raise ValueError("content must not be empty")

    @contextmanager
    def transaction(self):
        """Provide a database transaction context manager."""

        try:
            yield
            self._db.commit()
        except Exception:
            self._db.rollback()
            raise

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
