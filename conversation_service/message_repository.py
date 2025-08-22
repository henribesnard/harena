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

logger = logging.getLogger(__name__)


class ConversationMessageRepository:
    """Handle CRUD operations for :class:`ConversationMessage`."""

    def __init__(self, db: Session) -> None:
        self._db = db

    @contextmanager
    def transaction(self) -> Iterable[None]:
        """Context manager for DB transactions.

        Commits if the enclosed block succeeds, otherwise rolls back. In all
        cases the session is closed afterwards.
        """
        try:
            yield
            self._db.commit()
        except Exception:  # pragma: no cover - logging plus re-raise
            self._db.rollback()
            logger.exception("Database transaction failed; rolled back")
            raise
        finally:
            self._db.close()

    def _validate(self, *, conversation_db_id: int, user_id: int, content: str) -> None:
        if conversation_db_id <= 0 or user_id <= 0:
            raise ValueError("conversation_db_id and user_id must be positive")
        if not content or not content.strip():
            raise ValueError("content must be non-empty")

    def add(
        self,
        *,
        conversation_db_id: int,
        user_id: int,
        role: str,
        content: str,
    ) -> ConversationMessageDB:
        """Persist a single message."""

        MessageCreate(role=role, content=content)
        self._validate(
            conversation_db_id=conversation_db_id,
            user_id=user_id,
            content=content,
        )

        msg = ConversationMessageDB(
            conversation_id=conversation_db_id,
            user_id=user_id,
            role=role,
            content=content,
        )
        with self.transaction():
            self._db.add(msg)
            self._db.flush()
            self._db.refresh(msg)
        return msg

    def add_batch(
        self,
        *,
        conversation_db_id: int,
        user_id: int,
        messages: Sequence[MessageCreate],
    ) -> List[ConversationMessageDB]:
        """Persist multiple messages atomically."""

        instances: List[ConversationMessageDB] = []
        with self.transaction():
            for m in messages:
                MessageCreate(role=m.role, content=m.content)
                self._validate(
                    conversation_db_id=conversation_db_id,
                    user_id=user_id,
                    content=m.content,
                )
                obj = ConversationMessageDB(
                    conversation_id=conversation_db_id,
                    user_id=user_id,
                    role=m.role,
                    content=m.content,
                )
                self._db.add(obj)
                self._db.flush()
                self._db.refresh(obj)
                instances.append(obj)
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


__all__ = ["ConversationMessageRepository"]
