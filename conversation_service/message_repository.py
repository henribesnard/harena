"""Repository for persisting and retrieving conversation messages."""

from __future__ import annotations

from typing import List

from sqlalchemy.orm import Session

from db_service.models.conversation import ConversationMessage
from models.conversation_models import (
    ConversationMessage as ConversationMessageModel,
)


class ConversationMessageRepository:
    """Handle CRUD operations for :class:`ConversationMessage`."""

    def __init__(self, db: Session) -> None:
        self._db = db

    def add(
        self,
        *,
        conversation_id: str,
        user_id: int,
        role: str,
        content: str,
    ) -> ConversationMessage:
        msg = ConversationMessage(
            conversation_id=conversation_id,
            user_id=user_id,
            role=role,
            content=content,
        )
        self._db.add(msg)
        self._db.commit()
        self._db.refresh(msg)
        return msg

    def list_by_conversation(self, conversation_id: str) -> List[ConversationMessage]:
        return (
            self._db.query(ConversationMessage)
            .filter(ConversationMessage.conversation_id == conversation_id)
            .order_by(ConversationMessage.id)
            .all()
        )

    def list_models(self, conversation_id: str) -> List[ConversationMessageModel]:
        """Return conversation messages as API models.

        This helper converts the SQLAlchemy models returned by
        :meth:`list_by_conversation` into the pydantic models used by the
        public API.  It ensures callers don't need to be aware of the
        underlying ORM implementation and provides a serialisable structure
        suitable for passing to other services or agents.
        """

        return [
            ConversationMessageModel(role=m.role, content=m.content)
            for m in self.list_by_conversation(conversation_id)
        ]
