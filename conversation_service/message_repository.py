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
        """Return user/assistant messages as pydantic models.

        The underlying ORM model includes all internal agent messages.  For
        conversational context we only expose user and assistant messages,
        converting each row to the public :class:`ConversationMessage` model
        with an explicit timestamp.
        """

        return [
            ConversationMessageModel(
                user_id=m.user_id,
                conversation_id=m.conversation_id,
                role=m.role,
                content=m.content,
                timestamp=m.created_at,
            )
            for m in self.list_by_conversation(conversation_id)
            if m.role in {"user", "assistant"}
        ]
