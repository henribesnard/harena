from __future__ import annotations

from typing import Optional

from sqlalchemy.orm import Session

from db_service.models.conversation import Conversation


class ConversationRepository:
    """Persist and retrieve conversation sessions."""

    def __init__(self, db: Session) -> None:
        self._db = db

    def create(self, user_id: int, conversation_id: str) -> Conversation:
        conv = Conversation(user_id=user_id, conversation_id=conversation_id)
        self._db.add(conv)
        # Flush the session so that an ID is assigned without committing the
        # transaction.  The surrounding service is responsible for committing
        # or rolling back the unit of work.
        self._db.flush()
        self._db.refresh(conv)
        return conv

    def get_by_conversation_id(
        self, conversation_id: str
    ) -> Optional[Conversation]:
        return (
            self._db.query(Conversation)
            .filter(Conversation.conversation_id == conversation_id)
            .first()
        )
