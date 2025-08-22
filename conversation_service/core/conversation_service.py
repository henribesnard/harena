"""High level conversation operations."""

from __future__ import annotations

from datetime import datetime, timezone
import uuid
from typing import Iterable, List, Optional, Tuple

import sqlalchemy
from sqlalchemy import update
from sqlalchemy.orm import Session

from conversation_service.message_repository import ConversationMessageRepository
from conversation_service.repository import ConversationRepository
from conversation_service.models.conversation_models import (
    ConversationMessage,
    MessageCreate,
)
from db_service.models.conversation import Conversation


class ConversationService:
    """Coordinate conversation and message persistence."""

    def __init__(self, db: Session) -> None:
        self._db = db
        self._conv_repo = ConversationRepository(db)
        self._msg_repo = ConversationMessageRepository(db)

    # --- Conversation management ----------------------------------------------
    def create_conversation(self, user_id: int) -> Conversation:
        """Create and persist a new conversation for ``user_id``."""

        conv_id = uuid.uuid4().hex
        try:
            conv = self._conv_repo.create(user_id, conv_id)
            self._db.commit()
            self._db.refresh(conv)
        except Exception:  # pragma: no cover - defensive rollback
            self._db.rollback()
            raise
        return conv

    def list_history(self, conversation_id: str) -> List[ConversationMessage]:
        """Return chronological message history for ``conversation_id``."""

        try:
            return self._msg_repo.list_models(conversation_id)
        except sqlalchemy.exc.ProgrammingError:
            return []

    # --- Conversation queries -------------------------------------------------
    def get_for_user(self, conversation_id: str, user_id: int) -> Optional[Conversation]:
        """Return the conversation if owned by ``user_id``."""

        conv = self._conv_repo.get_by_conversation_id(conversation_id)
        if conv is None or conv.user_id != user_id:
            return None
        return conv

    # --- Persistence ----------------------------------------------------------
    def save_conversation_turn_atomic(
        self,
        *,
        conversation: Conversation,
        user_message: str,
        agent_messages: Iterable[Tuple[str, str]] = (),
        assistant_reply: str,
    ) -> None:
        """Persist a complete conversation turn within one transaction.

        Parameters
        ----------
        conversation:
            ORM conversation instance to attach messages to. The instance is
            updated with the new turn count and last activity timestamp.
        user_message:
            Message originating from the user.
        agent_messages:
            Optional intermediate agent messages as ``(role, content)`` pairs.
        assistant_reply:
            Final response returned to the user.
        """

        messages: List[MessageCreate] = [
            MessageCreate(role="user", content=user_message)
        ]
        messages.extend(
            MessageCreate(role=role, content=content)
            for role, content in agent_messages
        )
        messages.append(MessageCreate(role="assistant", content=assistant_reply))

        with self._db.begin():
            self._msg_repo.add_batch(
                conversation_db_id=conversation.id,
                user_id=conversation.user_id,
                messages=messages,
            )
            self._db.execute(
                update(Conversation)
                .where(Conversation.id == conversation.id)
                .values(
                    total_turns=Conversation.total_turns + 1,
                    last_activity_at=datetime.now(timezone.utc),
                )
            )

        self._db.refresh(conversation)

    def save_conversation_turn(
        self,
        *,
        conversation: Conversation,
        user_message: str,
        agent_messages: Iterable[Tuple[str, str]] = (),
        assistant_reply: str,
    ) -> None:
        """Public wrapper delegating to :meth:`save_conversation_turn_atomic`."""

        self.save_conversation_turn_atomic(
            conversation=conversation,
            user_message=user_message,
            agent_messages=agent_messages,
            assistant_reply=assistant_reply,
        )


__all__ = ["ConversationService"]
