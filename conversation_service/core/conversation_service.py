"""High level conversation operations."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable, List, Optional, Sequence, Tuple

from sqlalchemy import update
from sqlalchemy.orm import Session

from conversation_service.message_repository import ConversationMessageRepository
from conversation_service.repository import ConversationRepository
from conversation_service.models.conversation_models import MessageCreate
from db_service.models.conversation import Conversation


class ConversationService:
    """Coordinate conversation and message persistence."""

    def __init__(self, db: Session) -> None:
        self._db = db
        self._conv_repo = ConversationRepository(db)
        self._msg_repo = ConversationMessageRepository(db)

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

        class _Msg:
            def __init__(self, role: str, content: str) -> None:
                self.role = role
                self.content = content

        messages: List[object] = [MessageCreate(role="user", content=user_message)]
        for role, content in agent_messages:
            if not content.strip():
                raise ValueError("content must not be empty")
            messages.append(_Msg(role, content))
        messages.append(MessageCreate(role="assistant", content=assistant_reply))

        try:
            self._msg_repo.add_batch(
                conversation_db_id=conversation.id,
                user_id=conversation.user_id,
                messages=messages,  # type: ignore[arg-type]
            )
            self._db.execute(
                update(Conversation)
                    .where(Conversation.id == conversation.id)
                    .values(
                        total_turns=Conversation.total_turns + 1,
                        last_activity_at=datetime.now(timezone.utc),
                    )
            )
            self._db.commit()
            self._db.refresh(conversation)
        except Exception:  # pragma: no cover - defensive rollback
            self._db.rollback()
            raise


__all__ = ["ConversationService"]
