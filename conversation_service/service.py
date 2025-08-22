"""High level operations for conversations and their messages."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Iterable, List, Optional, Sequence, Tuple

from sqlalchemy.orm import Session

from db_service.models.conversation import Conversation, ConversationTurn
from conversation_service.message_repository import ConversationMessageRepository
from conversation_service.models.conversation_models import MessageCreate
from .repository import ConversationRepository


class ConversationService:
    """Unified service providing conversation persistence utilities."""

    def __init__(self, db: Session) -> None:
        self._db = db
        self._conv_repo = ConversationRepository(db)
        self._msg_repo = ConversationMessageRepository(db)

    # ------------------------------------------------------------------
    # Conversation metadata operations
    # ------------------------------------------------------------------
    def get_for_user(
        self, conversation_id: str, user_id: int
    ) -> Optional[Conversation]:
        """Return the conversation if owned by ``user_id``."""

        conv = self._conv_repo.get_by_conversation_id(conversation_id)
        if conv is None or conv.user_id != user_id:
            return None
        return conv

    def save_conversation_turn(
        self,
        conversation: Conversation,
        user_message: str,
        assistant_response: str,
    ) -> ConversationTurn:
        """Persist a complete conversation turn atomically."""

        turn_number = conversation.total_turns + 1
        turn = ConversationTurn(
            turn_id=uuid.uuid4().hex,
            conversation_id=conversation.id,
            turn_number=turn_number,
            user_message=user_message,
            assistant_response=assistant_response,
        )
        conversation.total_turns = turn_number
        conversation.last_activity_at = datetime.now(timezone.utc)
        self._db.add(turn)
        self._db.add(conversation)
        self._db.commit()
        self._db.refresh(turn)
        return turn

    # ------------------------------------------------------------------
    # Conversation message operations
    # ------------------------------------------------------------------
    def record_messages(
        self,
        *,
        conversation_db_id: int,
        user_id: int,
        user_message: str,
        agent_messages: Iterable[Tuple[str, str]],
        assistant_reply: str,
    ) -> None:
        """Persist a full conversation turn's messages atomically."""

        messages: List[MessageCreate] = [
            MessageCreate(role="user", content=user_message)
        ]
        messages.extend(
            MessageCreate(role=role, content=content)
            for role, content in agent_messages
        )
        messages.append(MessageCreate(role="assistant", content=assistant_reply))
        self._msg_repo.add_batch(
            conversation_db_id=conversation_db_id,
            user_id=user_id,
            messages=messages,
        )


__all__ = ["ConversationService"]
