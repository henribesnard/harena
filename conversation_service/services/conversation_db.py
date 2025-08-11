from __future__ import annotations

"""Database utilities for conversation management."""

from datetime import datetime
from typing import List, Optional

from sqlalchemy.orm import Session

from db_service.models.conversation import Conversation, ConversationTurn


class ConversationService:
    """Service layer for CRUD operations on conversations and turns."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def get_or_create_conversation(
        self, user_id: int, conversation_id: Optional[str] = None
    ) -> Conversation:
        """Return an existing conversation for user or create a new one.

        Args:
            user_id: ID of the authenticated user.
            conversation_id: Optional public identifier of the conversation.

        Raises:
            PermissionError: If conversation exists but belongs to another user.
        """
        if conversation_id:
            conversation = (
                self.db.query(Conversation)
                .filter(Conversation.conversation_id == conversation_id)
                .first()
            )
            if conversation:
                if conversation.user_id != user_id:
                    raise PermissionError("Conversation does not belong to user")
                return conversation

        conversation = Conversation(user_id=user_id)
        if conversation_id:
            conversation.conversation_id = conversation_id
        try:
            self.db.add(conversation)
            self.db.commit()
            self.db.refresh(conversation)
            return conversation
        except Exception:
            self.db.rollback()
            raise

    def add_turn(
        self,
        conversation: Conversation,
        user_message: str,
        assistant_response: str,
        processing_time_ms: float,
    ) -> ConversationTurn:
        """Persist a conversation turn and update conversation metadata."""
        turn_number = conversation.total_turns + 1
        turn = ConversationTurn(
            conversation_id=conversation.id,
            turn_number=turn_number,
            user_message=user_message,
            assistant_response=assistant_response,
            processing_time_ms=processing_time_ms,
        )
        try:
            self.db.add(turn)
            conversation.total_turns = turn_number
            conversation.last_activity_at = datetime.utcnow()
            self.db.commit()
            self.db.refresh(turn)
            self.db.refresh(conversation)
            return turn
        except Exception:
            self.db.rollback()
            raise

    def list_conversations(self, user_id: int) -> List[Conversation]:
        """List all conversations for a user ordered by recent activity."""
        return (
            self.db.query(Conversation)
            .filter(Conversation.user_id == user_id)
            .order_by(Conversation.last_activity_at.desc())
            .all()
        )

    def get_turns(self, conversation: Conversation) -> List[ConversationTurn]:
        """Return ordered turns for a conversation."""
        return (
            self.db.query(ConversationTurn)
            .filter(ConversationTurn.conversation_id == conversation.id)
            .order_by(ConversationTurn.turn_number)
            .all()
        )
