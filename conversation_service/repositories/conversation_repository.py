"""Persistence layer for conversations and turns.

The repository wraps all database interactions for conversation history.
It combines read and write operations originally split across separate
services so that the API and team orchestrator can depend on a single,
well defined interface.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session, joinedload

from db_service.models.conversation import Conversation, ConversationTurn


class ConversationRepository:
    """Repository handling persistence of conversations and turns."""

    def __init__(self, db: Session) -> None:
        self.db = db

    # ------------------------------------------------------------------
    def get_or_create_conversation(
        self, user_id: int, conversation_id: Optional[str] = None
    ) -> Conversation:
        """Return an existing conversation for user or create a new one."""
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

    # ------------------------------------------------------------------
    def add_turn(
        self,
        conversation_id: str,
        user_id: int,
        user_message: str,
        assistant_response: str,
        processing_time_ms: float,
        intent_result: Optional[Dict[str, Any]] = None,
        agent_chain: Optional[List[str]] = None,
        search_results_count: Optional[int] = None,
        confidence_score: Optional[float] = None,
        search_execution_time_ms: Optional[float] = None,
    ) -> ConversationTurn:
        """Persist a conversation turn and update metadata."""
        conversation = (
            self.db.query(Conversation)
            .filter(Conversation.conversation_id == conversation_id)
            .first()
        )
        if not conversation:
            raise ValueError("Conversation not found")
        if conversation.user_id != user_id:
            raise PermissionError("Conversation does not belong to user")

        turn_number = conversation.total_turns + 1
        turn = ConversationTurn(
            conversation_id=conversation.id,
            turn_number=turn_number,
            user_message=user_message,
            assistant_response=assistant_response,
            processing_time_ms=processing_time_ms,
            intent_result=intent_result,
            agent_chain=agent_chain or [],
            search_results_count=search_results_count or 0,
            confidence_score=confidence_score,
            search_execution_time_ms=search_execution_time_ms,
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

    # ------------------------------------------------------------------
    def list_conversations(self, user_id: int) -> List[Conversation]:
        """List all conversations for a user ordered by recent activity."""
        return (
            self.db.query(Conversation)
            .filter(Conversation.user_id == user_id)
            .order_by(Conversation.last_activity_at.desc())
            .all()
        )

    # ------------------------------------------------------------------
    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Return a conversation with its turns eagerly loaded."""
        return (
            self.db.query(Conversation)
            .options(joinedload(Conversation.turns))
            .filter(Conversation.conversation_id == conversation_id)
            .first()
        )

    # ------------------------------------------------------------------
    def get_conversation_turns(self, conversation_id: str) -> List[ConversationTurn]:
        """Return turns for a conversation."""
        conversation = self.get_conversation(conversation_id)
        return conversation.turns if conversation else []

    def get_turns(self, conversation: Conversation) -> List[ConversationTurn]:
        """Return ordered turns for a conversation instance."""
        return (
            self.db.query(ConversationTurn)
            .filter(ConversationTurn.conversation_id == conversation.id)
            .order_by(ConversationTurn.turn_number)
            .all()
        )
