from typing import List, Optional
from sqlalchemy.orm import Session, joinedload

from db_service.models.conversation import Conversation, ConversationTurn


class ConversationService:
    """Service layer for conversation read operations."""

    def __init__(self, db: Session):
        self.db = db

    def get_conversations(self, user_id: int, limit: int = 10, offset: int = 0) -> List[Conversation]:
        """Return conversations for a given user."""
        return (
            self.db.query(Conversation)
            .filter(Conversation.user_id == user_id)
            .order_by(Conversation.last_activity_at.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )

    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Return a conversation with its turns."""
        return (
            self.db.query(Conversation)
            .options(joinedload(Conversation.turns))
            .filter(Conversation.conversation_id == conversation_id)
            .first()
        )

    def get_conversation_turns(self, conversation_id: str) -> List[ConversationTurn]:
        """Return turns for a conversation."""
        conversation = self.get_conversation(conversation_id)
        return conversation.turns if conversation else []
