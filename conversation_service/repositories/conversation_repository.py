"""Repository layer for conversations and turns.

This module exposes a high level API to manipulate conversation objects
stored in the database. It converts SQLAlchemy ORM objects to Pydantic
schemas for use in services or API layers.

Example:
    >>> from sqlalchemy.orm import Session
    >>> from conversation_service.schemas import ConversationCreate, ConversationTurnCreate
    >>> repo = ConversationRepository(db_session)  # doctest: +SKIP
    >>> conv = repo.create(ConversationCreate(user_id=1, title="Demo"))  # doctest: +SKIP
    >>> repo.add_turn(conv.conversation_id, ConversationTurnCreate(user_message="hi", assistant_response="hello"))  # doctest: +SKIP
    >>> repo.get_conversation(conv.conversation_id, user_id=1).total_turns  # doctest: +SKIP
    1
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import List
from sqlalchemy.orm import Session, selectinload

from db_service.models.conversation import Conversation as ConversationORM, ConversationTurn as ConversationTurnORM
from conversation_service.schemas import (
    Conversation,
    ConversationCreate,
    ConversationTurn,
    ConversationTurnCreate,
)


class ConversationRepository:
    """CRUD operations for :class:`Conversation` and its turns.

    Parameters
    ----------
    db: Session
        SQLAlchemy session used to talk to the database.
    """

    def __init__(self, db: Session) -> None:
        self.db = db

    # ------------------------------------------------------------------
    def create(self, conversation_in: ConversationCreate) -> Conversation:
        """Create a new conversation.

        Parameters
        ----------
        conversation_in: ConversationCreate
            Input Pydantic model containing conversation data.

        Returns
        -------
        Conversation
            Pydantic representation of the stored conversation.

        Example:
            >>> repo.create(ConversationCreate(user_id=1))  # doctest: +SKIP
            Conversation(...)
        """
        db_conv = ConversationORM(
            user_id=conversation_in.user_id,
            title=conversation_in.title,
            language=conversation_in.language,
            domain=conversation_in.domain,
            conversation_metadata=conversation_in.conversation_metadata,
            user_preferences=conversation_in.user_preferences,
            session_metadata=conversation_in.session_metadata,
        )
        self.db.add(db_conv)
        self.db.commit()
        self.db.refresh(db_conv)
        return Conversation.model_validate(db_conv, from_attributes=True)

    # ------------------------------------------------------------------
    def get_conversation(self, conversation_id: str, user_id: int) -> Conversation:
        """Fetch a conversation by its identifier.

        Raises
        ------
        ValueError
            If no conversation matches the criteria.

        Example:
            >>> repo.get_conversation("abc", user_id=1)  # doctest: +SKIP
            Conversation(...)
        """
        query = (
            self.db.query(ConversationORM)
            .options(selectinload(ConversationORM.turns))
            .filter(
                ConversationORM.conversation_id == conversation_id,
                ConversationORM.user_id == user_id,
            )
        )
        db_conv = query.first()
        if not db_conv:
            raise ValueError("Conversation not found")
        return Conversation.model_validate(db_conv, from_attributes=True)

    # ------------------------------------------------------------------
    def list_by_user(self, user_id: int, skip: int = 0, limit: int = 100) -> List[Conversation]:
        """List conversations belonging to a user.

        Example:
            >>> repo.list_by_user(user_id=1)  # doctest: +SKIP
            [Conversation(...), ...]
        """
        db_convs = (
            self.db.query(ConversationORM)
            .filter(ConversationORM.user_id == user_id)
            .offset(skip)
            .limit(limit)
            .all()
        )
        return [Conversation.model_validate(c, from_attributes=True) for c in db_convs]

    # ------------------------------------------------------------------
    def add_turn(self, conversation_id: str, turn_in: ConversationTurnCreate) -> ConversationTurn:
        """Append a new turn to a conversation.

        Parameters
        ----------
        conversation_id: str
            Public identifier of the conversation.
        turn_in: ConversationTurnCreate
            Pydantic object describing the turn to add.

        Returns
        -------
        ConversationTurn
            The newly created turn.

        Example:
            >>> repo.add_turn("abc", ConversationTurnCreate(user_message="hi", assistant_response="yo"))  # doctest: +SKIP
            ConversationTurn(...)
        """
        conv = (
            self.db.query(ConversationORM)
            .filter(ConversationORM.conversation_id == conversation_id)
            .first()
        )
        if not conv:
            raise ValueError("Conversation not found")

        next_turn_number = conv.total_turns + 1
        db_turn = ConversationTurnORM(
            conversation_id=conv.id,
            turn_number=next_turn_number,
            user_message=turn_in.user_message,
            assistant_response=turn_in.assistant_response,
            turn_metadata=turn_in.turn_metadata,
        )
        self.db.add(db_turn)
        conv.total_turns = next_turn_number
        conv.last_activity_at = datetime.now(timezone.utc)
        self.db.add(conv)
        self.db.commit()
        self.db.refresh(db_turn)
        return ConversationTurn.model_validate(db_turn, from_attributes=True)

    # ------------------------------------------------------------------
    def delete(self, conversation_id: str, user_id: int) -> None:
        """Remove a conversation.

        Example:
            >>> repo.delete("abc", user_id=1)  # doctest: +SKIP
        """
        conv = (
            self.db.query(ConversationORM)
            .filter(
                ConversationORM.conversation_id == conversation_id,
                ConversationORM.user_id == user_id,
            )
            .first()
        )
        if not conv:
            raise ValueError("Conversation not found")
        self.db.delete(conv)
        self.db.commit()
