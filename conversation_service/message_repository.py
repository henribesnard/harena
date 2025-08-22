"""Repository for persisting and retrieving conversation messages."""

from __future__ import annotations

from typing import List

from sqlalchemy.orm import Session

from db_service.models.conversation import (
    Conversation,
    ConversationMessage as ConversationMessageDB,
)

from conversation_service.models.conversation_models import ConversationMessage


class ConversationMessageRepository:
    """Handle CRUD operations for :class:`ConversationMessage`."""

    def __init__(self, db: Session) -> None:
        self._db = db

    def add(
        self,
        *,
        conversation_db_id: int,
        user_id: int,
        role: str,
        content: str,
    ) -> ConversationMessageDB:
        """Persist a new message to the database.

        Parameters mirror the columns of :class:`ConversationMessageDB` so
        that callers can explicitly state the ``conversation_id`` (via
        ``conversation_db_id``) and ``user_id`` associated with the message
        along with its ``role`` and textual ``content``.

        Returns
        -------
        ConversationMessageDB
            The newly created ORM instance with an assigned primary key and
            timestamps.
        """

        # Create and immediately persist the ORM model so that callers can
        # query it straight away (for example to build conversation history).
        msg = ConversationMessageDB(
            conversation_id=conversation_db_id,
            user_id=user_id,
            role=role,
            content=content,
        )
        self._db.add(msg)
        self._db.commit()
        self._db.refresh(msg)
        return msg

    def add_batch(
        self,
        *,
        conversation_db_id: int,
        user_id: int,
        messages: List[tuple[str, str]],
    ) -> List[ConversationMessageDB]:
        """Persist multiple messages in a single transaction.

        Parameters
        ----------
        conversation_db_id:
            Database identifier of the conversation.
        user_id:
            Identifier of the user owning the conversation.
        messages:
            Sequence of ``(role, content)`` tuples representing the messages to
            persist in order.

        Returns
        -------
        List[ConversationMessageDB]
            The ORM instances corresponding to the newly created messages.
        """

        objs = [
            ConversationMessageDB(
                conversation_id=conversation_db_id,
                user_id=user_id,
                role=role,
                content=content,
            )
            for role, content in messages
        ]
        self._db.add_all(objs)
        # Flush so that auto-generated fields (e.g., primary keys, timestamps)
        # are populated before returning. The surrounding transaction is
        # responsible for committing.
        self._db.flush()
        for obj in objs:
            self._db.refresh(obj)
        return objs

    def list_by_conversation(self, conversation_id: str) -> List[ConversationMessageDB]:
        """Return ORM messages for ``conversation_id`` ordered chronologically."""

        return (
            self._db.query(ConversationMessageDB)
            .join(
                Conversation, Conversation.id == ConversationMessageDB.conversation_id
            )
            .filter(Conversation.conversation_id == conversation_id)
            # ``created_at`` is more explicit for chronological ordering than the
            # auto-incremented primary key.
            .order_by(ConversationMessageDB.created_at)
            .all()
        )

    def list_models(self, conversation_id: str) -> List[ConversationMessage]:
        """Return user/assistant messages as pydantic models.

        The underlying ORM model includes all internal agent messages.  For
        conversational context we only expose user and assistant messages,
        converting each row to the public :class:`ConversationMessage` model
        with an explicit timestamp.
        """

        return [
            ConversationMessage(
                user_id=m.user_id,
                conversation_id=conversation_id,
                role=m.role,
                content=m.content,
                timestamp=m.created_at,
            )
            for m in self.list_by_conversation(conversation_id)
            if m.role in {"user", "assistant"}
        ]


__all__ = ["ConversationMessageRepository"]
