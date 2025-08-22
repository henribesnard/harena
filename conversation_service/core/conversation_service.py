"""Utilities for persisting conversation turns."""

from __future__ import annotations

from typing import Iterable, List

from sqlalchemy.orm import Session

from conversation_service.message_repository import ConversationMessageRepository
from conversation_service.models.conversation_models import MessageCreate

__all__ = ["save_conversation_turn"]


def save_conversation_turn(
    db: Session,
    *,
    conversation_db_id: int,
    user_id: int,
    user_message: str,
    agent_messages: Iterable[MessageCreate],
    assistant_reply: str,
) -> None:
    """Persist a full turn of a conversation.

    Parameters
    ----------
    db:
        SQLAlchemy session used for persistence.
    conversation_db_id:
        Database identifier of the conversation.
    user_id:
        Identifier of the user owning the conversation.
    user_message:
        The message sent by the user.
    agent_messages:
        Iterable of agent messages produced while handling the user's message.
    assistant_reply:
        Final reply returned to the user.
    """

    repo = ConversationMessageRepository(db)
    messages: List[MessageCreate] = [
        MessageCreate(role="user", content=user_message)
    ]
    messages.extend(agent_messages)
    messages.append(MessageCreate(role="assistant", content=assistant_reply))
    repo.add_batch(
        conversation_db_id=conversation_db_id,
        user_id=user_id,
        messages=messages,
    )
    db.commit()
