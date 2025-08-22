"""Utilities for persisting conversation turns."""

from __future__ import annotations

from typing import Iterable, List, Tuple

from sqlalchemy.orm import Session

from conversation_service.message_repository import ConversationMessageRepository

__all__ = ["save_conversation_turn"]


def save_conversation_turn(
    db: Session,
    *,
    conversation_db_id: int,
    user_id: int,
    user_message: str,
    agent_messages: Iterable[Tuple[str, str]],
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
        Iterable of ``(role, content)`` pairs representing intermediate agent
        outputs produced while handling the user's message.
    assistant_reply:
        Final reply returned to the user.
    """

    repo = ConversationMessageRepository(db)
    messages: List[Tuple[str, str]] = [("user", user_message)]
    messages.extend(agent_messages)
    messages.append(("assistant", assistant_reply))
    repo.add_batch(
        conversation_db_id=conversation_db_id,
        user_id=user_id,
        messages=messages,
    )
    db.commit()
