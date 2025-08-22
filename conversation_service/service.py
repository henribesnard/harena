from __future__ import annotations

from typing import Sequence

from conversation_service.message_repository import ConversationMessageRepository
from conversation_service.models.conversation_models import MessageCreate


class ConversationService:
    """High level operations for conversations."""

    def __init__(self, repo: ConversationMessageRepository) -> None:
        self._repo = repo

    def save_conversation_turn(
        self,
        *,
        conversation_db_id: int,
        user_id: int,
        messages: Sequence[MessageCreate],
    ) -> None:
        """Persist a full conversation turn atomically.

        Parameters
        ----------
        conversation_db_id:
            Database identifier of the conversation.
        user_id:
            Identifier of the user owning the conversation.
        messages:
            Sequence of messages belonging to the turn, typically a user
            message followed by the assistant response.
        """

        self._repo.add_batch(
            conversation_db_id=conversation_db_id,
            user_id=user_id,
            messages=messages,
        )


__all__ = ["ConversationService"]
