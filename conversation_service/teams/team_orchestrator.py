"""In-memory orchestrator coordinating financial agent teams.

The orchestrator manages conversation sessions, keeping track of message
history per conversation and delegating message processing to a
:class:`FinancialTeam` instance.
"""

from __future__ import annotations

from typing import Dict, List, Optional
from uuid import uuid4

from models.conversation_models import ConversationMessage

from .financial_team import FinancialTeam


class TeamOrchestrator:
    """Coordinate agent teams and manage conversation history."""

    def __init__(self) -> None:
        self._conversations: Dict[str, List[ConversationMessage]] = {}
        self._teams: Dict[str, FinancialTeam] = {}

    def start_conversation(self, user_id: Optional[int] = None) -> str:
        """Create a new conversation and return its identifier."""

        conv_id = str(uuid4())
        self._conversations[conv_id] = []
        self._teams[conv_id] = FinancialTeam()
        return conv_id

    def get_history(self, conversation_id: str) -> Optional[List[ConversationMessage]]:
        """Return history for a conversation or ``None`` if not found."""

        return self._conversations.get(conversation_id)

    async def query_agents(self, conversation_id: str, message: str) -> str:
        """Process a message through the financial agent team."""

        history = self._conversations.setdefault(conversation_id, [])
        history.append(ConversationMessage(role="user", content=message))

        team = self._teams.setdefault(conversation_id, FinancialTeam())
        response = await team.process(message)
        reply = ""
        if response.success and isinstance(response.result, dict):
            reply = response.result.get("response", "")

        history.append(ConversationMessage(role="assistant", content=reply))
        return reply
