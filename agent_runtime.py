"""Runtime helpers for executing a team of assistant agents."""

from __future__ import annotations

from agent_types import AssistantAgent
from sqlalchemy.orm import Session
from conversation_service.teams.team_orchestrator import TeamOrchestrator


class AgentRuntime:
    """Lightweight wrapper around a :class:`TeamOrchestrator`."""

    def __init__(self, team: TeamOrchestrator) -> None:
        self.team = team

    async def run(self, message: str, user_id: int, db: Session) -> str:
        """Execute the agent pipeline for a user message.

        Parameters
        ----------
        message:
            The user's message to process.
        user_id:
            Identifier of the user sending the message.
        db:
            Database session used for persistence.
        """

        if self.team._conversation_id is None:
            # Initialise conversation lazily if not already started.
            self.team.start_conversation(user_id, db)

        response = await self.team.run(task=message, user_id=user_id, db=db)
        return getattr(response.chat_message, "content", "")

    def get_context(self) -> dict[str, str]:
        """Return the shared context accumulated by the team."""

        return dict(self.team.context)


def create_runtime(
    classifier: AssistantAgent | None = None,
    extractor: AssistantAgent | None = None,
    query_agent: AssistantAgent | None = None,
    responder: AssistantAgent | None = None,
) -> AgentRuntime:
    """Factory that builds an :class:`AgentRuntime` with the provided agents."""

    team = TeamOrchestrator(
        classifier=classifier,
        extractor=extractor,
        query_agent=query_agent,
        responder=responder,
    )
    return AgentRuntime(team)
