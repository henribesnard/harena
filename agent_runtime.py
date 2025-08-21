"""Runtime helpers for executing an AutoGen team."""

from __future__ import annotations

from autogen_agentchat.agents import AssistantAgent

from teams.team_orchestrator import TeamOrchestrator


class AgentRuntime:
    """Lightweight wrapper around a :class:`TeamOrchestrator`."""

    def __init__(self, team: TeamOrchestrator) -> None:
        self.team = team

    async def run(self, message: str) -> str:
        """Execute the agent pipeline for a user message."""

        result = await self.team.run(task=message)
        final_message = result.messages[-1]
        return getattr(final_message, "content", "")

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

