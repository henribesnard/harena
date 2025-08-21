import pytest

from conversation_service.agents.agent_team import AgentTeam
from conversation_service.agents.context_manager import ContextManager
from conversation_service.models.agent_models import AgentResponse


class DummyAgent:
    """Simple agent returning a preconfigured result."""

    def __init__(self, name: str, result: dict):
        self.name = name
        self.result = result

    async def process(self, input_data):  # type: ignore[override]
        return AgentResponse(
            agent_name=self.name,
            success=True,
            result=self.result,
            processing_time_ms=0,
        )


@pytest.mark.asyncio
async def test_agent_team_context_flow() -> None:
    ctx = ContextManager()
    team = AgentTeam(
        DummyAgent("intent", {"intent": "BALANCE_INQUIRY"}),
        DummyAgent("entity", {"entities": []}),
        DummyAgent("query", {"q": 1}),
        DummyAgent("response", {"text": "ok"}),
        context_manager=ctx,
    )

    final_response = await team.run("hello")

    assert final_response.result == {"text": "ok"}
    context = ctx.get_context()
    assert context["intent"] == {"intent": "BALANCE_INQUIRY"}
    assert context["entities"] == {"entities": []}
    assert context["query"] == {"q": 1}
    assert context["response"] == {"text": "ok"}
