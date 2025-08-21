import asyncio

from agent_types import ChatMessage, Response
from agent_runtime import AgentRuntime


class DummyTeam:
    """Minimal team that returns a static response and context."""

    def __init__(self) -> None:
        self.context = {
            "classification": "intent",
            "extraction": "entities",
            "query": "sql",
            "response": "done",
        }
        self._conversation_id = "conv"

    async def run(self, task: str, user_id: int, db) -> Response:
        return Response(
            chat_message=ChatMessage(content="done", source="assistant")
        )


def test_full_workflow_context_sharing() -> None:
    runtime = AgentRuntime(DummyTeam())

    result = asyncio.run(runtime.run("hello", user_id=1, db=None))

    assert result == "done"
    assert runtime.get_context() == {
        "classification": "intent",
        "extraction": "entities",
        "query": "sql",
        "response": "done",
    }
