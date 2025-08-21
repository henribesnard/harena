import asyncio

from agent_types import ChatMessage, Response
from agent_runtime import create_runtime


class DummyAgent:
    """Assistant that returns a preconfigured reply."""

    def __init__(self, name: str, reply: str) -> None:
        self.name = name
        self._reply = reply

    async def on_messages(self, messages, cancellation_token):
        return Response(chat_message=ChatMessage(content=self._reply, source=self.name))

    async def on_reset(self, cancellation_token) -> None:  # pragma: no cover - stateless
        return None


def test_full_workflow_context_sharing() -> None:
    runtime = create_runtime(
        classifier=DummyAgent("classification", "intent"),
        extractor=DummyAgent("extraction", "entities"),
        query_agent=DummyAgent("query", "sql"),
        responder=DummyAgent("response", "done"),
    )

    result = asyncio.run(runtime.run("hello"))

    assert result == "done"
    assert runtime.get_context() == {
        "classification": "intent",
        "extraction": "entities",
        "query": "sql",
        "response": "done",
    }
