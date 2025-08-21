"""Agent team orchestration using AutoGen 0.4 agents."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from autogen_agentchat.agents import AssistantAgent, BaseChatAgent
from autogen_agentchat.base import Response, TaskResult, Team
from autogen_agentchat.base._task import AgentEvent, ChatMessage
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken


class _EchoAgent(AssistantAgent):
    """Fallback assistant that simply echoes incoming text."""

    produced_message_types = [TextMessage]

    def __init__(self, name: str) -> None:
        BaseChatAgent.__init__(self, name=name, description="echo agent")

    async def on_messages(
        self, messages: Sequence[ChatMessage], cancellation_token: CancellationToken
    ) -> Response:
        content = messages[0].content if messages else ""
        return Response(chat_message=TextMessage(content=content, source=self.name))

    async def on_reset(self, cancellation_token: CancellationToken) -> None:  # pragma: no cover - stateless
        return None


class TeamOrchestrator(Team):
    """Coordinate a team of assistant agents with shared context."""

    def __init__(
        self,
        classifier: AssistantAgent | None = None,
        extractor: AssistantAgent | None = None,
        query_agent: AssistantAgent | None = None,
        responder: AssistantAgent | None = None,
    ) -> None:
        self.classifier = classifier or _EchoAgent("classification")
        self.extractor = extractor or _EchoAgent("extraction")
        self.query_agent = query_agent or _EchoAgent("query")
        self.responder = responder or _EchoAgent("response")
        self.context: dict[str, Any] = {}

    async def run(
        self,
        *,
        task: str | ChatMessage | Sequence[ChatMessage] | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> TaskResult:
        if cancellation_token is None:
            cancellation_token = CancellationToken()
        if isinstance(task, str):
            message = TextMessage(content=task, source="user")
        elif isinstance(task, TextMessage):
            message = task
        else:
            raise ValueError("Task must be a string or TextMessage")

        outputs: list[AgentEvent | ChatMessage] = [message]

        # Intent classification
        resp = await self.classifier.on_messages([message], cancellation_token)
        self.context["classification"] = resp.chat_message.content
        outputs.append(resp.chat_message)

        # Entity extraction
        msg = TextMessage(content=resp.chat_message.content, source=self.classifier.name)
        resp = await self.extractor.on_messages([msg], cancellation_token)
        self.context["extraction"] = resp.chat_message.content
        outputs.append(resp.chat_message)

        # Query generation
        msg = TextMessage(content=resp.chat_message.content, source=self.extractor.name)
        resp = await self.query_agent.on_messages([msg], cancellation_token)
        self.context["query"] = resp.chat_message.content
        outputs.append(resp.chat_message)

        # Response generation
        msg = TextMessage(content=resp.chat_message.content, source=self.query_agent.name)
        resp = await self.responder.on_messages([msg], cancellation_token)
        self.context["response"] = resp.chat_message.content
        outputs.append(resp.chat_message)

        return TaskResult(messages=outputs)

    def run_stream(
        self,
        *,
        task: str | ChatMessage | Sequence[ChatMessage] | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> Any:
        async def _gen() -> Any:
            result = await self.run(task=task, cancellation_token=cancellation_token)
            for msg in result.messages:
                yield msg
            yield result

        return _gen()

    async def reset(self) -> None:
        self.context.clear()
        token = CancellationToken()
        await self.classifier.on_reset(token)
        await self.extractor.on_reset(token)
        await self.query_agent.on_reset(token)
        await self.responder.on_reset(token)

    async def save_state(self) -> Mapping[str, Any]:
        return {"context": dict(self.context)}

    async def load_state(self, state: Mapping[str, Any]) -> None:
        self.context = dict(state.get("context", {}))

