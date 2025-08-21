from __future__ import annotations

"""Simple orchestrator that persists conversation messages to the database."""

import asyncio
import logging
import time
from typing import Dict, List, Optional
from uuid import uuid4

import httpx
from sqlalchemy.orm import Session

from conversation_service.core.metrics_collector import (
    MetricsCollector,
    metrics_collector,
)
from conversation_service.message_repository import ConversationMessageRepository
from conversation_service.repository import ConversationRepository
from models.conversation_models import ConversationMessage as ConversationMessageModel


logger = logging.getLogger(__name__)

"""Agent team orchestration using lightweight agent protocol."""

from __future__ import annotations

from typing import Any, Mapping, Sequence, AsyncGenerator

from agent_types import AssistantAgent, ChatMessage, Response, TaskResult


class _EchoAgent:
    """Fallback assistant that simply echoes incoming text."""

    def __init__(self, name: str) -> None:
        self.name = name

    async def on_messages(
        self, messages: Sequence[ChatMessage], cancellation_token: Any
    ) -> Response:
        content = messages[0].content if messages else ""
        return Response(chat_message=ChatMessage(content=content, source=self.name))

    async def on_reset(self, cancellation_token: Any) -> None:  # pragma: no cover - stateless
        return None


class TeamOrchestrator:
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
        cancellation_token: Any | None = None,
    ) -> TaskResult:
        if isinstance(task, str):
            message = ChatMessage(content=task, source="user")
        elif isinstance(task, ChatMessage):
            message = task
        else:
            raise ValueError("Task must be a string or ChatMessage")

        outputs: list[ChatMessage] = [message]

        # Intent classification
        resp = await self.classifier.on_messages([message], cancellation_token)
        self.context["classification"] = resp.chat_message.content
        outputs.append(resp.chat_message)

        # Entity extraction
        msg = ChatMessage(content=resp.chat_message.content, source=self.classifier.name)
        resp = await self.extractor.on_messages([msg], cancellation_token)
        self.context["extraction"] = resp.chat_message.content
        outputs.append(resp.chat_message)

        # Query generation
        msg = ChatMessage(content=resp.chat_message.content, source=self.extractor.name)
        resp = await self.query_agent.on_messages([msg], cancellation_token)
        self.context["query"] = resp.chat_message.content
        outputs.append(resp.chat_message)

        # Response generation
        msg = ChatMessage(content=resp.chat_message.content, source=self.query_agent.name)
        resp = await self.responder.on_messages([msg], cancellation_token)
        self.context["response"] = resp.chat_message.content
        outputs.append(resp.chat_message)

class TeamOrchestrator:
    """Coordinate agent interactions and store message history."""

    def __init__(self, metrics: Optional[MetricsCollector] = None) -> None:
        self._metrics = metrics or metrics_collector
        self._total_calls = 0
        self._error_calls = 0

    def start_conversation(self, user_id: int, db: Session) -> str:
        start = time.time()
        conv_id = str(uuid4())
        ConversationRepository(db).create(user_id=user_id, conversation_id=conv_id)
        duration = (time.time() - start) * 1000
        self._metrics.record_orchestrator_call(
            operation="start_conversation", success=True, processing_time_ms=duration
        )
        return conv_id

    def get_history(
        self, conversation_id: str, db: Session
    ) -> Optional[List[ConversationMessageModel]]:
        repo = ConversationMessageRepository(db)
        msgs = repo.list_by_conversation(conversation_id)
        if not msgs:
            return None
        return [
            ConversationMessageModel(role=m.role, content=m.content) for m in msgs
        ]

    async def _call_agent(self, message: str) -> str:
        async with httpx.AsyncClient() as _client:
            await asyncio.sleep(0)
        return f"Echo: {message}"

    async def query_agents(
        self, conversation_id: str, message: str, user_id: int, db: Session
    ) -> str:
        start = time.time()
        repo = ConversationMessageRepository(db)
        repo.add(
            conversation_id=conversation_id,
            user_id=user_id,
            role="user",
            content=message,
        )
        self._total_calls += 1
        try:
            reply = await self._call_agent(message)
        except Exception:
            self._error_calls += 1
            logger.exception("Agent processing failed")
            reply = "Désolé, une erreur est survenue lors du traitement de votre demande."
        repo.add(
            conversation_id=conversation_id,
            user_id=user_id,
            role="assistant",
            content=reply,
        )
        duration = (time.time() - start) * 1000
        self._metrics.record_orchestrator_call(
            operation="query_agents", success=True, processing_time_ms=duration
        )
        return reply

    def get_error_metrics(self) -> Dict[str, float]:
        return {
            "total_calls": float(self._total_calls),
            "error_calls": float(self._error_calls),
            "error_rate": self._error_calls / self._total_calls
            if self._total_calls
            else 0.0,
        }

    def run_stream(
        self,
        *,
        task: str | ChatMessage | Sequence[ChatMessage] | None = None,
        cancellation_token: Any | None = None,
    ) -> AsyncGenerator[ChatMessage | TaskResult, None]:
        async def _gen() -> AsyncGenerator[ChatMessage | TaskResult, None]:
            result = await self.run(task=task, cancellation_token=cancellation_token)
            for msg in result.messages:
                yield msg
            yield result

        return _gen()

    async def reset(self) -> None:
        self.context.clear()
        token = None
        await self.classifier.on_reset(token)
        await self.extractor.on_reset(token)
        await self.query_agent.on_reset(token)
        await self.responder.on_reset(token)

    async def save_state(self) -> Mapping[str, Any]:
        return {"context": dict(self.context)}

    async def load_state(self, state: Mapping[str, Any]) -> None:
        self.context = dict(state.get("context", {}))
