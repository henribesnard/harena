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

import time
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
import asyncio
import logging
from typing import Dict, List, Optional
from uuid import uuid4

from sqlalchemy.orm import Session

from models.conversation_models import ConversationMessage
from conversation_service.repository import ConversationRepository
import httpx

from models.conversation_models import ConversationMessage
from conversation_service.core.metrics_collector import (
    MetricsCollector,
    metrics_collector,
)

logger = logging.getLogger(__name__)


        return TaskResult(messages=outputs)

    def __init__(self, metrics: Optional[MetricsCollector] = None) -> None:
        self._conversations: Dict[str, List[ConversationMessage]] = {}
        self._metrics = metrics or metrics_collector

    def start_conversation(self, user_id: int, db: Session) -> str:
        """Create a new conversation, persist it and return its identifier."""
        conv_id = str(uuid4())
        self._conversations[conv_id] = []
        ConversationRepository(db).create(
            user_id=user_id, conversation_id=conv_id
    def start_conversation(self, user_id: Optional[int] = None) -> str:
        """Create a new conversation and return its identifier."""
        start = time.time()
        conv_id = str(uuid4())
        self._conversations[conv_id] = []
        duration = (time.time() - start) * 1000
        self._metrics.record_orchestrator_call(
            operation="start_conversation", success=True, processing_time_ms=duration
        )
        return conv_id

    def get_history(
        self, conversation_id: str
    ) -> Optional[List[ConversationMessage]]:
        """Return history for a conversation or ``None`` if not found."""
        start = time.time()
        history = self._conversations.get(conversation_id)
        duration = (time.time() - start) * 1000
        self._metrics.record_orchestrator_call(
            operation="get_history",
            success=history is not None,
            processing_time_ms=duration,
        )
        return history
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
    def __init__(self) -> None:
        self._conversations: Dict[str, List[ConversationMessage]] = {}
        self._total_calls: int = 0
        self._error_calls: int = 0

    def start_conversation(self, user_id: Optional[int] = None) -> str:
        """Create a new conversation and return its identifier."""
        try:
            conv_id = str(uuid4())
            self._conversations[conv_id] = []
            return conv_id
        except Exception:  # pragma: no cover - unexpected failure
            logger.exception("Failed to start conversation")
            raise RuntimeError("Unable to start conversation at the moment")

    def get_history(self, conversation_id: str) -> Optional[List[ConversationMessage]]:
        """Return history for a conversation or ``None`` if not found."""
        try:
            return self._conversations.get(conversation_id)
        except Exception:  # pragma: no cover - guard against corrupt state
            logger.exception("Failed to retrieve conversation history")
            return None

    async def _call_agent(self, message: str, max_retries: int = 3) -> str:
        """Placeholder agent call with retry logic for LLM/HTTP operations."""
        last_error: Optional[Exception] = None
        for attempt in range(1, max_retries + 1):
            try:
                # In a real implementation, an HTTP/LLM request would be made here.
                async with httpx.AsyncClient() as _client:
                    await asyncio.sleep(0)  # Placeholder for external call
                return f"Echo: {message}"
            except Exception as exc:  # pragma: no cover - network path
                last_error = exc
                logger.warning(
                    "Agent call failed (attempt %s/%s): %s",
                    attempt,
                    max_retries,
                    exc,
                )
                if attempt < max_retries:
                    await asyncio.sleep(2 ** (attempt - 1))
        assert last_error is not None
        raise last_error

    async def save_state(self) -> Mapping[str, Any]:
        return {"context": dict(self.context)}

    async def load_state(self, state: Mapping[str, Any]) -> None:
        self.context = dict(state.get("context", {}))

        Errors are logged and surfaced with user-friendly messages. The current
        behaviour echoes the user's message as a placeholder.
        """
        start = time.time()
        history = self._conversations.setdefault(conversation_id, [])
        history.append(ConversationMessage(role="user", content=message))
        self._total_calls += 1
        try:
            history = self._conversations.setdefault(conversation_id, [])
            history.append(ConversationMessage(role="user", content=message))
        except Exception:
            self._error_calls += 1
            logger.exception("Failed to record user message")
            return "Une erreur est survenue lors de l'enregistrement du message."

        try:
            reply = await self._call_agent(message)
        except Exception:
            self._error_calls += 1
            logger.exception("Agent processing failed")
            reply = "Désolé, une erreur est survenue lors du traitement de votre demande."
        history.append(ConversationMessage(role="assistant", content=reply))

        duration = (time.time() - start) * 1000
        self._metrics.record_orchestrator_call(
            operation="query_agents", success=True, processing_time_ms=duration
        )
        return reply

    def get_error_metrics(self) -> Dict[str, float]:
        """Return basic error rate metrics."""
        return {
            "total_calls": float(self._total_calls),
            "error_calls": float(self._error_calls),
            "error_rate":
                self._error_calls / self._total_calls if self._total_calls else 0.0,
        }
