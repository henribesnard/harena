"""Simple orchestrator that chains classification, extraction, querying and response."""

from __future__ import annotations

import json

import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session
from agent_types import ChatMessage, TaskResult

from sqlalchemy.orm import Session

from agent_types import ChatMessage, TaskResult
from conversation_service.agents.entity_extractor_agent import EntityExtractorAgent
from conversation_service.agents.intent_classifier_agent import IntentClassifierAgent
from conversation_service.agents.query_generator_agent import QueryGeneratorAgent
from conversation_service.agents.response_generator_agent import ResponseGeneratorAgent
from conversation_service.core.metrics_collector import metrics_collector
from conversation_service.message_repository import ConversationMessageRepository
from conversation_service.repository import ConversationRepository

logger = logging.getLogger(__name__)


class TeamOrchestrator:
    """Run a pipeline of assistant agents sequentially."""

    def __init__(
        self,
        classifier: Optional[Any] = None,
        extractor: Optional[Any] = None,
        query_agent: Optional[Any] = None,
        responder: Optional[Any] = None,
        classifier: Optional[IntentClassifierAgent] = None,
        extractor: Optional[EntityExtractorAgent] = None,
        query_agent: Optional[QueryGeneratorAgent] = None,
        responder: Optional[ResponseGeneratorAgent] = None,
    ) -> None:
        """Initialise the orchestrator with optional agent instances."""

        self._classifier = classifier
        self._extractor = extractor
        self._query_agent = query_agent
        self._responder = responder
        self.context: Dict[str, Any] = {}
        self._metrics = metrics_collector
        self._total_calls = 0
        self._error_calls = 0
        self._total_calls = 0
        self._error_calls = 0
        self._metrics = metrics_collector

    async def run(self, task: str) -> TaskResult:
        """Execute the pipeline and return the resulting messages."""
        messages: List[ChatMessage] = [ChatMessage(content=task, source="user")]
        self.context = {}
        for agent in [
            self._classifier,
            self._extractor,
            self._query_agent,
            self._responder,
        ]:
            if agent is None:
                continue
            response = await agent.on_messages(messages, None)
            msg = response.chat_message
            messages.append(msg)
            name = getattr(agent, "name", agent.__class__.__name__)
            self.context[name] = msg.content
        return TaskResult(messages=messages)

    async def query_agents(
        self, conversation_id: str, message: str, user_id: int, db: Session
    ) -> str:
        """Run classification → extraction → query → response for a message."""
        """Run the agent pipeline for a user message and return the reply."""

        start = time.time()
        history_models = self.get_history(conversation_id, db) or []
        ctx: Dict[str, Any] = {
            "user_id": user_id,
            "history": [m.model_dump() for m in history_models],
        }
        repo = ConversationMessageRepository(db)
        repo.add(
            conversation_id=conversation_id,
            user_id=user_id,
            role="user",
            content=message,
        )
        self._total_calls += 1
        try:
            # 1. Intent classification
            ctx = await self._call_agent(
                self._classifier,
                {"user_message": message},
                ctx,
                repo,
                conversation_id,
                user_id,
            )

            # 2. Entity extraction
            ctx = await self._call_agent(
                self._extractor,
                {"user_message": message, "intent": ctx.get("intent")},
                ctx,
                repo,
                conversation_id,
                user_id,
            )

            # 3. Query generation
            ctx = await self._call_agent(
                self._query_agent,
                {
                    "intent": ctx.get("intent"),
                    "entities": ctx.get("entities"),
                },
                ctx,
                repo,
                conversation_id,
                user_id,
            )

            # 4. Response generation
            ctx = await self._call_agent(
                self._responder,
                {"search_response": ctx.get("search_response")},
                ctx,
                repo,
                conversation_id,
                user_id,
            )

            reply = ctx.get("response", "")
        except Exception:  # pragma: no cover - defensive
            reply = context.get("response", "")
            success = True
        except Exception:
            success = False
            self._error_calls += 1
            logger.exception("Agent processing failed")
            reply = (
                "Désolé, une erreur est survenue lors du traitement de votre demande."
            )
        repo.add(
            conversation_id=conversation_id,
            user_id=user_id,
            role="assistant",
            content=reply,
        )
        duration = (time.time() - start) * 1000
        self._metrics.record_orchestrator_call(
            operation="query_agents", success=success, processing_time_ms=duration
        )
        return reply

    async def _call_agent(
        self,
        agent: Optional[Any],
        payload: Dict[str, Any],
    def start_conversation(self, user_id: int, db: Session) -> str:
        """Create a new conversation for ``user_id`` and return its identifier."""

        conversation_id = str(uuid.uuid4())
        ConversationRepository(db).create(user_id, conversation_id)
        return conversation_id

    def get_history(
        self, conversation_id: str, db: Session
    ) -> Optional[List["ConversationMessage"]]:
        """Return the persisted history for ``conversation_id`` if it exists."""

        repo = ConversationRepository(db)
        if repo.get_by_conversation_id(conversation_id) is None:
            return None
        return ConversationMessageRepository(db).list_models(conversation_id)

    async def _call_agent(
        self,
        agent: Optional[object],
        context: Dict[str, Any],
        repo: ConversationMessageRepository,
        conversation_id: str,
        user_id: int,
    ) -> Dict[str, Any]:
        """Invoke ``agent`` with ``payload`` while updating/persisting context."""
        """Execute ``agent`` with ``context`` and persist its output."""

        if agent is None:
            return context

        # Propagate current context to the agent
        payload["context"] = context
        start = time.time()
        response = await agent.process(payload)  # type: ignore[call-arg]
        duration_ms = int((time.time() - start) * 1000)

        result: Dict[str, Any]
        if hasattr(response, "result"):
            result = response.result  # type: ignore[attr-defined]
        else:
            result = response or {}

        # Update shared context and persist agent output
        context.update(result)
        name = getattr(agent, "name", agent.__class__.__name__)
        repo.add(
            conversation_id=conversation_id,
            user_id=user_id,
            role=name,
            content=json.dumps(result, ensure_ascii=False),
        )

        await self._metrics.record_agent_call(
            agent_name=name, success=True, processing_time_ms=duration_ms
        )

        return context

    def start_conversation(self, user_id: int, db: Session) -> str:
        """Create and persist a new conversation session."""

        conv_id = uuid.uuid4().hex
        repo = ConversationRepository(db)
        repo.create(user_id=user_id, conversation_id=conv_id)
        return conv_id

    def get_history(self, conversation_id: str, db: Session):
        """Return persisted user/assistant message history."""

        repo = ConversationMessageRepository(db)
        return repo.list_models(conversation_id)
        start = time.time()
        agent_name = getattr(getattr(agent, "config", None), "name", None) or getattr(
            agent, "name", agent.__class__.__name__
        )
        success = False
        try:
            response = await agent.process(context)
            if not getattr(response, "success", False):
                raise RuntimeError(getattr(response, "error_message", "agent error"))
            result = getattr(response, "result", {}) or {}
            context.update(result)
            content = (
                result.get("response")
                or result.get("intent")
                or str(result.get("entities", result))
            )
            repo.add(
                conversation_id=conversation_id,
                user_id=user_id,
                role=agent_name,
                content=str(content),
            )
            success = True
            return context
        finally:
            duration = (time.time() - start) * 1000
            self._metrics.record_orchestrator_call(
                operation=agent_name, success=success, processing_time_ms=duration
            )

    def get_error_metrics(self) -> Dict[str, float]:
        """Return counters summarising orchestrator errors."""

        return {
            "total_calls": float(self._total_calls),
            "error_calls": float(self._error_calls),
            "error_rate": (
                self._error_calls / self._total_calls if self._total_calls else 0.0
            ),
        }

