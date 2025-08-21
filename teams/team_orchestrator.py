"""Simple orchestrator chaining intent classification, entity extraction,
query generation and response production.

The first two stages (intent classification and entity extraction) are
independent and therefore executed concurrently using ``asyncio.gather``.
The results are then fed sequentially into the query and response
generators.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from agent_types import ChatMessage, TaskResult
from conversation_service.agents.entity_extractor_agent import (
    EntityExtractorAgent,
)
from conversation_service.agents.intent_classifier_agent import (
    IntentClassifierAgent,
)
from conversation_service.agents.query_generator_agent import (
    QueryGeneratorAgent,
)
from conversation_service.agents.response_generator_agent import (
    ResponseGeneratorAgent,
)
from conversation_service.core.metrics_collector import metrics_collector
from conversation_service.message_repository import ConversationMessageRepository
from conversation_service.repository import ConversationRepository


logger = logging.getLogger(__name__)


class TeamOrchestrator:
    """Run a pipeline of assistant agents.

    Intent classification and entity extraction are launched concurrently
    to reduce overall latency.  The shared ``ctx`` dictionary is updated by
    each agent with its result.
    """

    def __init__(
        self,
        classifier: Optional[IntentClassifierAgent] = None,
        extractor: Optional[EntityExtractorAgent] = None,
        query_agent: Optional[QueryGeneratorAgent] = None,
        responder: Optional[ResponseGeneratorAgent] = None,
    ) -> None:
        self._classifier = classifier
        self._extractor = extractor
        self._query_agent = query_agent
        self._responder = responder

        self.context: Dict[str, Any] = {}
        self._metrics = metrics_collector
        self._total_calls = 0
        self._error_calls = 0

    async def run(self, task: str) -> TaskResult:
        """Execute the pipeline and return resulting messages.

        This helper is used only in tests and mirrors the behaviour of the
        main ``query_agents`` method but without persistence.
        """

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

    def start_conversation(self, user_id: int, db: Session) -> str:
        """Create a new conversation for ``user_id`` and return its ID."""

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

    async def query_agents(
        self, conversation_id: str, message: str, user_id: int, db: Session
    ) -> str:
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
        success = True
        try:
            # 1 & 2. Classification and extraction in parallel
            tasks = []
            if self._classifier is not None:
                tasks.append(
                    self._call_agent(
                        self._classifier,
                        {"user_message": message},
                        ctx,
                        repo,
                        conversation_id,
                        user_id,
                    )
                )
            if self._extractor is not None:
                tasks.append(
                    self._call_agent(
                        self._extractor,
                        {"user_message": message},
                        ctx,
                        repo,
                        conversation_id,
                        user_id,
                    )
                )
            if tasks:
                await asyncio.gather(*tasks)

            # 3. Query generation
            ctx = await self._call_agent(
                self._query_agent,
                {"intent": ctx.get("intent"), "entities": ctx.get("entities")},
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
        context: Dict[str, Any],
        repo: ConversationMessageRepository,
        conversation_id: str,
        user_id: int,
    ) -> Dict[str, Any]:
        """Invoke ``agent`` with ``payload`` while updating/persisting context."""

        if agent is None:
            return context

        payload["context"] = context
        start = time.time()
        response = await agent.process(payload)  # type: ignore[call-arg]
        duration_ms = int((time.time() - start) * 1000)

        result: Dict[str, Any]
        if hasattr(response, "result"):
            result = response.result  # type: ignore[attr-defined]
        else:
            result = response or {}

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

    def get_error_metrics(self) -> Dict[str, float]:
        """Return counters summarising orchestrator errors."""

        return {
            "total_calls": float(self._total_calls),
            "error_calls": float(self._error_calls),
            "error_rate": (
                self._error_calls / self._total_calls if self._total_calls else 0.0
            ),
        }


__all__ = ["TeamOrchestrator"]

