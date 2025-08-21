"""Simple orchestrator chaining intent classification, entity extraction,
query generation and response production."""
from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from conversation_service.agents.entity_extractor_agent import (
    EntityExtractorAgent,
)
from conversation_service.agents.intent_classifier_agent import (
    IntentClassifierAgent,
)
from conversation_service.agents.query_generator_agent import QueryGeneratorAgent
from conversation_service.agents.response_generator_agent import (
    ResponseGeneratorAgent,
)
from conversation_service.core.metrics_collector import metrics_collector
from conversation_service.message_repository import (
    ConversationMessageRepository,
    ConversationMessage,
)
from conversation_service.repository import ConversationRepository


logger = logging.getLogger(__name__)


class TeamOrchestrator:
    """Run a pipeline of assistant agents.

    Each stage updates a shared ``ctx`` dictionary whose values are
    consumed by the subsequent agents.
    """

    def __init__(
        self,
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

    async def query_agents(
        self, conversation_id: str, message: str, user_id: int, db: Session
    ) -> str:
        """Run the agent pipeline for a user message and return the reply."""
        start = time.time()
        history_models = self.get_history(conversation_id, db) or []
        ctx: Dict[str, Any] = {
            "user_id": user_id,
            "history": [asdict(m) for m in history_models],
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
            pipeline = [
                (
                    self._classifier,
                    lambda c: {"user_message": message},
                ),
                (
                    self._extractor,
                    lambda c: {
                        "user_message": message,
                        "intent": c.get("intent"),
                    },
                ),
                (
                    self._query_agent,
                    lambda c: {
                        "intent": c.get("intent"),
                        "entities": c.get("entities"),
                    },
                ),
                (
                    self._responder,
                    lambda c: {"search_response": c.get("search_response")},
                ),
            ]
            for agent, builder in pipeline:
                ctx = await self._call_agent(
                    agent,
                    builder(ctx),
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
        self.context = dict(ctx)
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
        """Execute ``agent`` with ``payload`` and persist its output."""
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

    def start_conversation(self, user_id: int, db: Session) -> str:
        """Create a new conversation and preload its history."""
        conv_id = uuid.uuid4().hex
        ConversationRepository(db).create(user_id, conv_id)
        history = ConversationMessageRepository(db).list_models(conv_id)
        self.context = {
            "user_id": user_id,
            "history": [asdict(m) for m in history],
        }
        return conv_id

    def get_history(
        self, conversation_id: str, db: Session
    ) -> Optional[List[ConversationMessage]]:
        """Return the persisted history for ``conversation_id`` if it exists."""
        repo = ConversationRepository(db)
        if repo.get_by_conversation_id(conversation_id) is None:
            return None
        return ConversationMessageRepository(db).list_models(conversation_id)

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

