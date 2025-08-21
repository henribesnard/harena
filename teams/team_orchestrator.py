"""Simple orchestrator that persists conversation messages to the database."""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional
from uuid import uuid4

from sqlalchemy.orm import Session

from conversation_service.core.metrics_collector import (
    MetricsCollector,
    metrics_collector,
)
from conversation_service.message_repository import ConversationMessageRepository
from conversation_service.repository import ConversationRepository
from models.conversation_models import (
    ConversationMessage as ConversationMessageModel,
)

from agent_types import ChatMessage, TaskResult

logger = logging.getLogger(__name__)


class TeamOrchestrator:
    """Coordinate agent interactions and store message history."""

    def __init__(
        self,
        metrics: Optional[MetricsCollector] = None,
        classifier=None,
        extractor=None,
        query_agent=None,
        responder=None,
    ) -> None:
        self._metrics = metrics or metrics_collector
        self._classifier = classifier
        self._extractor = extractor
        self._query_agent = query_agent
        self._responder = responder
        self.context: Dict[str, Any] = {}
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
        msgs = repo.list_models(conversation_id)
        return msgs or None

    async def _call_agent(
        self,
        agent,
        context: Dict[str, Any],
        repo: ConversationMessageRepository,
        conversation_id: str,
        user_id: int,
    ) -> Dict[str, Any]:
        if not agent:
            return context
        agent_name = getattr(
            getattr(agent, "config", None), "name", agent.__class__.__name__
        )
        input_payload = {
            "user_message": context.get("user_message", ""),
            "context": context,
        }
        repo.add(
            conversation_id=conversation_id,
            user_id=user_id,
            role=f"{agent_name}_input",
            content=json.dumps(input_payload),
        )
        result = await agent.process(input_payload)
        output = result.result if result and getattr(result, "result", None) else {}
        repo.add(
            conversation_id=conversation_id,
            user_id=user_id,
            role=f"{agent_name}_output",
            content=json.dumps(output),
        )
        if isinstance(output, dict):
            context.update(output)
        return context

    async def run(self, task: str) -> TaskResult:
        """Execute assistant agents sequentially and track context."""
        messages = [ChatMessage(content=task, source="user")]
        self.context = {}
        for agent in [
            self._classifier,
            self._extractor,
            self._query_agent,
            self._responder,
        ]:
            if not agent:
                continue
            response = await agent.on_messages(messages, None)
            msg = response.chat_message
            messages.append(msg)
            self.context[getattr(agent, "name", agent.__class__.__name__)] = msg.content
        return TaskResult(messages=messages)

    async def query_agents(
        self, conversation_id: str, message: str, user_id: int, db: Session
    ) -> str:
        start = time.time()
        history_models = self.get_history(conversation_id, db) or []
        context: Dict[str, Any] = {
            "user_message": message,
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
            context = await self._call_agent(
                self._classifier, context, repo, conversation_id, user_id
            )
            context = await self._call_agent(
                self._extractor, context, repo, conversation_id, user_id
            )
            context = await self._call_agent(
                self._query_agent, context, repo, conversation_id, user_id
            )
            context = await self._call_agent(
                self._responder, context, repo, conversation_id, user_id
            )
            reply = context.get("response", "")
        except Exception:
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
            operation="query_agents", success=True, processing_time_ms=duration
        )
        return reply

    def get_error_metrics(self) -> Dict[str, float]:
        return {
            "total_calls": float(self._total_calls),
            "error_calls": float(self._error_calls),
            "error_rate": (
                self._error_calls / self._total_calls if self._total_calls else 0.0
            ),
        }
