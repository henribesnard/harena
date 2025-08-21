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

