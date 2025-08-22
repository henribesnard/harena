"""Simple orchestrator chaining intent classification, entity extraction,
query generation and response production."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional

import sqlalchemy
from sqlalchemy.orm import Session

from agent_types import ChatMessage, Response
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
from conversation_service.models.conversation_models import ConversationMessage
from conversation_service.repository import ConversationRepository

logger = logging.getLogger(__name__)


class TeamOrchestrator:

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
        self._conversation_id: Optional[str] = None
        self._conversation_db_id: Optional[int] = None
        self._user_id: Optional[int] = None
        self._db: Optional[Session] = None

    async def run(self, task: str, user_id: int, db: Session) -> Response:
        if self._conversation_id is None:
            # Lazily initialise conversation and context
            self.start_conversation(user_id, db)
        else:
            # Ensure persistence attributes are set for subsequent calls
            self._user_id = self._user_id or user_id
            self._db = self._db or db
            if self._conversation_db_id is None:
                conv = ConversationRepository(db).get_by_conversation_id(
                    self._conversation_id
                )
                self._conversation_db_id = conv.id if conv is not None else None

        reply = await self.query_agents(
            self._conversation_id, task, self._user_id, self._db
        )
        return Response(chat_message=ChatMessage(content=reply, source="assistant"))

    async def query_agents(
        self, conversation_id: str, message: str, user_id: int, db: Session
    ) -> str:
        start = time.time()
        history_models = self.get_history(conversation_id, db) or []
        ctx: Dict[str, Any] = {
            "user_id": user_id,
            "history": [m.model_dump() for m in history_models],
        }

        repo = ConversationMessageRepository(db)
        # Store the incoming user message so that subsequent calls have access to
        # the full conversation history.
        if self._conversation_db_id is None:
            raise RuntimeError("Conversation database id not initialised")
        repo.add(
            conversation_id=conversation_id,
            conversation_db_id=self._conversation_db_id,
            user_id=user_id,
            role="user",
            content=message,
        )

        self._total_calls += 1
        success = True
        try:
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

        # Persist the assistant's reply as the last turn in the conversation.
        repo.add(
            conversation_id=conversation_id,
            conversation_db_id=self._conversation_db_id,
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
        if result:
            if self._conversation_db_id is None:
                raise RuntimeError("Conversation database id not initialised")
            repo.add(
                conversation_id=conversation_id,
                conversation_db_id=self._conversation_db_id,
                user_id=user_id,
                role=name,
                content=json.dumps(result, ensure_ascii=False),
            )

        await self._metrics.record_agent_call(
            agent_name=name, success=True, processing_time_ms=duration_ms
        )
        return context

    def start_conversation(self, user_id: int, db: Session) -> str:
        """Start a new conversation and load any existing history.

        If the conversation messages table has not been created yet, an empty
        history is returned instead of raising an error.

        Args:
            user_id: Identifier for the user starting the conversation.
            db: SQLAlchemy session used for persistence.

        Returns:
            The identifier of the newly created conversation.
        """

        conv_id = uuid.uuid4().hex
        conv = ConversationRepository(db).create(user_id, conv_id)
        history = ConversationMessageRepository(db).list_models(conv_id)
        ConversationRepository(db).create(user_id, conv_id)
        try:
            history = ConversationMessageRepository(db).list_models(conv_id)
        except sqlalchemy.exc.ProgrammingError:
            history = []

        self.context = {
            "user_id": user_id,
            "history": [m.model_dump() for m in history],
        }
        self._conversation_id = conv_id
        self._conversation_db_id = conv.id
        self._user_id = user_id
        self._db = db
        return conv_id

    def get_history(
        self, conversation_id: str, db: Session
    ) -> Optional[List[ConversationMessage]]:
        repo = ConversationRepository(db)
        if repo.get_by_conversation_id(conversation_id) is None:
            return None
        return ConversationMessageRepository(db).list_models(conversation_id)

    def get_error_metrics(self) -> Dict[str, float]:
        return {
            "total_calls": float(self._total_calls),
            "error_calls": float(self._error_calls),
            "error_rate": (
                self._error_calls / self._total_calls if self._total_calls else 0.0
            ),
        }


__all__ = ["TeamOrchestrator"]
