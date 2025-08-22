"""Simple orchestrator chaining intent classification, entity extraction,
query generation and response production."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

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
from conversation_service.message_repository import ConversationMessageRepository
from conversation_service.models.conversation_models import (
    ConversationMessage,
    MessageCreate,
)
from conversation_service.core.conversation_service import ConversationService
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
        if not message.strip():
            raise ValueError("message must not be empty")

        history_models = self.get_history(conversation_id, db) or []
        ctx: Dict[str, Any] = {
            "user_id": user_id,
            "history": [m.model_dump() for m in history_models],
        }

        # Validate user message before any processing to avoid partial writes
        MessageCreate(role="user", content=message)

        service = ConversationService(db)
        conv = ConversationRepository(db).get_by_conversation_id(conversation_id)
        if conv is None:
            raise RuntimeError("Conversation database id not initialised")
        agent_messages: List[Tuple[str, str]] = []

        self._total_calls += 1
        try:
            tasks = []
            if self._classifier is not None:
                tasks.append(
                    self._call_agent_safe(
                        self._classifier,
                        {"user_message": message},
                        ctx,
                        agent_messages,
                    )
                )
            if self._extractor is not None:
                tasks.append(
                    self._call_agent_safe(
                        self._extractor,
                        {"user_message": message},
                        ctx,
                        agent_messages,
                    )
                )
            if tasks:
                await asyncio.gather(*tasks)

            ctx = await self._call_agent_safe(
                self._query_agent,
                {
                    "intent": ctx.get("intent"),
                    "entities": ctx.get("entities"),
                },
                ctx,
                agent_messages,
            )
            ctx = await self._call_agent_safe(
                self._responder,
                {"search_response": ctx.get("search_response")},
                ctx,
                agent_messages,
            )
            reply = ctx.get("response", "")
        except Exception:  # pragma: no cover - defensive
            self._error_calls += 1
            logger.exception("Agent processing failed")
            reply = (
                "Désolé, une erreur est survenue lors du traitement de votre demande."
            )

        service.save_conversation_turn_atomic(
            conversation=conv,
            user_message=message,
            agent_messages=agent_messages,
            assistant_reply=reply,
        )

        self.context = dict(ctx)
        return reply

    async def _call_agent_safe(
        self,
        agent: Optional[Any],
        payload: Dict[str, Any],
        context: Dict[str, Any],
        messages: List[Tuple[str, str]],
    ) -> Dict[str, Any]:
        if agent is None:
            return context

        payload["context"] = context
        name = getattr(agent, "name", agent.__class__.__name__)
        try:
            response = await agent.process(payload)  # type: ignore[call-arg]
            result: Dict[str, Any]
            if hasattr(response, "result"):
                result = response.result  # type: ignore[attr-defined]
            else:
                result = response or {}
            context.update(result)
            if result:
                messages.append(
                    (name, json.dumps(result, ensure_ascii=False))
                )
        except Exception:
            result = {}
            logger.exception("Agent %s failed", name)

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
        try:
            conv = ConversationRepository(db).create(user_id, conv_id)
            db.commit()
        except Exception:
            db.rollback()
            raise

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
        try:
            return ConversationMessageRepository(db).list_models(conversation_id)
        except sqlalchemy.exc.ProgrammingError:
            return []

    def get_error_metrics(self) -> Dict[str, float]:
        return {
            "total_calls": float(self._total_calls),
            "error_calls": float(self._error_calls),
            "error_rate": (
                self._error_calls / self._total_calls if self._total_calls else 0.0
            ),
        }


__all__ = ["TeamOrchestrator"]
