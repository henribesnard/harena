"""Orchestration utilities for coordinating multiple agents.

The :class:`AgentTeam` sequentially invokes a set of conversational agents
(intent classification, entity extraction, query generation and response
production).  Results from each step are stored in a shared
:class:`~conversation_service.agents.context_manager.ContextManager` so that
subsequent agents can leverage previous outputs.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Protocol

from .context_manager import ContextManager
from ..models.core_models import AgentResponse


class ConversationAgent(Protocol):
    """Minimal protocol implemented by agents used within the team."""

    async def process(self, input_data: Dict[str, Any]) -> AgentResponse:
        ...


class AgentTeam:
    """Pipeline executor for the Harena conversation agents."""

    def __init__(
        self,
        intent_agent: ConversationAgent,
        entity_agent: ConversationAgent,
        query_agent: ConversationAgent,
        response_agent: ConversationAgent,
        context_manager: Optional[ContextManager] = None,
    ) -> None:
        self.intent_agent = intent_agent
        self.entity_agent = entity_agent
        self.query_agent = query_agent
        self.response_agent = response_agent
        self.context = context_manager or ContextManager()

    async def run(self, user_message: str) -> AgentResponse:
        """Execute the agent pipeline for a user message."""

        # Intent classification
        intent_resp = await self.intent_agent.process(
            {"message": user_message, "context": self.context.get_context()}
        )
        if intent_resp.success:
            self.context.update(intent=intent_resp.result)

        # Entity extraction
        entity_resp = await self.entity_agent.process(
            {
                "message": user_message,
                "intent": self.context.get("intent"),
                "context": self.context.get_context(),
            }
        )
        if entity_resp.success:
            self.context.update(entities=entity_resp.result)

        # Query generation
        query_resp = await self.query_agent.process(
            {
                "intent": self.context.get("intent"),
                "entities": self.context.get("entities"),
                "context": self.context.get_context(),
            }
        )
        if query_resp.success:
            self.context.update(query=query_resp.result)

        # Response generation
        response_resp = await self.response_agent.process(
            {
                "query": self.context.get("query"),
                "context": self.context.get_context(),
            }
        )
        if response_resp.success:
            self.context.update(response=response_resp.result)

        return response_resp
