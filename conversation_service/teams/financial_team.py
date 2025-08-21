"""Utilities to assemble the financial conversation agents into a team.

The :class:`FinancialTeam` wires together the intent classifier, entity
extractor, query generator and response generator agents.  A shared
:class:`~conversation_service.agents.context_manager.ContextManager` is
used so intermediate results can be exchanged between agents during the
processing pipeline.
"""

from __future__ import annotations

from typing import Optional

from conversation_service.agents.agent_team import AgentTeam
from conversation_service.agents.context_manager import ContextManager
from conversation_service.agents.intent_classifier import IntentClassifierAgent
from conversation_service.agents.entity_extractor import EntityExtractorAgent
from conversation_service.agents.query_generator import QueryGeneratorAgent
from conversation_service.agents.response_generator import ResponseGeneratorAgent
from conversation_service.models.core_models import AgentResponse
from openai_client import OpenAIClient


class FinancialTeam:
    """Bundle the four core financial agents with a shared context."""

    def __init__(self, openai_client: Optional[OpenAIClient] = None) -> None:
        self.openai_client = openai_client or OpenAIClient(api_key="dummy")
        self.context = ContextManager()
        self.agent_team = AgentTeam(
            IntentClassifierAgent(self.openai_client),
            EntityExtractorAgent(self.openai_client),
            QueryGeneratorAgent(self.openai_client),
            ResponseGeneratorAgent(self.openai_client),
            context_manager=self.context,
        )

    async def process(self, message: str) -> AgentResponse:
        """Run the full agent pipeline for ``message``."""

        return await self.agent_team.run(message)
