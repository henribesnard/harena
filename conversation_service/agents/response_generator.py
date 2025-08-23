from __future__ import annotations

"""Final response generation agent."""

import json
from typing import Any, List

from conversation_service.models.agent_models import (
    AgentConfig,
    AgentResponse,
    IntentResult,
    DynamicFinancialEntity,
)
from .base import BaseAgent


DEFAULT_CONFIG = AgentConfig(
    name="response-generator",
    system_prompt="Craft a helpful response based on intent, entities and query.",
    model="gpt-4o-mini",
)


class ResponseGenerator(BaseAgent):
    """Agent composing the final answer for the user."""

    def __init__(self, client: Any, config: AgentConfig | None = None) -> None:
        super().__init__(client, config or DEFAULT_CONFIG)

    async def respond(
        self,
        intent: IntentResult,
        entities: List[DynamicFinancialEntity],
        query: str,
    ) -> AgentResponse:
        payload = {
            "intent": intent.model_dump(),
            "entities": [e.model_dump() for e in entities],
            "query": query,
        }
        messages = [{"role": "user", "content": json.dumps(payload)}]
        return await self._run(messages, AgentResponse)
