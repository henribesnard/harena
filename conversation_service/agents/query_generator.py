from __future__ import annotations

"""Query generation agent."""

import json
from typing import Any, List

from pydantic import BaseModel

from conversation_service.models.agent_models import (
    AgentConfig,
    IntentResult,
    DynamicFinancialEntity,
)
from .base import BaseAgent


class QueryModel(BaseModel):
    query: str


DEFAULT_CONFIG = AgentConfig(
    name="query-generator",
    system_prompt="Generate a search query based on intent and entities.",
    model="gpt-4o-mini",
)


class QueryGenerator(BaseAgent):
    """Agent converting intents and entities into a query string."""

    def __init__(self, client: Any, config: AgentConfig | None = None) -> None:
        super().__init__(client, config or DEFAULT_CONFIG)

    async def generate(
        self, intent: IntentResult, entities: List[DynamicFinancialEntity]
    ) -> str:
        payload = {
            "intent": intent.model_dump(),
            "entities": [e.model_dump() for e in entities],
        }
        messages = [{"role": "user", "content": json.dumps(payload)}]
        result = await self._run(messages, QueryModel)
        return result.query
