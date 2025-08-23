from __future__ import annotations

"""Entity extraction agent."""

from typing import Any, List

from pydantic import BaseModel, Field

from conversation_service.models.agent_models import AgentConfig, DynamicFinancialEntity
from .base import BaseAgent


class EntityExtractionResult(BaseModel):
    entities: List[DynamicFinancialEntity] = Field(default_factory=list)


DEFAULT_CONFIG = AgentConfig(
    name="entity-extractor",
    system_prompt="Extract financial entities and return JSON.",
    model="gpt-4o-mini",
)


class EntityExtractor(BaseAgent):
    """Agent extracting entities from user text."""

    def __init__(self, client: Any, config: AgentConfig | None = None) -> None:
        super().__init__(client, config or DEFAULT_CONFIG)

    async def extract(self, text: str) -> List[DynamicFinancialEntity]:
        messages = [{"role": "user", "content": text}]
        result = await self._run(messages, EntityExtractionResult)
        return result.entities
