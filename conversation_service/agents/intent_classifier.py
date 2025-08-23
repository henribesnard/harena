from __future__ import annotations

"""Intent classification agent."""

from typing import Any

from conversation_service.models.agent_models import AgentConfig, IntentResult
from .base import BaseAgent


DEFAULT_CONFIG = AgentConfig(
    name="intent-classifier",
    system_prompt="Classify the user's intent and return JSON.",
    model="gpt-4o-mini",
)


class IntentClassifier(BaseAgent):
    """Agent responsible for intent classification."""

    def __init__(self, client: Any, config: AgentConfig | None = None) -> None:
        super().__init__(client, config or DEFAULT_CONFIG)

    async def classify(self, text: str) -> IntentResult:
        messages = [{"role": "user", "content": text}]
        return await self._run(messages, IntentResult)
