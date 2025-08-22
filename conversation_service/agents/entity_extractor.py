"""Entity extraction agent."""

from typing import Any, Dict, List, Optional, Tuple

import asyncio
import json
import time

from .base_agent import BaseFinancialAgent
from ..models.agent_models import AgentConfig
from ..prompts import entity_prompts
from ..models.core_models import FinancialEntity


DEFAULT_TTL = 180


class EntityExtractorAgent(BaseFinancialAgent):
    """Extract structured entities from user queries."""

    def __init__(self, openai_client):
        config = AgentConfig(
            name="entity_extractor",
            system_message=entity_prompts.get_prompt(),
            model_name="gpt-4o-mini",
        )
        super().__init__(config=config, openai_client=openai_client)
        self.examples = entity_prompts.get_examples()

    async def _process_implementation(
        self, input_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Extract entities from the user message.

        Uses few-shot examples to guide the extraction and retries the OpenAI
        call when transient errors occur.
        """

        user_message = input_data.get("user_message", "")
        context = input_data.get("context", {})

        prompt_parts = [f"Message utilisateur : {user_message}"]
        if context:
            prompt_parts.append(f"Contexte : {json.dumps(context)}")
        prompt = "\n".join(prompt_parts)

        last_error: Optional[Exception] = None
        for attempt in range(3):
            try:
                response = await asyncio.wait_for(
                    self._call_openai(
                        prompt,
                        few_shot_examples=self.examples,
                    ),
                    timeout=self.config.timeout_seconds,
                )
                raw = response["content"].strip()
                entities: Dict[str, str] = {}
                for part in raw.split(","):
                    if ":" in part:
                        key, value = part.split(":", 1)
                        entities[key.strip()] = value.strip()
                return {"input": input_data, "context": context, "entities": entities}
            except Exception as exc:  # pragma: no cover - network/timeout
                last_error = exc
                if attempt >= 2:
                    raise
                await asyncio.sleep(2 ** attempt)

        if last_error:
            raise last_error
        return None


class EntityExtractionCache:
    """In-memory cache for extracted entities."""

    def __init__(self) -> None:
        self._store: Dict[str, Tuple[List[FinancialEntity], float]] = {}
        self.hits: int = 0

    @staticmethod
    def _make_key(user_id: str, intent: str, message: str) -> str:
        return f"{user_id}:{intent}:{message}"

    def get(
        self, user_id: str, intent: str, message: str, ttl: int = DEFAULT_TTL
    ) -> Optional[List[FinancialEntity]]:
        key = self._make_key(user_id, intent, message)
        entry = self._store.get(key)
        if entry is None:
            return None
        entities, timestamp = entry
        if time.time() - timestamp > ttl:
            self._store.pop(key, None)
            return None
        self.hits += 1
        return entities

    def set(
        self,
        user_id: str,
        intent: str,
        message: str,
        entities: List[FinancialEntity],
        ttl: int = DEFAULT_TTL,
    ) -> None:
        key = self._make_key(user_id, intent, message)
        self._store[key] = (entities, time.time())

    def clear(self) -> None:
        self._store.clear()
        self.hits = 0


__all__ = ["EntityExtractorAgent", "EntityExtractionCache"]
