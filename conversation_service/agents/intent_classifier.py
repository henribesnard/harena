"""Intent classification agent."""

from typing import Any, Dict, Optional, Tuple

import asyncio
import time

from .base_agent import BaseFinancialAgent
from ..models.agent_models import AgentConfig
from ..prompts import intent_prompts
from ..models.core_models import IntentResult


DEFAULT_TTL = 300


class IntentClassifierAgent(BaseFinancialAgent):
    """Classify user intent using language model prompts."""

    def __init__(self, openai_client):
        config = AgentConfig(
            name="intent_classifier",
            system_message=intent_prompts.get_prompt(),
            model_name="gpt-4o-mini",
        )
        super().__init__(config=config, openai_client=openai_client)
        self.examples = intent_prompts.get_examples()

    async def _process_implementation(
        self, input_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Classify user intent using few-shot prompting.

        The method builds a chat completion request using the system prompt and
        few-shot examples defined in :mod:`conversation_service.prompts`.  It
        retries the OpenAI call on transient network or timeout errors.
        """

        user_message = input_data.get("user_message", "")
        context = input_data.get("context", {})

        prompt = user_message

        # Retry locally in addition to the retry logic of ``OpenAIClient``
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
                intent = response["content"].strip()
                return {"input": input_data, "context": context, "intent": intent}
            except Exception as exc:  # pragma: no cover - network/timeout
                last_error = exc
                if attempt >= 2:
                    raise
                await asyncio.sleep(2 ** attempt)

        # Should not reach here; propagate last error for caller
        if last_error:
            raise last_error
        return None


class IntentClassificationCache:
    """Simple in-memory cache for intent classification results."""

    def __init__(self) -> None:
        self._store: Dict[str, Tuple[IntentResult, float]] = {}
        self.hits: int = 0

    @staticmethod
    def _make_key(user_id: str, message: str) -> str:
        return f"{user_id}:{message}"

    def get(self, user_id: str, message: str, ttl: int = DEFAULT_TTL) -> Optional[IntentResult]:
        key = self._make_key(user_id, message)
        entry = self._store.get(key)
        if entry is None:
            return None
        result, timestamp = entry
        if time.time() - timestamp > ttl:
            self._store.pop(key, None)
            return None
        self.hits += 1
        return result

    def set(
        self,
        user_id: str,
        message: str,
        result: IntentResult,
        ttl: int = DEFAULT_TTL,
    ) -> None:
        key = self._make_key(user_id, message)
        self._store[key] = (result, time.time())

    def clear(self) -> None:
        self._store.clear()
        self.hits = 0


__all__ = ["IntentClassifierAgent", "IntentClassificationCache"]
