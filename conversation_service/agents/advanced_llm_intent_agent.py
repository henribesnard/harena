"""Advanced OpenAI LLM intent agent with few-shot prompts, caching, and retries."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict, Optional, Tuple

from .llm_intent_agent import LLMIntentAgent
from ..models.agent_models import AgentConfig
from ..utils.cache import generate_cache_key
from ..utils.llm_intent_cache import llm_intent_cache

logger = logging.getLogger(__name__)

# Few-shot examples to prime the LLM
FEW_SHOT_EXAMPLES: Tuple[Tuple[str, str], ...] = (
    (
        "Quel est mon solde actuel ?",
        '{"intent": "BALANCE_INQUIRY", "confidence": 0.9, "entities": []}',
    ),
    (
        "Bonjour",
        '{"intent": "GREETING", "confidence": 0.9, "entities": []}',
    ),
)


class AdvancedLLMIntentAgent(LLMIntentAgent):
    """Extended OpenAI intent agent with prompt engineering, retry, and caching."""

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        cache: Optional[Any] = None,
        max_retries: int = 3,
        backoff_factor: float = 0.5,
        openai_client: Optional[Any] = None,
    ) -> None:
        self.cache = cache or llm_intent_cache
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor

        if config is None:
            config = AgentConfig(
                name="advanced_llm_intent_agent",
                model_client_config={
                    "model": "gpt-4o-mini",
                    "api_key": os.getenv("OPENAI_API_KEY", ""),
                    "base_url": "https://api.openai.com/v1",
                },
                system_message=self._build_system_message(),
                max_consecutive_auto_reply=1,
                description="LLM intent agent with few-shot prompts and caching",
                temperature=0.0,
                max_tokens=200,
                timeout_seconds=8,
            )

        super().__init__(config=config, openai_client=openai_client)

    # ------------------------------------------------------------------
    @staticmethod
    def _build_system_message() -> str:
        base = LLMIntentAgent._build_system_message()
        examples = "\n".join(
            f"Utilisateur: {u}\nAssistant: {a}" for u, a in FEW_SHOT_EXAMPLES
        )
        return base + "\n\nExemples:\n" + examples

    # ------------------------------------------------------------------
    async def detect_intent(self, user_message: str, user_id: int) -> Dict[str, Any]:
        """Detect intent with caching, retry, and fallback."""
        cache_key = generate_cache_key(user_message, user_id)
        try:
            cached = await self.cache.get(cache_key)
        except Exception as err:  # pragma: no cover - cache failure shouldn't crash
            logger.debug("Cache retrieval failed: %s", err)
            cached = None
        if cached is not None:
            return cached

        attempt = 0
        delay = self.backoff_factor
        last_error: Optional[Exception] = None
        while attempt < self.max_retries:
            try:
                result = await super().detect_intent(user_message, user_id)
                try:
                    await self.cache.set(cache_key, result)
                except Exception as err:  # pragma: no cover
                    logger.debug("Cache set failed: %s", err)
                return result
            except Exception as err:
                last_error = err
                attempt += 1
                logger.warning(
                    "Intent detection attempt %d failed: %s", attempt, err
                )
                await asyncio.sleep(delay)
                delay *= 2

        logger.error(
            "All retries failed for AdvancedLLMIntentAgent, falling back: %s",
            last_error,
        )
        fallback_agent = LLMIntentAgent(openai_client=self._openai_client)
        return await fallback_agent.detect_intent(user_message, user_id)

__all__ = ["AdvancedLLMIntentAgent"]
