"""LLM-only intent detection agent.

This module defines :class:`LLMIntentAgent`, a simple agent that relies on the
DeepSeek LLM to classify user messages into a restricted set of financial
intents and to extract entities.  The tests exercise only a small portion of the
original project, so the implementation here focuses on the functionality
required by the unit tests: constructing a system prompt, sending a request to
``DeepSeekClient`` and returning an :class:`~IntentResult` instance describing
the detected intent.
"""

from __future__ import annotations

import asyncio
import json
import time
import logging
from typing import Any, Dict, List, Optional

from .base_financial_agent import BaseFinancialAgent
from ..core.deepseek_client import DeepSeekClient
from ..models.agent_models import AgentConfig
from ..models.financial_models import (
    DetectionMethod,
    EntityType,
    FinancialEntity,
    IntentCategory,
    IntentResult,
)
from ..utils.intent_cache import IntentResultCache
from ..prompts.intent_prompts import (
    INTENT_SYSTEM_PROMPT,
    INTENT_EXAMPLES_FEW_SHOT,
)

logger = logging.getLogger(__name__)


class LLMOutputParsingError(RuntimeError):
    """Raised when the LLM response cannot be parsed as JSON."""


class LLMIntentAgent(BaseFinancialAgent):
    """Intent detection agent that relies solely on the DeepSeek LLM."""

    def __init__(
        self,
        deepseek_client: DeepSeekClient,
        config: Optional[AgentConfig] = None,
    ) -> None:
        if config is None:
            config = AgentConfig(
                name="llm_intent_agent",
                model_client_config={
                    "model": "deepseek-chat",
                    "api_key": deepseek_client.api_key,
                    "base_url": deepseek_client.base_url,
                },
                system_message=self._build_system_message(),
                max_consecutive_auto_reply=1,
                description="LLM-based intent detection agent",
                temperature=0.0,
                max_tokens=200,
                timeout_seconds=8,
            )

        super().__init__(name=config.name, config=config, deepseek_client=deepseek_client)
        self._intent_cache = IntentResultCache(max_size=100)
        self._max_retries = 3

    # ------------------------------------------------------------------
    # Operation API
    async def _execute_operation(self, input_data: Dict[str, Any], user_id: int) -> Dict[str, Any]:
        """Run the intent detection operation used by the base class."""

        user_message = input_data.get("user_message", "")
        if not user_message:
            raise ValueError("user_message is required for intent detection")
        return await self.detect_intent(user_message, user_id)

    # ------------------------------------------------------------------
    @staticmethod
    def _build_system_message() -> str:
        """Construct the system prompt given to the LLM."""

        return (
            f"{INTENT_SYSTEM_PROMPT}\n\n{INTENT_EXAMPLES_FEW_SHOT}"
            "\n\nRéponds uniquement avec un JSON strict."
        )

    # ------------------------------------------------------------------
    async def detect_intent(
        self, user_message: str, user_id: int
    ) -> Dict[str, Any]:
        """Detect the intent of ``user_message``.

        The DeepSeek client returns a JSON document describing the intent and any
        extracted entities.  This method converts that JSON into an
        :class:`IntentResult` and wraps it in the structure used by the rest of
        the service.
        """

        cached = self._intent_cache.get(user_message)
        if cached is not None:
            data = {
                "intent": cached.intent_type,
                "confidence": cached.confidence,
                "entities": [e.model_dump() for e in cached.entities],
            }
            return {
                "content": json.dumps(data),
                "metadata": {
                    "intent_result": cached,
                    "detection_method": DetectionMethod.LLM_BASED,
                    "confidence": cached.confidence,
                    "intent_type": cached.intent_type,
                    "entities": [e.model_dump() for e in cached.entities],
                    "cache_hit": True,
                },
                "confidence_score": cached.confidence,
            }

        start_time = time.perf_counter()
        messages = [
            {"role": "system", "content": self.config.system_message},
            {"role": "user", "content": user_message},
        ]

        response = None
        for attempt in range(self._max_retries):
            try:
                response = await self.deepseek_client.generate_response(
                    messages=messages,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    user=str(user_id),
                    use_cache=True,
                )
                break
            except Exception as err:
                logger.warning("DeepSeek call failed (attempt %s): %s", attempt + 1, err)
                await asyncio.sleep(2 ** attempt)
        if response is None:
            raise RuntimeError("LLM call failed")
        parsed = getattr(response, "output_parsed", None)
        if parsed is None:
            raw_text = getattr(response, "output_text", None)
            if raw_text is None:
                raw_text = getattr(response, "content", "")
            try:
                parsed = json.loads(raw_text)
            except json.JSONDecodeError as err:
                raise LLMOutputParsingError(
                    f"Invalid JSON in LLM response: {err}"
                ) from err
        data = parsed

        intent_type = data.get("intent", "OUT_OF_SCOPE")
        entities: List[FinancialEntity] = []
        for ent in data.get("entities", []):
            type_key = ent.get("entity_type") or ent.get("type") or ""
            try:
                e_type = EntityType(type_key.upper())
            except Exception:  # pragma: no cover - unknown entity types
                logger.debug("Unknown entity type: %s", type_key)
                continue
            entities.append(
                FinancialEntity(
                    entity_type=e_type,
                    raw_value=ent.get("value", ""),
                    normalized_value=ent.get("value", ""),
                    confidence=ent.get("confidence", data.get("confidence", 0.0)),
                    detection_method=DetectionMethod.LLM_BASED,
                )
            )

        intent_result = IntentResult(
            intent_type=intent_type,
            intent_category=self._infer_category(intent_type),
            confidence=data.get("confidence", 0.0),
            entities=entities,
            method=DetectionMethod.LLM_BASED,
            processing_time_ms=(time.perf_counter() - start_time) * 1000,
        )

        result = {
            "content": json.dumps(data),
            "metadata": {
                "intent_result": intent_result,
                "detection_method": DetectionMethod.LLM_BASED,
                "confidence": intent_result.confidence,
                "intent_type": intent_result.intent_type,
                "entities": [
                    e.model_dump() if hasattr(e, "model_dump") else e.__dict__
                    for e in intent_result.entities
                ],
            },
            "confidence_score": intent_result.confidence,
        }

        self._intent_cache.set(user_message, intent_result)

        return result

    # ------------------------------------------------------------------
    @staticmethod
    def _infer_category(intent: str) -> IntentCategory:
        """Map an intent string to an :class:`IntentCategory`."""

        if intent.startswith("SEARCH") or intent == "TRANSACTION_SEARCH":
            return IntentCategory.TRANSACTION_SEARCH
        if (
            intent.startswith("ANALYZE")
            or intent.endswith("ANALYSIS")
            or intent in {"SPENDING_ANALYSIS", "COUNT_TRANSACTIONS"}
        ):
            return IntentCategory.SPENDING_ANALYSIS
        if intent in {"BALANCE_INQUIRY", "BALANCE_CHECK"}:
            return IntentCategory.BALANCE_INQUIRY
        if intent == "GREETING":
            return IntentCategory.GREETING
        return IntentCategory.GENERAL_QUESTION

    # ------------------------------------------------------------------
    def get_cache_metrics(self) -> Dict[str, int]:
        """Expose intent cache metrics for monitoring."""
        return self._intent_cache.get_metrics()

    def _parse_llm_output(self, llm_output: str) -> IntentResult:
        """Parse LLM JSON output into IntentResult."""
        try:
            data = json.loads(llm_output)
        except json.JSONDecodeError as exc:
            logger.warning("Failed to parse LLM output: %s", exc)
            return IntentResult(
                intent_type="GENERAL",
                intent_category=IntentCategory.GENERAL_QUESTION,
                confidence=0.0,
                entities=[],
                method=DetectionMethod.FALLBACK,
                processing_time_ms=0.0,
            )

        intent_type = data.get("intent", "GENERAL").upper()
        entities_data = data.get("entities", [])
        entities: List[FinancialEntity] = []
        for ent in entities_data:
            ent_type_str = str(ent.get("entity_type") or ent.get("type", "OTHER")).upper()
            value = ent.get("value")
            try:
                entity_type = EntityType(ent_type_str)
            except ValueError:
                entity_type = EntityType.OTHER
            entities.append(
                FinancialEntity(
                    entity_type=entity_type,
                    raw_value=str(value),
                    normalized_value=value,
                    confidence=0.9,
                    start_position=None,
                    end_position=None,
                    detection_method=DetectionMethod.LLM_BASED,
                )
            )

        if intent_type == "SEARCH_BY_OPERATION_TYPE":
            category = IntentCategory.FINANCIAL_QUERY
        elif intent_type in {
            "SEARCH_BY_MERCHANT",
            "SEARCH_BY_CATEGORY",
            "SEARCH_BY_TEXT",
        }:
            category = IntentCategory.TRANSACTION_SEARCH
        elif intent_type in {
            "SPENDING_ANALYSIS",
            "CATEGORY_ANALYSIS",
            "COUNT_TRANSACTIONS",
        }:
            category = IntentCategory.SPENDING_ANALYSIS
        elif intent_type in {"BALANCE_CHECK", "BALANCE_INQUIRY"}:
            category = IntentCategory.BALANCE_INQUIRY
        elif intent_type == "GREETING":
            category = IntentCategory.GREETING
        else:
            category = IntentCategory.GENERAL_QUESTION

        return IntentResult(
            intent_type=intent_type,
            intent_category=category,
            confidence=data.get("confidence", 0.9),
            entities=entities,
            method=DetectionMethod.LLM_BASED,
            processing_time_ms=0.0,
        )

