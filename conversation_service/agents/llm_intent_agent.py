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

import json
import time
import logging
import time
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

logger = logging.getLogger(__name__)


ALLOWED_INTENTS = [
    "SEARCH_BY_TEXT",
    "SEARCH_BY_MERCHANT",
    "SEARCH_BY_CATEGORY",
    "SEARCH_BY_AMOUNT",
    "SEARCH_BY_DATE",
    "SEARCH_BY_OPERATION_TYPE",
    "ANALYZE_SPENDING",
    "ANALYZE_TRENDS",
    "COUNT_TRANSACTIONS",
    "TRANSACTION_SEARCH",
    "SPENDING_ANALYSIS",
    "BALANCE_INQUIRY",
    "GENERAL_QUESTION",
    "GREETING",
    "OUT_OF_SCOPE",
]


class LLMIntentAgent(BaseFinancialAgent):
    """Intent detection agent that relies solely on the DeepSeek LLM."""
    """Intent detection agent relying solely on DeepSeek LLM."""

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

        intents = ", ".join(ALLOWED_INTENTS)
        entity_types = ", ".join(e.value for e in EntityType)
        return (
            "Tu es un classificateur d'intentions financières."
            "\nIntents autorisés : "
            + intents
            + "\nTypes d'entités autorisés : "
            + entity_types
            + "\nIgnore toute autre catégorie."
            + "\nRéponds uniquement avec un JSON strict au format :"
            ' {"intent": "INTENT", "confidence": 0-1,'
            ' "entities": [{"type": "TYPE", "value": "VALEUR"}]}.'
        )

    # ------------------------------------------------------------------
    async def detect_intent(self, user_message: str, user_id: int) -> Dict[str, Any]:
        """Detect the intent of ``user_message``.

        The DeepSeek client is expected to return a JSON document containing the
        intent and a list of extracted entities.  The method converts this output
        into an :class:`IntentResult` instance and wraps it in the structure used
        by the rest of the service.
        """

        start = time.perf_counter()

    async def _execute_operation(
        self, input_data: Dict[str, Any], user_id: int
    ) -> Dict[str, Any]:
        user_message = input_data.get("user_message", "")
        if not user_message:
            raise ValueError("user_message is required for intent detection")
        return await self.detect_intent(user_message, user_id)

    async def detect_intent(
        self, user_message: str, user_id: int
    ) -> Dict[str, Any]:
        start_time = time.perf_counter()
        response = await self.deepseek_client.generate_response(
            messages=[
                {"role": "system", "content": self.config.system_message},
                {"role": "user", "content": user_message},
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            user=str(user_id),
            use_cache=True,
        )
        try:
            data = json.loads(response.content)
        except Exception as err:  # pragma: no cover - defensive fallback
            logger.warning("Failed to parse LLM output: %s", err)
            data = {"intent": "OUT_OF_SCOPE", "confidence": 0.0, "entities": []}

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
            processing_time_ms=(time.perf_counter() - start) * 1000,
        )

        return {
            "content": json.dumps(data),
            "metadata": {
                "intent_result": intent_result,
                "detection_method": DetectionMethod.LLM_BASED,
                "confidence": intent_result.confidence,
                "intent_type": intent_result.intent_type,
                "entities": [
                    e.model_dump() if hasattr(e, "model_dump") else e.__dict__
        intent_result = self._parse_llm_output(response.content)
        intent_result.processing_time_ms = (time.perf_counter() - start_time) * 1000

        return {
            "content": response.content,
            "metadata": {
                "intent_result": intent_result,
                "detection_method": intent_result.method,
                "confidence": intent_result.confidence,
                "intent_type": intent_result.intent_type,
                "entities": [
                    e.model_dump() if hasattr(e, "model_dump") else e.dict()
                    for e in intent_result.entities
                ],
                "intent_detected": intent_result.intent_type,
                "entities_extracted": [
                    e.model_dump() if hasattr(e, "model_dump") else e.__dict__
                    e.model_dump() if hasattr(e, "model_dump") else e.dict()
                    for e in intent_result.entities
                ],
            },
            "confidence_score": intent_result.confidence,
        }

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
                method=DetectionMethod.AI_PARSE_FALLBACK,
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

        if intent_type in {
            "SEARCH_BY_MERCHANT",
            "SEARCH_BY_CATEGORY",
            "SEARCH_BY_TEXT",
            "COUNT_TRANSACTIONS",
        }:
            category = IntentCategory.TRANSACTION_SEARCH
        elif intent_type in {"SPENDING_ANALYSIS", "CATEGORY_ANALYSIS"}:
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

