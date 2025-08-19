"""Enhanced LLM Intent Agent.

This module introduces :class:`EnhancedLLMIntentAgent`, an extension of
:class:`LLMIntentAgent` adding robust error handling and latency
measurement. The agent wraps the base implementation with a fallback
mechanism so that the system can gracefully recover from LLM failures.
The measured latency is stored in the returned :class:`IntentResult`.
"""

from __future__ import annotations

import json
import time
import logging
from typing import Any, Dict, Optional

from .llm_intent_agent import LLMIntentAgent
from .base_financial_agent import BaseFinancialAgent
from ..core.deepseek_client import DeepSeekClient
from ..models.agent_models import AgentConfig
from ..models.financial_models import (
    DetectionMethod,
    IntentCategory,
    IntentResult,
)

logger = logging.getLogger(__name__)


class EnhancedLLMIntentAgent(LLMIntentAgent):
    """LLM intent agent with fallback and latency tracking."""

    def __init__(
        self,
        deepseek_client: DeepSeekClient,
        fallback_agent: Optional[BaseFinancialAgent] = None,
        config: Optional[AgentConfig] = None,
        openai_client: Optional[Any] = None,
    ) -> None:
        super().__init__(deepseek_client=deepseek_client, config=config, openai_client=openai_client)
        self.fallback_agent = fallback_agent

    async def detect_intent(self, user_message: str, user_id: int) -> Dict[str, Any]:
        """Detect intent and measure latency, with graceful fallback."""

        start_time = time.perf_counter()
        try:
            result = await super().detect_intent(user_message, user_id)
            intent_result: IntentResult = result["metadata"]["intent_result"]
            intent_result.processing_time_ms = (
                time.perf_counter() - start_time
            ) * 1000
            # If LLM failed and fallback is available, use fallback result
            if (
                self.fallback_agent
                and intent_result.intent_type == "OUT_OF_SCOPE"
                and intent_result.confidence == 0.0
            ):
                raise RuntimeError("LLM intent detection failed")
            if (
                intent_result.intent_type == "OUT_OF_SCOPE"
                and self.fallback_agent is not None
            ):
                fallback = await self.fallback_agent.detect_intent(
                    user_message, user_id
                )
                intent_result = fallback["metadata"]["intent_result"]
                intent_result.method = DetectionMethod.FALLBACK
                intent_result.processing_time_ms = (
                    time.perf_counter() - start_time
                ) * 1000
                fallback["metadata"].update(
                    {
                        "detection_method": DetectionMethod.FALLBACK,
                        "intent_result": intent_result,
                        "confidence": intent_result.confidence,
                        "intent_type": intent_result.intent_type,
                        "entities": [
                            e.model_dump() if hasattr(e, "model_dump") else e.__dict__
                            for e in intent_result.entities
                        ],
                    }
                )
                fallback["confidence_score"] = intent_result.confidence
                return fallback
            return result
        except Exception as exc:  # pragma: no cover - tested via fallback
            logger.warning("LLM call failed, falling back: %s", exc)
            if self.fallback_agent:
                fallback = await self.fallback_agent.detect_intent(
                    user_message, user_id
                )
                intent_result = fallback["metadata"]["intent_result"]
                intent_result.method = DetectionMethod.FALLBACK
                intent_result.processing_time_ms = (
                    time.perf_counter() - start_time
                ) * 1000
                fallback["metadata"].update(
                    {
                        "detection_method": DetectionMethod.FALLBACK,
                        "intent_result": intent_result,
                        "confidence": intent_result.confidence,
                        "intent_type": intent_result.intent_type,
                        "entities": [
                            e.model_dump() if hasattr(e, "model_dump") else e.__dict__
                            for e in intent_result.entities
                        ],
                    }
                )
                fallback["confidence_score"] = intent_result.confidence
                return fallback

            intent_result = IntentResult(
                intent_type="OUT_OF_SCOPE",
                intent_category=IntentCategory.GENERAL_QUESTION,
                confidence=0.0,
                entities=[],
                method=DetectionMethod.FALLBACK,
                processing_time_ms=(time.perf_counter() - start_time) * 1000,
            )
            return {
                "content": json.dumps(
                    {
                        "intent": intent_result.intent_type,
                        "confidence": intent_result.confidence,
                        "entities": [],
                    }
                ),
                "metadata": {
                    "intent_result": intent_result,
                    "detection_method": DetectionMethod.FALLBACK,
                    "confidence": 0.0,
                    "intent_type": intent_result.intent_type,
                    "entities": [],
                },
                "confidence_score": 0.0,
            }
