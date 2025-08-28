"""Autogen-based agent for financial intent classification."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import Counter
from typing import Any, Dict

from autogen import AssistantAgent
from conversation_service.prompts.autogen.intent_classification_prompts import (
    AUTOGEN_INTENT_SYSTEM_MESSAGE,
)

try:
    from autogen.agentchat.contrib.capabilities.teachability import Teachability
except Exception:  # pragma: no cover - capability is optional
    Teachability = None


class IntentClassifierAgent(AssistantAgent):
    """Assistant agent configured for financial intent classification."""

    def __init__(self, *_, **__):
        super().__init__(
            name="intent_classifier",
            system_message=AUTOGEN_INTENT_SYSTEM_MESSAGE,
            llm_config=self._create_llm_config(),
            human_input_mode="NEVER",
            code_execution_config=False,
            max_consecutive_auto_reply=1,
        )
        if Teachability is not None:
            self.add_capability(Teachability(verbosity=0))

        # Metrics and caching
        self.success_count = 0
        self.error_count = 0
        self.intent_cache: Dict[str, Dict[str, Any]] = {}
        self.intent_frequency: Counter[str] = Counter()

        # Dedicated logger
        self.logger = logging.getLogger("IntentClassifierAgent")

    @staticmethod
    def _create_llm_config() -> Dict[str, Any]:
        return {
            "config_list": [
                {
                    "model": "deepseek-chat",
                    "response_format": {"type": "json_object"},
                    "temperature": 0.1,
                    "max_tokens": 800,
                    "cache_seed": 42,
                }
            ]
        }

    async def classify_for_team(self, user_message: str, user_id: int) -> Dict[str, Any]:
        """Classify an intent and enrich it with team context.

        The method calls the LLM with a timeout, parses the JSON response and
        augments it with additional metadata useful for downstream teams.
        In case of timeout or JSON errors, a fallback response is returned.
        """

        if user_message in self.intent_cache:
            self.logger.debug("Cache hit for message")
            self.success_count += 1
            return self.intent_cache[user_message]

        start_time = time.monotonic()
        try:
            raw_reply = await asyncio.wait_for(
                self.a_generate_reply(user_message), timeout=30
            )
            content = raw_reply if isinstance(raw_reply, str) else str(raw_reply)
            parsed = json.loads(content)
            intent = parsed.get("intent", "GENERAL_INQUIRY")
            confidence = float(parsed.get("confidence", 0.0))

            team_context = {
                "original_message": user_message,
                "user_id": user_id,
                "ready_for_entity_extraction": confidence >= 0.5,
                "suggested_entities_focus": self.suggest_entities_focus(
                    intent, confidence
                ),
            }

            result = {**parsed, "team_context": team_context}
            self.intent_cache[user_message] = result
            self.intent_frequency[intent] += 1
            self.success_count += 1

            elapsed = time.monotonic() - start_time
            self.logger.info(
                "Intent classified in %.2fs for user %s", elapsed, user_id
            )
            return result
        except json.JSONDecodeError:
            self.error_count += 1
            self.logger.error("Malformed JSON from intent classifier: %s", content)
        except asyncio.TimeoutError:
            self.error_count += 1
            self.logger.error("Intent classification timeout for user %s", user_id)
        except Exception as exc:  # pragma: no cover - unexpected runtime issues
            self.error_count += 1
            self.logger.exception(
                "Intent classification failed for user %s: %s", user_id, exc
            )

        # Fallback response on error
        fallback = {
            "intent": "GENERAL_INQUIRY",
            "confidence": 0.3,
            "team_context": {
                "original_message": user_message,
                "user_id": user_id,
                "ready_for_entity_extraction": False,
                "suggested_entities_focus": self.suggest_entities_focus(
                    "GENERAL_INQUIRY", 0.3
                ),
            },
        }
        return fallback

    def suggest_entities_focus(self, intent_type: str, confidence: float) -> Dict[str, Any]:
        """Suggest entity extraction focus based on intent and confidence."""

        if confidence < 0.5:
            return {"priority_entities": [], "strategy": "clarify"}

        intent_type = intent_type.upper()
        if intent_type.startswith("SEARCH") or intent_type.endswith("TRANSACTION") or intent_type in {
            "TRANSACTION_SEARCH",
            "SEARCH_BY_DATE",
            "SEARCH_BY_AMOUNT",
            "SEARCH_BY_MERCHANT",
            "SEARCH_BY_CATEGORY",
            "SEARCH_BY_AMOUNT_AND_DATE",
            "SEARCH_BY_OPERATION_TYPE",
            "SEARCH_BY_TEXT",
            "COUNT_TRANSACTIONS",
            "MERCHANT_INQUIRY",
            "FILTER_REQUEST",
        }:
            return {
                "priority_entities": ["date", "amount", "merchant", "category"],
                "strategy": "transaction_filters",
            }

        if intent_type in {
            "SPENDING_ANALYSIS",
            "SPENDING_ANALYSIS_BY_CATEGORY",
            "SPENDING_ANALYSIS_BY_PERIOD",
            "SPENDING_COMPARISON",
            "TREND_ANALYSIS",
            "CATEGORY_ANALYSIS",
            "COMPARISON_QUERY",
        }:
            return {
                "priority_entities": ["category", "date_range"],
                "strategy": "spending_analysis",
            }

        if intent_type in {
            "BALANCE_INQUIRY",
            "ACCOUNT_BALANCE_SPECIFIC",
            "BALANCE_EVOLUTION",
        }:
            return {
                "priority_entities": ["account"],
                "strategy": "account_lookup",
            }

        return {"priority_entities": [], "strategy": "generic"}

    async def classify_intent(self, *args, **kwargs):  # pragma: no cover
        """Placeholder classification method."""
        raise NotImplementedError("Intent classification not implemented.")
