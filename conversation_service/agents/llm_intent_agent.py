"""LLM-only intent detection agent.

This module defines :class:`LLMIntentAgent`, a simple agent that relies on the
OpenAI API to classify user messages into a restricted set of financial
intents and to extract entities.  The tests exercise only a small portion of the
original project, so the implementation here focuses on the functionality
required by the unit tests: constructing a system prompt, sending a structured
output request to OpenAI and returning an :class:`~IntentResult` instance
describing the detected intent.
"""

from __future__ import annotations

import asyncio
import json
import time
import logging
import os
import unicodedata
import re
from typing import Any, Dict, List, Optional
from datetime import datetime

try:  # pragma: no cover - library may be absent in tests
    from openai import AsyncOpenAI
except Exception:  # pragma: no cover - fall back to a stub
    AsyncOpenAI = None  # type: ignore

from .base_financial_agent import BaseFinancialAgent
from ..models.agent_models import AgentConfig
from ..models.financial_models import (
    DetectionMethod,
    EntityType,
    FinancialEntity,
    IntentCategory,
    IntentResult,
)
from ..utils.intent_cache import IntentResultCache
from ..core.taxonomies import get_taxonomy
from ..prompts.intent_prompts import (
    INTENT_SYSTEM_PROMPT,
    INTENT_EXAMPLES_FEW_SHOT,
)
from ..constants import TRANSACTION_TYPES

# Mapping to harmonise categories with internal enums
CATEGORY_MAP: Dict[str, str] = {
    "ACCOUNT_BALANCE": "BALANCE_INQUIRY",
    "GENERAL_QUESTION": "UNCLEAR_INTENT",
}

logger = logging.getLogger(__name__)

FRENCH_MONTHS = [
    "janvier",
    "février",
    "fevrier",
    "mars",
    "avril",
    "mai",
    "juin",
    "juillet",
    "août",
    "aout",
    "septembre",
    "octobre",
    "novembre",
    "décembre",
    "decembre",
]
# Regex capturing expressions like "en juin", "au mois de mai" or "pendant mars",
# optionally followed by a year
MONTH_PHRASE_REGEX = re.compile(
    r"(?:\b(?:en|au\s+mois\s+de|mois\s+de|durant|pendant)\s+)?\b(" + "|".join(FRENCH_MONTHS) + r")\b(?:\s+(\d{4}))?",
    re.IGNORECASE,
)


class LLMOutputParsingError(RuntimeError):
    """Raised when the LLM response cannot be parsed as JSON."""


class LLMIntentAgent(BaseFinancialAgent):
    """Intent detection agent using OpenAI structured outputs."""

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        openai_client: Optional[Any] = None,
    ) -> None:
        api_key = os.getenv("OPENAI_API_KEY", "")
        if config is None:
            config = AgentConfig(
                name="llm_intent_agent",
                model_client_config={
                    "model": "gpt-4o-mini",
                    "api_key": api_key,
                    "base_url": "https://api.openai.com/v1",
                },
                system_message=self._build_system_message(),
                max_consecutive_auto_reply=1,
                description="LLM-based intent detection agent",
                temperature=0.0,
                max_tokens=200,
                timeout_seconds=8,
            )
        else:
            config.model_client_config.setdefault("api_key", api_key)

        super().__init__(name=config.name, config=config)
        self._intent_cache = IntentResultCache(max_size=100)
        self._max_retries = 3
        if openai_client is not None:
            self._openai_client = openai_client
        elif AsyncOpenAI is not None:
            self._openai_client = AsyncOpenAI(api_key=config.model_client_config["api_key"])
        else:  # pragma: no cover - requires openai package
            raise ImportError("openai package is required unless openai_client is provided")

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
        allowed = " ou ".join(TRANSACTION_TYPES)
        return (
            f"{INTENT_SYSTEM_PROMPT}\nLes valeurs autorisées pour transaction_type sont : {allowed}."
            f"\n\n{INTENT_EXAMPLES_FEW_SHOT}"
            "\n\nRéponds uniquement avec un JSON strict."
        )

    @staticmethod
    def _detect_transaction_type(message: str) -> Optional[List[str]]:
        """Detect basic keywords implying transaction type.

        Returns a list of detected transaction types.  Multiple types may be
        returned when the message mentions both credits and debits (e.g.
        "entrées et sorties").
        """

        normalized = unicodedata.normalize("NFD", message).encode(
            "ascii", "ignore"
        ).decode("utf-8").lower()

        debit_keywords = [
            "depense",
            "depenses",
            "depensees",
            "depenser",
            "sortie",
            "sorties",
            "debit",
            "debits",
        ]
        credit_keywords = [
            "entree",
            "entrees",
            "entree d argent",
            "entree d'argent",
            "gains",
            "gain",
            "credit",
            "credits",
        ]

        types: List[str] = []
        if any(k in normalized for k in debit_keywords):
            types.append("debit")
        if any(k in normalized for k in credit_keywords):
            types.append("credit")
        return types or None

    @staticmethod
    def _normalise_amount(value: Any) -> Any:
        """Convert amount-like strings to floats.

        This strips common currency units and whitespace, handling both
        simple numeric strings and dictionaries containing numeric values.
        """

        if isinstance(value, str):
            cleaned = (
                value.replace("€", "")
                .replace("euros", "")
                .replace("euro", "")
                .replace(",", ".")
                .strip()
            )
            cleaned = re.sub(r"[^0-9.\-]", "", cleaned.lower())
            try:
                return float(cleaned)
            except ValueError:
                return value
        if isinstance(value, dict):
            return {
                k: LLMIntentAgent._normalise_amount(v) for k, v in value.items()
            }
        return value

    @staticmethod
    def _extract_months(message: str) -> List[tuple[str, str]]:
        """Return list of (raw_month, normalized_with_year) found in ``message``."""
        current_year = datetime.utcnow().year
        results: List[tuple[str, str]] = []
        for match in MONTH_PHRASE_REGEX.finditer(message):
            month = match.group(1).lower()
            year = match.group(2) or str(current_year)
            normalized = f"{month} {year}"
            results.append((month, normalized))
        return results

    def _regex_fallback(self, user_message: str) -> Dict[str, Any]:
        """Fallback intent detection using regex for French months."""
        months = self._extract_months(user_message)
        entities = [
            FinancialEntity(
                entity_type=EntityType.DATE,
                raw_value=raw,
                normalized_value=norm,
                confidence=0.5,
                detection_method=DetectionMethod.FALLBACK,
            )
            for raw, norm in months
        ]
        intent_type = "SEARCH_BY_DATE" if entities else "GENERAL_QUESTION"
        intent_category = (
            IntentCategory.FINANCIAL_QUERY
            if entities
            else IntentCategory.GENERAL_QUESTION
        )
        confidence = 0.5 if entities else 0.0
        data = {
            "intent_type": intent_type,
            "intent_category": intent_category.value,
            "confidence": confidence,
            "entities": [e.model_dump() for e in entities],
        }
        intent_result = IntentResult(
            intent_type=intent_type,
            intent_category=intent_category,
            confidence=confidence,
            entities=entities,
            method=DetectionMethod.FALLBACK,
            processing_time_ms=0.0,
        )
        return {
            "content": json.dumps(data),
            "metadata": {
                "intent_result": intent_result,
                "detection_method": DetectionMethod.FALLBACK,
                "confidence": confidence,
                "intent_type": intent_type,
                "intent_category": intent_category.value,
                "entities": [e.model_dump() for e in entities],
                "suggested_actions": None,
            },
            "confidence_score": confidence,
        }

    # ------------------------------------------------------------------
    async def detect_intent(
        self, user_message: str, user_id: int
    ) -> Dict[str, Any]:
        """Detect the intent of ``user_message``.

        The OpenAI client returns a JSON document describing the intent and any
        extracted entities.  This method converts that JSON into an
        :class:`IntentResult` and wraps it in the structure used by the rest of
        the service.
        """

        cached = self._intent_cache.get(user_message)
        if cached is not None:
            data = {
                "intent_type": cached.intent_type,
                "intent_category": cached.intent_category.value
                if hasattr(cached.intent_category, "value")
                else str(cached.intent_category),
                "confidence": cached.confidence,
                "entities": [e.model_dump() for e in cached.entities],
            }
            if cached.suggested_actions:
                data["suggested_actions"] = cached.suggested_actions
            return {
                "content": json.dumps(data),
                "metadata": {
                    "intent_result": cached,
                    "detection_method": DetectionMethod.LLM_BASED,
                    "confidence": cached.confidence,
                    "intent_type": cached.intent_type,
                    "intent_category": data["intent_category"],
                    "entities": [e.model_dump() for e in cached.entities],
                    "suggested_actions": cached.suggested_actions,
                    "cache_hit": True,
                },
                "confidence_score": cached.confidence,
            }

        start_time = time.perf_counter()
        messages = [
            {"role": "system", "content": self.config.system_message},
            {"role": "user", "content": user_message},
        ]

        schema = {
            "type": "object",
            "properties": {
                "intent_type": {"type": "string"},
                "intent_category": {"type": "string"},
                "confidence": {"type": "number"},
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "entity_type": {"type": "string"},
                            "value": {"type": "string"},
                            "confidence": {"type": "number"},
                            "normalized_value": {
                                "oneOf": [
                                    {"type": "string"},
                                    {"type": "number"},
                                    {"type": "object"},
                                ]
                            },
                        },
                        "required": ["entity_type", "value", "confidence"],
                    },
                },
            },
            "required": ["intent_type", "intent_category", "confidence", "entities"],
        }

        try:
            response = None
            for attempt in range(self._max_retries):
                try:
                    response = await self._openai_client.chat.completions.create(
                        model=self.config.model_client_config["model"],
                        messages=messages,
                        response_format={
                            "type": "json_schema",
                            "json_schema": {"name": "intent_result", "schema": schema},
                        },
                        temperature=self.config.temperature,
                        max_tokens=self.config.max_tokens,
                    )
                    break
                except Exception as err:  # pragma: no cover - retry logic
                    logger.warning(
                        "OpenAI call failed (attempt %s): %s", attempt + 1, err
                    )
                    await asyncio.sleep(2 ** attempt)
            if response is None:
                raise RuntimeError("LLM call failed")

            data = json.loads(response.choices[0].message.content)
        except Exception as err:  # pragma: no cover - fallback on regex
            logger.warning("LLM processing failed, using regex fallback: %s", err)
            return self._regex_fallback(user_message)
        suggested_actions = data.get("suggested_actions")
        if isinstance(suggested_actions, str):
            suggested_actions = [suggested_actions]

        intent_type = data.get("intent_type", "OUT_OF_SCOPE")
        taxonomy = get_taxonomy(intent_type)
        if taxonomy:
            category = taxonomy.category
            if not suggested_actions:
                suggested_actions = taxonomy.suggested_actions
        raw_category = data.get("intent_category", "GENERAL_QUESTION").upper()
        mapped_category = CATEGORY_MAP.get(raw_category, raw_category)
        try:
            intent_category = IntentCategory(mapped_category)
        except Exception:
            intent_category = IntentCategory.GENERAL_QUESTION

        confidence = float(data.get("confidence", 0.85))

        entities: List[FinancialEntity] = []
        for ent in data.get("entities", []):
            type_key = ent.get("entity_type") or ent.get("type") or ""
            try:
                e_type = EntityType(type_key.upper())
            except Exception:  # pragma: no cover - unknown entity types
                logger.debug("Unknown entity type: %s", type_key)
                continue
            ent_conf = float(ent.get("confidence", confidence))
            raw_value = ent.get("value", "")
            normalized = ent.get("normalized_value", raw_value)
            normalized = self._normalise_amount(normalized)
            entities.append(
                FinancialEntity(
                    entity_type=e_type,
                    raw_value=raw_value,
                    normalized_value=normalized,
                    confidence=ent_conf,
                    detection_method=DetectionMethod.LLM_BASED,
                )
            )
        # Inject month entities when the LLM fails to extract them
        if not any(e.entity_type == EntityType.DATE for e in entities):
            months = self._extract_months(user_message)
            if months:
                for raw, norm in months:
                    ent = FinancialEntity(
                        entity_type=EntityType.DATE,
                        raw_value=raw,
                        normalized_value=norm,
                        confidence=0.5,
                        detection_method=DetectionMethod.FALLBACK,
                    )
                    entities.append(ent)
                    data.setdefault("entities", []).append(
                        {
                            "entity_type": EntityType.DATE.value,
                            "value": raw,
                            "confidence": 0.5,
                            "normalized_value": norm,
                        }
                    )
                if intent_type in {"GENERAL_QUESTION", "OUT_OF_SCOPE"}:
                    intent_type = "SEARCH_BY_DATE"
                    intent_category = IntentCategory.FINANCIAL_QUERY
                    confidence = 0.5
                    data.update(
                        {
                            "intent_type": intent_type,
                            "intent_category": intent_category.value,
                            "confidence": confidence,
                        }
                    )

        # Apply simple keyword-based transaction type heuristics
        tx_types = self._detect_transaction_type(user_message)
        if tx_types and not any(
            e.entity_type == EntityType.TRANSACTION_TYPE for e in entities
        ):
            raw_value = ", ".join(tx_types)
            normalized_value: Any = tx_types[0] if len(tx_types) == 1 else tx_types
            entities.append(
                FinancialEntity(
                    entity_type=EntityType.TRANSACTION_TYPE,
                    raw_value=raw_value,
                    normalized_value=normalized_value,
                    confidence=confidence,
                    detection_method=DetectionMethod.RULE_BASED,
                )
            )
            data.setdefault("entities", []).append(
                {
                    "entity_type": EntityType.TRANSACTION_TYPE.value,
                    "value": raw_value,
                    "confidence": confidence,
                    "normalized_value": normalized_value,
                }
            )

        if suggested_actions is not None:
            data["suggested_actions"] = suggested_actions

        intent_result = IntentResult(
            intent_type=intent_type,
            intent_category=intent_category,
            confidence=confidence,
            entities=entities,
            method=DetectionMethod.LLM_BASED,
            processing_time_ms=(time.perf_counter() - start_time) * 1000,
            suggested_actions=suggested_actions,
        )

        result = {
            "content": json.dumps(data),
            "metadata": {
                "intent_result": intent_result,
                "detection_method": DetectionMethod.LLM_BASED,
                "confidence": intent_result.confidence,
                "intent_type": intent_result.intent_type,
                "intent_category": intent_result.intent_category.value
                if hasattr(intent_result.intent_category, "value")
                else str(intent_result.intent_category),
                "entities": [
                    e.model_dump() if hasattr(e, "model_dump") else e.__dict__
                    for e in intent_result.entities
                ],
                "suggested_actions": suggested_actions,
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
            ent_conf = float(ent.get("confidence", 0.9))
            normalized = ent.get("normalized_value", value)
            if isinstance(normalized, str):
                try:
                    normalized = float(normalized)
                except ValueError:
                    pass
            try:
                entity_type = EntityType(ent_type_str)
            except ValueError:
                entity_type = EntityType.OTHER
            entities.append(
                FinancialEntity(
                    entity_type=entity_type,
                    raw_value=str(value),
                    normalized_value=normalized,
                    confidence=ent_conf,
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

        suggested_actions = data.get("suggested_actions")
        if isinstance(suggested_actions, str):
            suggested_actions = [suggested_actions]

        taxonomy = get_taxonomy(intent_type)
        if taxonomy:
            category = taxonomy.category
            if not suggested_actions:
                suggested_actions = taxonomy.suggested_actions

        return IntentResult(
            intent_type=intent_type,
            intent_category=category,
            confidence=data.get("confidence", 0.9),
            entities=entities,
            method=DetectionMethod.LLM_BASED,
            processing_time_ms=0.0,
            suggested_actions=suggested_actions,
        )

