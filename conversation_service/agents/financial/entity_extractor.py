"""Autogen-based agent for financial entity extraction."""
from __future__ import annotations

import json
import logging
import asyncio
from datetime import date
from typing import Any, Dict, Tuple

from autogen import AssistantAgent

from conversation_service.models.conversation import (
    CategoryEntity,
    EntityExtractionResult,
    ExtractedAmount,
    ExtractedDate,
    ExtractedMerchant,
    TransactionTypeEntity,
)
from conversation_service.prompts.autogen.collaboration_prompts import (
    AUTOGEN_ENTITY_EXTRACTION_SYSTEM_MESSAGE,
    get_entity_extraction_prompt_for_autogen,
)


class EntityExtractorAgent(AssistantAgent):
    """Assistant agent configured for financial entity extraction."""

    def __init__(self, *_, intent_context: dict | None = None, **__):
        system_message = get_entity_extraction_prompt_for_autogen(intent_context)
        super().__init__(
            name="entity_extractor",
            system_message=system_message,
            llm_config=self._create_llm_config(),
            human_input_mode="NEVER",
            code_execution_config=False,
            max_consecutive_auto_reply=1,
        )

        self.base_system_message = AUTOGEN_ENTITY_EXTRACTION_SYSTEM_MESSAGE

        # Metrics and caching
        self.success_count = 0
        self.error_count = 0
        self.extraction_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}

        # Dedicated logger
        self.logger = logging.getLogger("EntityExtractorAgent")

    @staticmethod
    def _create_llm_config() -> Dict[str, Any]:
        return {
            "config_list": [
                {
                    "model": "deepseek-chat",
                    "response_format": {"type": "json_object"},
                }
            ],
            "temperature": 0.05,
            "max_tokens": 1500,
            "cache_seed": 42,
        }

    def _build_extraction_prompt(self, user_message: str, intent_type: str) -> str:
        """Build the extraction prompt injecting team intent context."""

        return f"INTENT: {intent_type}\nUSER_MESSAGE: {user_message}"

    async def extract_entities_from_team_context(self, team_context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract entities for a given team context."""

        user_message = team_context.get("user_message") or team_context.get(
            "original_message", ""
        )
        intent_type = team_context.get("intent") or team_context.get("intent_type", "")
        cache_key = (user_message, intent_type)
        if cache_key in self.extraction_cache:
            self.logger.debug("Cache hit for message and intent")
            self.success_count += 1
            return self.extraction_cache[cache_key]

        prompt = self._build_extraction_prompt(user_message, intent_type)
        try:
            raw_reply = await asyncio.wait_for(
                self.a_generate_reply(prompt), timeout=5
            )
            parsed = self._parse_extraction_response(raw_reply)
            parsed["team_context"] = {
                **team_context,
                **parsed.get("team_context", {}),
            }
            if parsed.get("extraction_success"):
                self.extraction_cache[cache_key] = parsed
                self.success_count += 1
            else:
                self.error_count += 1
            return parsed
        except asyncio.TimeoutError:
            self.error_count += 1
            extraction_metadata = {"error": "timeout"}
        except Exception as exc:  # pragma: no cover - unexpected runtime issues
            self.error_count += 1
            self.logger.exception("Entity extraction failed: %s", exc)
            extraction_metadata = {"error": str(exc)}

        return {
            "extraction_success": False,
            "entities": EntityExtractionResult(
                extraction_metadata=extraction_metadata
            ),
            "team_context": team_context,
        }

    def _parse_extraction_response(self, raw_response: Any) -> Dict[str, Any]:
        """Parse the raw LLM response into structured entities."""

        content = raw_response if isinstance(raw_response, str) else str(raw_response)
        try:
            parsed = json.loads(content)
        except Exception as exc:
            return {
                "extraction_success": False,
                "entities": EntityExtractionResult(
                    extraction_metadata={"error": f"invalid_json: {exc}"}
                ),
                "team_context": {},
            }

        valid, validation_metadata = self._validate_extracted_entities(parsed)
        if not valid:
            return {
                "extraction_success": False,
                "entities": EntityExtractionResult(
                    extraction_metadata=validation_metadata
                ),
                "team_context": parsed.get("team_context", {}),
            }

        entities_list = parsed.get("entities", [])
        extraction_metadata = parsed.get("extraction_metadata", {})
        result = EntityExtractionResult(extraction_metadata=extraction_metadata)

        for entity in entities_list:
            etype = entity.get("type")
            value = entity.get("value")
            if etype == "amount":
                try:
                    result.amounts.append(
                        ExtractedAmount(
                            value=float(value),
                            currency=entity.get("currency", ""),
                        )
                    )
                except Exception:  # pragma: no cover - skip malformed amounts
                    continue
            elif etype == "merchant":
                result.merchants.append(ExtractedMerchant(name=str(value)))
            elif etype == "date":
                try:
                    result.dates.append(
                        ExtractedDate(date=date.fromisoformat(str(value)))
                    )
                except Exception:  # pragma: no cover - skip malformed dates
                    continue
            elif etype == "category":
                result.categories.append(CategoryEntity(name=str(value)))
            elif etype == "transaction_type":
                result.transaction_types.append(
                    TransactionTypeEntity(transaction_type=str(value))
                )

        return {
            "extraction_success": bool(parsed.get("extraction_success", False)),
            "entities": result,
            "team_context": parsed.get("team_context", {}),
        }

    def _validate_extracted_entities(self, entities_data: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Validate extracted entities structure and values."""

        try:
            entities = entities_data.get("entities")
            if not isinstance(entities, list):
                raise ValueError("entities must be a list")

            for item in entities:
                if not isinstance(item, dict):
                    raise ValueError("each entity must be an object")
                etype = item.get("type")
                value = item.get("value")
                if etype is None or value is None:
                    raise ValueError("entity missing type or value")
                if etype == "amount":
                    amount_value = float(value)
                    if amount_value <= 0:
                        raise ValueError("amount must be positive")
                elif etype == "date":
                    date.fromisoformat(str(value))
        except Exception as exc:
            return False, {"error": str(exc)}

        return True, {}
