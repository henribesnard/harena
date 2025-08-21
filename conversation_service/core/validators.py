"""Common Pydantic validators and business rule helpers for agents."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from pydantic import field_validator, ValidationError

from ..models.enums import IntentType
from ..models.conversation_models import IntentResult

__all__ = ["CurrencyAmountValidators", "HarenaValidators"]


class CurrencyAmountValidators:
    """Reusable mixin providing currency and amount validators."""

    @field_validator("currency")
    @classmethod
    def validate_currency_code(cls, value: str) -> str:
        """Ensure currency codes follow ISO 4217."""
        if not re.fullmatch(r"[A-Z]{3}", value):
            raise ValueError("currency must be a 3-letter ISO code")
        return value

    @field_validator("amount")
    @classmethod
    def validate_positive_amount(cls, value: float) -> float:
        """Verify that monetary amounts are positive."""
        if value < 0:
            raise ValueError("amount must be positive")
        return value


class HarenaValidators:
    """Custom validators for Harena-specific business rules."""

    @staticmethod
    def validate_harena_scope(intent: str) -> bool:
        """Validate intent is within Harena's consultation scope."""
        try:
            intent_enum = IntentType(intent)
            result = IntentResult(
                intent=intent_enum,
                confidence=0.8,  # Dummy confidence for validation
                reasoning="Scope validation check",
            )
            return result.is_supported_by_harena()
        except (ValueError, ValidationError):
            return False

    @staticmethod
    def validate_french_iban(iban: str) -> bool:
        """Validate French IBAN format (basic check)."""
        if not iban:
            return False

        # Clean IBAN
        clean_iban = re.sub(r"[^A-Z0-9]", "", iban.upper())

        # Check French IBAN format: FR + 2 check digits + 23 chars
        if not re.match(r"^FR\d{25}$", clean_iban):
            return False

        return len(clean_iban) == 27

    @staticmethod
    def normalize_dynamic_amount(amount_str: str) -> Optional[str]:
        """Normalize monetary amount with dynamic parsing."""
        try:
            # Dynamic cleaning - remove all non-numeric except decimal separators and signs
            clean_amount = re.sub(r"[^\d.,-+]", "", amount_str.strip())

            # Handle various decimal separators dynamically
            if "," in clean_amount and "." in clean_amount:
                # Assume last separator is decimal point
                if clean_amount.rfind(",") > clean_amount.rfind("."):
                    clean_amount = clean_amount.replace(".", "").replace(",", ".")
                else:
                    clean_amount = clean_amount.replace(",", "")
            elif "," in clean_amount:
                # Could be thousands or decimal - keep as decimal point
                clean_amount = clean_amount.replace(",", ".")

            # Validate and format
            amount_float = float(clean_amount)
            if abs(amount_float) > 10000000:  # 10M safety limit
                return None

            return f"{amount_float:.2f}"
        except (ValueError, TypeError):
            return None

    @staticmethod
    def requires_clarification(
        user_message: str,
        intent: str,
        entities: List[Dict[str, Any]],
    ) -> Tuple[bool, List[str]]:
        """Determine if user message requires clarification."""
        clarification_needed = False
        questions: List[str] = []

        # Check message length - too short might be ambiguous
        if len(user_message.strip()) < 3:
            clarification_needed = True
            questions.append("Pouvez-vous préciser votre demande ?")
            return clarification_needed, questions

        # Check for vague pronouns without context
        vague_words = ["ça", "ce", "cette", "celui", "celle", "le truc", "la chose"]
        if any(word in user_message.lower() for word in vague_words):
            clarification_needed = True
            questions.append("À quoi faites-vous référence exactement ?")

        # Check intent-specific clarification needs
        if intent == IntentType.CATEGORY_ANALYSIS.value:
            if not any(e.get("type") == "CATEGORY" for e in entities):
                clarification_needed = True
                questions.append("Quelle catégorie de dépenses souhaitez-vous analyser ?")
        elif intent == IntentType.MERCHANT_ANALYSIS.value:
            if not any(e.get("type") == "MERCHANT" for e in entities):
                clarification_needed = True
                questions.append("Chez quel marchand souhaitez-vous voir vos dépenses ?")

        return clarification_needed, questions

    @staticmethod
    def validate_conversation_flow(
        intent: str,
        entities: List[Dict[str, Any]],
        stage: str,
    ) -> List[str]:
        """Validate conversation flow consistency."""
        warnings: List[str] = []

        # Check entity-intent consistency
        entity_types = [e.get("type") for e in entities]

        # Analysis intents should have relevant entities
        if intent == "CATEGORY_ANALYSIS" and "CATEGORY" not in entity_types:
            warnings.append("Category analysis should have category entity")

        if intent == "MERCHANT_ANALYSIS" and "MERCHANT" not in entity_types:
            warnings.append("Merchant analysis should have merchant entity")

        if intent == "TEMPORAL_ANALYSIS" and "DATE_RANGE" not in entity_types:
            warnings.append("Temporal analysis should have date range entity")

        # Check for action entities in consultation intents
        action_entities = {"BENEFICIARY", "AUTHENTICATION", "PAYMENT_REFERENCE"}
        consultation_intents = {
            "BALANCE_INQUIRY",
            "SPENDING_ANALYSIS",
            "CATEGORY_ANALYSIS",
            "MERCHANT_ANALYSIS",
            "TRANSACTION_SEARCH",
        }

        if intent in consultation_intents:
            found_action = [e for e in entity_types if e in action_entities]
            if found_action:
                warnings.append(
                    f"Action entities in consultation intent: {found_action}"
                )

        # Check for unsupported actions at response stage
        unsupported_actions = {
            "TRANSFER_REQUEST",
            "PAYMENT_REQUEST",
            "CARD_OPERATIONS",
            "LOAN_REQUEST",
            "ACCOUNT_MODIFICATION",
            "INVESTMENT_OPERATIONS",
        }

        if intent in unsupported_actions and stage != "response":
            warnings.append(
                f"Unsupported action {intent} should be handled at response stage"
            )

        return warnings

    @staticmethod
    def suggest_clarification_questions(
        intent: str, entities: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate helpful clarification questions based on intent and entities."""
        questions: List[str] = []

        if intent == IntentType.INSUFFICIENT_CONTEXT.value:
            questions.extend(
                [
                    "Que souhaitez-vous savoir sur vos finances ?",
                    "Voulez-vous consulter vos comptes, analyser vos dépenses, ou autre chose ?",
                    "Pouvez-vous préciser votre demande ?",
                ]
            )
        elif intent == IntentType.AMBIGUOUS.value:
            questions.extend(
                [
                    "Pouvez-vous reformuler votre demande ?",
                    "Souhaitez-vous voir des transactions ou faire une analyse ?",
                    "Quelle information cherchez-vous exactement ?",
                ]
            )
        elif intent == IntentType.UNKNOWN.value:
            questions.extend(
                [
                    "Je peux vous aider avec vos finances. Que voulez-vous savoir ?",
                    "Souhaitez-vous voir votre solde, vos transactions, ou analyser vos dépenses ?",
                    "Avez-vous une question sur un produit bancaire spécifique ?",
                ]
            )

        # Intent-specific clarification
        elif (
            intent == IntentType.CATEGORY_ANALYSIS.value
            and not any(e.get("type") == "CATEGORY" for e in entities)
        ):
            questions.extend(
                [
                    "Quelle catégorie voulez-vous analyser ? (restaurant, transport, shopping...)",
                    "Sur quelle période souhaitez-vous cette analyse ?",
                ]
            )
        elif (
            intent == IntentType.MERCHANT_ANALYSIS.value
            and not any(e.get("type") == "MERCHANT" for e in entities)
        ):
            questions.extend(
                [
                    "Chez quel marchand souhaitez-vous voir vos dépenses ?",
                    "Voulez-vous une analyse sur une période spécifique ?",
                ]
            )

        return questions[:3]  # Limit to 3 questions max
