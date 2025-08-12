import pytest

from conversation_service.models.financial_models import (
    FinancialEntity,
    IntentResult,
    EntityType,
    IntentCategory,
    DetectionMethod,
)


def test_enumerations_values():
    assert EntityType.AMOUNT.value == "AMOUNT"
    assert IntentCategory.TRANSACTION_SEARCH.value == "TRANSACTION_SEARCH"
    assert DetectionMethod.LLM_BASED.value == "llm_based"


def test_financial_entity_and_validation():
    entity = FinancialEntity(
        entity_type=EntityType.AMOUNT,
        raw_value="10€",
        normalized_value=10,
        confidence=0.9,
    )

    assert entity.detection_method == DetectionMethod.HYBRID
    assert entity.to_search_filter() == {"field": "amount", "value": 10}

    with pytest.raises(ValueError):
        FinancialEntity(
            entity_type=EntityType.AMOUNT,
            raw_value="10€",
            normalized_value=10,
            confidence=0.9,
            start_position=5,
            end_position=3,
        )


def test_intent_result_defaults_and_entities():
    entity = FinancialEntity(
        entity_type=EntityType.AMOUNT,
        raw_value="10€",
        normalized_value=10,
        confidence=0.9,
    )

    result = IntentResult(
        intent_type="TRANSACTION_SEARCH_BY_AMOUNT",
        intent_category=IntentCategory.TRANSACTION_SEARCH,
        confidence=0.95,
        entities=[entity],
        method=DetectionMethod.LLM_BASED,
        processing_time_ms=12.3,
    )

    assert result.requires_clarification is False
    assert result.search_required is True
    assert result.has_entity_type(EntityType.AMOUNT)
    assert result.get_entities_by_type(EntityType.AMOUNT) == [entity]
