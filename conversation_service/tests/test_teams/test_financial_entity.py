from conversation_service.models.core_models import FinancialEntity, EntityType


def test_is_action_related_true():
    entity = FinancialEntity(
        entity_type=EntityType.BENEFICIARY,
        raw_value="John",
        normalized_value="John",
        confidence=0.9,
    )
    assert entity.is_action_related() is True


def test_to_search_filter_amount():
    entity = FinancialEntity(
        entity_type=EntityType.AMOUNT,
        raw_value="100",
        normalized_value="100",
        confidence=0.9,
    )
    filt = entity.to_search_filter()
    assert filt["range"]["amount_abs"]["gte"] == 90.0
