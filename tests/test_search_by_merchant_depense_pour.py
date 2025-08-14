from conversation_service.intent_rules import create_rule_engine
from conversation_service.models.financial_models import (
    DetectionMethod,
    EntityType,
    FinancialEntity,
)


def test_depense_pour_netflix_matches_rule_and_entity():
    engine = create_rule_engine()
    message = "Combien j'ai dépensé pour Netflix ce mois ?"

    # Verify RuleMatch intent and normalized entity
    match = engine.match_intent(message, confidence_threshold=0.3)
    assert match is not None
    assert match.intent == "SEARCH_BY_MERCHANT"
    merchants = match.entities.get("merchant")
    assert merchants
    assert merchants[0].normalized_value.get("merchant") == "NETFLIX"

    # Convert rule-engine entities to FinancialEntity without HybridIntentAgent
    entities = []
    for entity_list in match.entities.values():
        for e in entity_list:
            try:
                entity_type = EntityType(e.entity_type.upper())
            except ValueError:
                continue
            entities.append(
                FinancialEntity(
                    entity_type=entity_type,
                    raw_value=e.raw_value,
                    normalized_value=e.normalized_value,
                    confidence=e.confidence,
                    start_position=e.position[0],
                    end_position=e.position[1],
                    detection_method=DetectionMethod.RULE_BASED,
                )
            )
    assert any(
        e.entity_type == EntityType.MERCHANT and e.normalized_value.get("merchant") == "NETFLIX"
        for e in entities
    )

