from conversation_service.intent_rules import create_rule_engine
from conversation_service.agents.hybrid_intent_agent import HybridIntentAgent
from conversation_service.models.financial_models import EntityType


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

    # Convert to FinancialEntity via HybridIntentAgent without full initialization
    agent = HybridIntentAgent.__new__(HybridIntentAgent)
    entities = agent._convert_rule_entities(match.entities)
    assert any(
        e.entity_type == EntityType.MERCHANT and e.normalized_value.get("merchant") == "NETFLIX"
        for e in entities
    )

