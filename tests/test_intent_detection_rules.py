import pytest

from conversation_service.intent_rules import create_rule_engine


# Create a single rule engine instance for all tests
engine = create_rule_engine()


@pytest.mark.parametrize(
    "text, expected_intent",
    [
        ("recherche pizza", "SEARCH_BY_TEXT"),
        ("combien d'opérations ce mois", "COUNT_TRANSACTIONS"),
        ("nombre de mouvements ce mois", "COUNT_TRANSACTIONS"),
        ("tendance budget 2025", "ANALYZE_TRENDS"),
        ("bonjour", "GREETING"),
    ],
)
def test_rule_engine_detects_intents(text: str, expected_intent: str) -> None:
    """Verify that the rule engine matches user text to the correct intent."""
    match = engine.match_intent(text, confidence_threshold=0.3)
    assert match is not None, f"No intent detected for '{text}'"
    assert match.intent == expected_intent

