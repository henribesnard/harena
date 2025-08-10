from conversation_service.intent_rules.rule_loader import RuleLoader
from conversation_service.intent_rules.rule_engine import create_rule_engine


def test_loader_parses_new_fields():
    loader = RuleLoader()
    greeting_rule = loader.get_conversational_rules().get("GREETING")
    assert greeting_rule is not None
    assert greeting_rule.no_search_needed is True
    assert greeting_rule.suggested_responses
    assert isinstance(greeting_rule.suggested_responses, list)


def test_rule_match_includes_new_fields():
    loader = RuleLoader()
    engine = create_rule_engine(loader)
    match = engine.match_intent("bonjour")
    assert match is not None
    greeting_rule = loader.get_conversational_rules()["GREETING"]
    assert match.no_search_needed == greeting_rule.no_search_needed
    assert match.suggested_responses == greeting_rule.suggested_responses
