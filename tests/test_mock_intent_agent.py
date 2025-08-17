import asyncio
import pytest

import conversation_service.agents.base_financial_agent as base_financial_agent

base_financial_agent.AUTOGEN_AVAILABLE = True

from conversation_service.agents.mock_intent_agent import (
    MockIntentAgent,
    MOCK_INTENT_RESPONSES,
)
from conversation_service.models.financial_models import (
    DetectionMethod,
    IntentCategory,
    IntentResult,
)


@pytest.mark.parametrize("question,expected", list(MOCK_INTENT_RESPONSES.items()))
def test_mock_intent_agent_returns_predefined_results(question, expected):
    agent = MockIntentAgent()
    result = asyncio.run(agent.detect_intent(question, user_id=1))
    intent_result = result["metadata"]["intent_result"]

    assert intent_result.intent_type == expected["intent_type"]
    assert intent_result.intent_category.value == expected["intent_category"]
    assert abs(intent_result.confidence - expected["confidence"]) < 1e-6
    assert len(intent_result.entities) == len(expected["entities"])
    assert intent_result.method == DetectionMethod.RULE_BASED

    for exp_entity, entity in zip(expected["entities"], intent_result.entities):
        assert entity.entity_type.value == exp_entity["entity_type"]
        assert entity.raw_value == exp_entity["raw_value"]
        assert entity.normalized_value == exp_entity["normalized_value"]
        assert abs(entity.confidence - exp_entity["confidence"]) < 1e-6
        assert entity.detection_method == DetectionMethod.RULE_BASED
        if "position" in exp_entity:
            assert entity.start_position == exp_entity["position"][0]
            assert entity.end_position == exp_entity["position"][1]


def test_detect_intent_returns_intent_result_for_known_question():
    agent = MockIntentAgent()
    result = asyncio.run(agent.detect_intent("Mes transactions Netflix ce mois", user_id=1))
    intent_result = result["metadata"]["intent_result"]
    assert isinstance(intent_result, IntentResult)
    assert intent_result.method == DetectionMethod.RULE_BASED
    assert result["metadata"]["detection_method"] == DetectionMethod.RULE_BASED


def test_conversational_intents_are_supported():
    agent = MockIntentAgent()

    greeting = asyncio.run(agent.detect_intent("Bonjour, comment ça va ?", user_id=1))
    greeting_result = greeting["metadata"]["intent_result"]
    assert greeting_result.intent_category == IntentCategory.GREETING

    confirmation = asyncio.run(agent.detect_intent("Merci pour l'information", user_id=1))
    confirmation_result = confirmation["metadata"]["intent_result"]
    assert confirmation_result.intent_category == IntentCategory.CONFIRMATION


def test_payment_request_returns_unclear_intent():
    agent = MockIntentAgent()
    result = asyncio.run(agent.detect_intent("Peux-tu transférer 500 euros à Marie ?", user_id=1))
    intent_result = result["metadata"]["intent_result"]
    assert intent_result.intent_category == IntentCategory.UNCLEAR_INTENT
