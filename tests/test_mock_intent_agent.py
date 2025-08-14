import asyncio
import pytest

import conversation_service.agents.base_financial_agent as base_financial_agent

base_financial_agent.AUTOGEN_AVAILABLE = True

from conversation_service.agents.mock_intent_agent import (
    MockIntentAgent,
    MOCK_INTENT_RESPONSES,
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

    for exp_entity, entity in zip(expected["entities"], intent_result.entities):
        assert entity.entity_type.value == exp_entity["entity_type"]
        assert entity.raw_value == exp_entity["raw_value"]
        assert entity.normalized_value == exp_entity["normalized_value"]
        assert abs(entity.confidence - exp_entity["confidence"]) < 1e-6
        if "position" in exp_entity:
            assert entity.start_position == exp_entity["position"][0]
            assert entity.end_position == exp_entity["position"][1]
