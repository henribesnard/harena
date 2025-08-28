"""Tests for the Autogen-based IntentClassifierAgent."""

import asyncio

import pytest

from conversation_service.agents.financial.intent_classifier import (
    IntentClassifierAgent,
)
from conversation_service.prompts.autogen.intent_classification_prompts import (
    AUTOGEN_INTENT_SYSTEM_MESSAGE,
)


def test_intent_classifier_agent_configuration():
    agent = IntentClassifierAgent()
    assert agent.name == "intent_classifier"
    assert agent.system_message == AUTOGEN_INTENT_SYSTEM_MESSAGE
    assert agent.max_consecutive_auto_reply == 1
    assert agent.llm_config == {
        "config_list": [
            {
                "model": "deepseek-chat",
                "response_format": {"type": "json_object"},
                "temperature": 0.1,
                "max_tokens": 800,
                "cache_seed": 42,
            }
        ]
    }


@pytest.mark.asyncio
async def test_classify_for_team_success(monkeypatch):
    agent = IntentClassifierAgent()

    async def fake_reply(message):
        return '{"intent_type": "TRANSACTION_SEARCH", "confidence": 0.9}'

    monkeypatch.setattr(agent, "a_generate_reply", fake_reply, raising=False)

    result = await agent.classify_for_team("list my transactions", 42)

    assert result["intent_type"] == "TRANSACTION_SEARCH"
    assert result["team_context"]["original_message"] == "list my transactions"
    assert result["team_context"]["user_id"] == 42
    assert result["team_context"]["ready_for_entity_extraction"] is True
    assert (
        result["team_context"]["suggested_entities_focus"]
        == agent.suggest_entities_focus("TRANSACTION_SEARCH", 0.9)
    )
    assert agent.success_count == 1
    assert agent.error_count == 0

    # Subsequent call should use cache
    cached = await agent.classify_for_team("list my transactions", 42)
    assert cached == result
    assert agent.success_count == 2


@pytest.mark.asyncio
async def test_classify_for_team_timeout(monkeypatch):
    agent = IntentClassifierAgent()

    async def fake_reply(message):  # simulate long processing
        await asyncio.sleep(35)
        return '{"intent_type": "TRANSACTION_SEARCH", "confidence": 0.9}'

    monkeypatch.setattr(agent, "a_generate_reply", fake_reply, raising=False)

    result = await agent.classify_for_team("slow request", 1)
    assert result["intent_type"] == "UNKNOWN"
    assert result["team_context"]["ready_for_entity_extraction"] is False
    assert agent.error_count == 1


def test_suggest_entities_focus_mapping():
    agent = IntentClassifierAgent()
    res = agent.suggest_entities_focus("BALANCE_INQUIRY", 0.9)
    assert res["priority_entities"] == ["account"]
    assert res["strategy"] == "account_lookup"
