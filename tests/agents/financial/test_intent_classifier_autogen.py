import asyncio
import json
import random

import pytest

from conversation_service.agents.financial.intent_classifier import (
    IntentClassifierAgent,
)


@pytest.mark.asyncio
async def test_classification_parity_with_phase1(monkeypatch):
    """Ensure classifications match Phase 1 expectations using a mocked DeepSeek reply."""
    agent = IntentClassifierAgent()

    async def fake_deepseek(message: str):
        return json.dumps({"intent": "TRANSACTION_SEARCH", "confidence": 0.85})

    monkeypatch.setattr(agent, "a_generate_reply", fake_deepseek, raising=False)

    result = await agent.classify_for_team("list my transactions", user_id=123)

    assert result["intent"] == "TRANSACTION_SEARCH"
    assert result["confidence"] == 0.85


@pytest.mark.asyncio
async def test_team_context_presence_and_coherence(monkeypatch):
    """Verify that team_context is populated with consistent information."""
    agent = IntentClassifierAgent()

    async def fake_reply(message: str):
        return json.dumps({"intent": "BALANCE_INQUIRY", "confidence": 0.6})

    monkeypatch.setattr(agent, "a_generate_reply", fake_reply, raising=False)

    result = await agent.classify_for_team("what is my balance", user_id=5)
    team_ctx = result["team_context"]

    assert team_ctx["original_message"] == "what is my balance"
    assert team_ctx["user_id"] == 5
    assert team_ctx["ready_for_entity_extraction"] is True
    assert team_ctx["suggested_entities_focus"] == agent.suggest_entities_focus(
        "BALANCE_INQUIRY", 0.6
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["malformed", "timeout"])
async def test_fallback_malformed_json_and_timeout(monkeypatch, mode):
    """The agent should return a fallback response on JSON errors or timeouts."""
    agent = IntentClassifierAgent()

    if mode == "malformed":
        async def fake_reply(message: str):
            return "not-json"
    else:
        async def fake_reply(message: str):
            raise asyncio.TimeoutError()

    monkeypatch.setattr(agent, "a_generate_reply", fake_reply, raising=False)

    result = await agent.classify_for_team("trigger error", user_id=0)
    assert result["intent"] == "GENERAL_INQUIRY"
    assert result["confidence"] == 0.3
    assert result["team_context"]["ready_for_entity_extraction"] is False


@pytest.mark.asyncio
async def test_autogen_cache_seed_deterministic_responses(monkeypatch):
    """AutoGen cache via cache_seed should produce identical replies on repeats."""
    agent = IntentClassifierAgent()

    async def deterministic_reply(message: str):
        seed = agent.llm_config["config_list"][0]["cache_seed"]
        rnd = random.Random(seed + hash(message))
        return json.dumps({"intent": "ECHO", "confidence": rnd.random()})

    monkeypatch.setattr(agent, "a_generate_reply", deterministic_reply, raising=False)

    first = await agent.classify_for_team("repeatable", user_id=1)
    agent.intent_cache.clear()
    second = await agent.classify_for_team("repeatable", user_id=1)

    assert first == second


@pytest.mark.asyncio
async def test_round_robin_group_chat_integration(monkeypatch):
    """The agent integrates with a RoundRobinGroupChat-like orchestrator."""
    agent = IntentClassifierAgent()

    async def fake_reply(message: str):
        return json.dumps({"intent": "GREETING", "confidence": 0.9})

    monkeypatch.setattr(agent, "a_generate_reply", fake_reply, raising=False)

    class DummyRoundRobinGroupChat:
        def __init__(self, agents):
            self.agents = agents

        async def a_run(self, message: str, user_id: int):
            results = []
            for ag in self.agents:
                results.append(await ag.classify_for_team(message, user_id))
            return results

    chat = DummyRoundRobinGroupChat([agent])
    responses = await chat.a_run("hello", user_id=77)

    assert responses[0]["intent"] == "GREETING"
    assert responses[0]["team_context"]["user_id"] == 77
