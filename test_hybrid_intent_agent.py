import pytest
import sys
import types
import asyncio

# Stub autogen so BaseFinancialAgent can import
autogen_stub = types.ModuleType("autogen")
class AssistantAgent:
    def __init__(self, *args, **kwargs):
        pass
autogen_stub.AssistantAgent = AssistantAgent
sys.modules["autogen"] = autogen_stub

# Stub httpx to satisfy indirect imports
sys.modules["httpx"] = types.ModuleType("httpx")

# Stub DeepSeekClient module to avoid heavy dependencies
deepseek_stub = types.ModuleType("deepseek_client")
class _StubDeepSeekClient:
    pass
deepseek_stub.DeepSeekClient = _StubDeepSeekClient
sys.modules["conversation_service.core.deepseek_client"] = deepseek_stub

from conversation_service.agents.hybrid_intent_agent import HybridIntentAgent
from conversation_service.models.financial_models import (
    IntentResult,
    IntentCategory,
    FinancialEntity,
)


class DummyDeepSeekClient:
    api_key = "test"
    base_url = "http://test"

    async def generate_response(self, *args, **kwargs):
        class Resp:
            content = "Intention: GENERAL\nConfiance: 0.5\n"
        return Resp()


def test_rule_based_detection_returns_intent_result():
    agent = HybridIntentAgent(deepseek_client=DummyDeepSeekClient())
    result = asyncio.run(agent.detect_intent("bonjour"))
    intent_result = result["metadata"]["intent_result"]
    assert isinstance(intent_result, IntentResult)
    assert isinstance(intent_result.intent_category, IntentCategory)
    assert isinstance(intent_result.entities, list)
    assert all(isinstance(e, FinancialEntity) for e in intent_result.entities)
    assert result["metadata"]["detection_method"] == "rules"


def test_ai_fallback_returns_intent_result():
    agent = HybridIntentAgent(deepseek_client=DummyDeepSeekClient())
    result = asyncio.run(agent.detect_intent("message sans correspondance"))
    intent_result = result["metadata"]["intent_result"]
    assert result["metadata"]["detection_method"] == "ai_fallback"
    assert isinstance(intent_result, IntentResult)
    assert isinstance(intent_result.intent_category, IntentCategory)
    assert isinstance(intent_result.entities, list)


def test_error_handling_returns_fallback(monkeypatch):
    agent = HybridIntentAgent(deepseek_client=DummyDeepSeekClient())

    async def broken_rule(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(agent, "_try_rule_based_detection", broken_rule)

    result = asyncio.run(agent.detect_intent("bonjour"))
    intent_result = result["metadata"]["intent_result"]
    assert result["metadata"]["detection_method"] == "fallback"
    assert isinstance(intent_result, IntentResult)
    assert intent_result.intent_type == "GENERAL"
    assert isinstance(intent_result.entities, list)
