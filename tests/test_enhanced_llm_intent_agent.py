import asyncio
import json

import pytest
import conversation_service.agents.base_financial_agent as base_financial_agent
base_financial_agent.AUTOGEN_AVAILABLE = True

from conversation_service.agents.enhanced_llm_intent_agent import (
    EnhancedLLMIntentAgent,
)
from conversation_service.agents.llm_intent_agent import LLMIntentAgent
from conversation_service.models.financial_models import (
    DetectionMethod,
    IntentCategory,
    IntentResult,
)


class DummyDeepSeekClient:
    api_key = "test-key"
    base_url = "http://api.example.com"

    async def generate_response(self, messages, temperature, max_tokens, user, use_cache):
        class Response:
            content = json.dumps(
                {
                    "intent": "SEARCH_BY_MERCHANT",
                    "confidence": 0.9,
                    "entities": [{"type": "MERCHANT", "value": "Netflix"}],
                }
            )

        return Response()


class ErrorDeepSeekClient(DummyDeepSeekClient):
    async def generate_response(self, *args, **kwargs):
        raise RuntimeError("LLM failure")


class FallbackAgent:
    async def detect_intent(self, user_message: str, user_id: int):
        intent_result = IntentResult(
            intent_type="FALLBACK_INTENT",
            intent_category=IntentCategory.GENERAL_QUESTION,
            confidence=1.0,
            entities=[],
            method=DetectionMethod.AI_FALLBACK,
            processing_time_ms=0.0,
        )
        return {
            "content": "{}",
            "metadata": {
                "intent_result": intent_result,
                "detection_method": DetectionMethod.AI_FALLBACK,
                "confidence": intent_result.confidence,
                "intent_type": intent_result.intent_type,
                "entities": [],
            },
            "confidence_score": intent_result.confidence,
        }


def test_full_flow_detects_intent_and_latency():
    agent = EnhancedLLMIntentAgent(deepseek_client=DummyDeepSeekClient())
    result = asyncio.run(agent.detect_intent("Combien j’ai dépensé pour Netflix ?", 1))
    intent_result = result["metadata"]["intent_result"]
    assert intent_result.intent_type == "SEARCH_BY_MERCHANT"
    assert intent_result.processing_time_ms > 0


def test_fallback_when_llm_errors():
    agent = EnhancedLLMIntentAgent(
        deepseek_client=ErrorDeepSeekClient(), fallback_agent=FallbackAgent()
    )
    result = asyncio.run(agent.detect_intent("Bonjour", 1))
    intent_result = result["metadata"]["intent_result"]
    assert intent_result.intent_type == "FALLBACK_INTENT"
    assert intent_result.method == DetectionMethod.AI_ERROR_FALLBACK


def test_latency_measurement(monkeypatch):
    import conversation_service.agents.enhanced_llm_intent_agent as ela

    calls = {"count": 0}

    def fake_perf_counter():
        calls["count"] += 1
        return 100.0 if calls["count"] == 1 else 100.123

    monkeypatch.setattr(ela.time, "perf_counter", fake_perf_counter)
    agent = EnhancedLLMIntentAgent(deepseek_client=DummyDeepSeekClient())
    result = asyncio.run(agent.detect_intent("ping", 1))
    intent_result = result["metadata"]["intent_result"]
    assert intent_result.processing_time_ms == pytest.approx(123.0, abs=1e-6)


def test_matches_old_llm_intent_agent_output():
    enhanced = EnhancedLLMIntentAgent(deepseek_client=DummyDeepSeekClient())
    legacy = LLMIntentAgent(deepseek_client=DummyDeepSeekClient())
    res_new = asyncio.run(enhanced.detect_intent("Combien ?", 1))
    res_old = asyncio.run(legacy.detect_intent("Combien ?", 1))
    assert (
        res_new["metadata"]["intent_result"].intent_type
        == res_old["metadata"]["intent_result"].intent_type
    )
