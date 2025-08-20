import asyncio
import json
import os
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
    base_url = "https://api.openai.com/v1"


class DummyOpenAIClient:
    def __init__(self, content: str):
        self._content = content

        class _Completions:
            async def create(_self, *args, **kwargs):
                class Choice:
                    message = type("Msg", (), {"content": content})

                return type("Resp", (), {"choices": [Choice()]})

        class _Chat:
            def __init__(self):
                self.completions = _Completions()

        self.chat = _Chat()


class ErrorOpenAIClient(DummyOpenAIClient):
    def __init__(self):
        class _Completions:
            async def create(_self, *args, **kwargs):
                raise RuntimeError("LLM failure")

        class _Chat:
            def __init__(self):
                self.completions = _Completions()

        self.chat = _Chat()


class FallbackAgent:
    async def detect_intent(self, user_message: str, user_id: int):
        intent_result = IntentResult(
            intent_type="FALLBACK_INTENT",
            intent_category=IntentCategory.GENERAL_QUESTION,
            confidence=1.0,
            entities=[],
            method=DetectionMethod.FALLBACK,
            processing_time_ms=0.0,
        )
        return {
            "content": "{}",
            "metadata": {
                "intent_result": intent_result,
                "detection_method": DetectionMethod.FALLBACK,
                "confidence": intent_result.confidence,
                "intent_type": intent_result.intent_type,
                "entities": [],
            },
            "confidence_score": intent_result.confidence,
        }


def test_full_flow_detects_intent_and_latency():
    os.environ["OPENAI_API_KEY"] = "openai-test-key"
    openai_client = DummyOpenAIClient(
        json.dumps(
                {
                    "intent_type": "SEARCH_BY_MERCHANT",
                    "intent_category": "TRANSACTION_SEARCH",
                    "confidence": 0.9,
                    "entities": [{"entity_type": "MERCHANT", "value": "Netflix", "confidence": 0.9}],
                }
            )
    )
    agent = EnhancedLLMIntentAgent(
        deepseek_client=DummyDeepSeekClient(), openai_client=openai_client
    )
    assert agent.config.model_client_config["api_key"] == "openai-test-key"
    result = asyncio.run(agent.detect_intent("Combien j’ai dépensé pour Netflix ?", 1))
    intent_result = result["metadata"]["intent_result"]
    assert intent_result.intent_type == "SEARCH_BY_MERCHANT"
    assert intent_result.processing_time_ms > 0


def test_fallback_when_llm_errors():
    os.environ["OPENAI_API_KEY"] = "openai-test-key"
    agent = EnhancedLLMIntentAgent(
        deepseek_client=DummyDeepSeekClient(),
        openai_client=ErrorOpenAIClient(),
        fallback_agent=FallbackAgent(),
    )
    assert agent.config.model_client_config["api_key"] == "openai-test-key"
    result = asyncio.run(agent.detect_intent("Salut", 1))
    intent_result = result["metadata"]["intent_result"]
    assert intent_result.intent_type == "FALLBACK_INTENT"
    assert intent_result.method == DetectionMethod.FALLBACK


def test_latency_measurement(monkeypatch):
    os.environ["OPENAI_API_KEY"] = "openai-test-key"
    import conversation_service.agents.enhanced_llm_intent_agent as ela

    calls = {"count": 0}

    def fake_perf_counter():
        calls["count"] += 1
        return 100.0 if calls["count"] == 1 else 100.123

    monkeypatch.setattr(ela.time, "perf_counter", fake_perf_counter)
    openai_client = DummyOpenAIClient(
        json.dumps(
            {
                "intent_type": "SEARCH_BY_MERCHANT",
                "intent_category": "TRANSACTION_SEARCH",
                "confidence": 0.9,
                "entities": [],
            }
        )
    )
    agent = EnhancedLLMIntentAgent(
        deepseek_client=DummyDeepSeekClient(), openai_client=openai_client
    )
    assert agent.config.model_client_config["api_key"] == "openai-test-key"
    result = asyncio.run(agent.detect_intent("ping", 1))
    intent_result = result["metadata"]["intent_result"]
    assert intent_result.processing_time_ms == pytest.approx(123.0, abs=1e-6)


def test_matches_old_llm_intent_agent_output():
    os.environ["OPENAI_API_KEY"] = "openai-test-key"
    openai_client1 = DummyOpenAIClient(
        json.dumps(
            {
                "intent_type": "SEARCH_BY_MERCHANT",
                "intent_category": "TRANSACTION_SEARCH",
                "confidence": 0.9,
                "entities": [],
            }
        )
    )
    openai_client2 = DummyOpenAIClient(
        json.dumps(
            {
                "intent_type": "SEARCH_BY_MERCHANT",
                "intent_category": "TRANSACTION_SEARCH",
                "confidence": 0.9,
                "entities": [],
            }
        )
    )
    enhanced = EnhancedLLMIntentAgent(
        deepseek_client=DummyDeepSeekClient(), openai_client=openai_client1
    )
    legacy = LLMIntentAgent(
        deepseek_client=DummyDeepSeekClient(), openai_client=openai_client2
    )
    assert enhanced.config.model_client_config["api_key"] == "openai-test-key"
    assert legacy.config.model_client_config["api_key"] == "openai-test-key"
    res_new = asyncio.run(enhanced.detect_intent("Combien ?", 1))
    res_old = asyncio.run(legacy.detect_intent("Combien ?", 1))
    assert (
        res_new["metadata"]["intent_result"].intent_type
        == res_old["metadata"]["intent_result"].intent_type
    )
