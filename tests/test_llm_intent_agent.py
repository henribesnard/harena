import asyncio
import os
import pytest
import conversation_service.agents.base_financial_agent as base_financial_agent
base_financial_agent.AUTOGEN_AVAILABLE = True

from conversation_service.agents.llm_intent_agent import LLMIntentAgent
from conversation_service.models.financial_models import EntityType, IntentCategory


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


def test_llm_intent_agent_parses_output_correctly():
    os.environ["OPENAI_API_KEY"] = "openai-test-key"
    openai_client = DummyOpenAIClient(
        '{"intent_type": "SEARCH_BY_MERCHANT", "intent_category": "TRANSACTION_SEARCH", "confidence": 0.77, "entities": [{"entity_type": "MERCHANT", "value": "Netflix", "confidence": 0.77}]}'
    )
    agent = LLMIntentAgent(
        deepseek_client=DummyDeepSeekClient(), openai_client=openai_client
    )
    assert agent.config.model_client_config["api_key"] == "openai-test-key"
    assert agent.config.model_client_config["model"] == "gpt-4o-mini"
    result = asyncio.run(
        agent.detect_intent("Combien j’ai dépensé pour Netflix ce mois ?", user_id=1)
    )
    intent_result = result["metadata"]["intent_result"]
    assert intent_result.intent_type == "SEARCH_BY_MERCHANT"
    assert intent_result.confidence == 0.77
    merchant = next(
        e for e in intent_result.entities if e.entity_type == EntityType.MERCHANT
    )
    assert merchant.normalized_value == "Netflix"
    assert merchant.confidence == 0.77


def test_category_mapping_applied():
    os.environ["OPENAI_API_KEY"] = "openai-test-key"
    openai_client = DummyOpenAIClient(
        '{"intent_type": "BALANCE_CHECK", "intent_category": "ACCOUNT_BALANCE", "confidence": 0.9, "entities": []}'
    )
    agent = LLMIntentAgent(
        deepseek_client=DummyDeepSeekClient(), openai_client=openai_client
    )
    assert agent.config.model_client_config["api_key"] == "openai-test-key"
    result = asyncio.run(agent.detect_intent("Quel est mon solde ?", user_id=1))
    intent_result = result["metadata"]["intent_result"]
    assert intent_result.intent_type == "BALANCE_CHECK"
    assert intent_result.intent_category == IntentCategory.BALANCE_INQUIRY


@pytest.mark.parametrize(
    "message, expected",
    [
        ("J'ai dépensé 20 euros", "debit"),
        ("Liste de mes sorties", "debit"),
        ("Une entrée d'argent inattendue", "credit"),
        ("Mes gains du mois", "credit"),
    ],
)
def test_transaction_type_post_processing(message, expected):
    os.environ["OPENAI_API_KEY"] = "openai-test-key"
    openai_client = DummyOpenAIClient(
        '{"intent_type": "TRANSACTION_SEARCH", "intent_category": "TRANSACTION_SEARCH", "confidence": 0.9, "entities": []}'
    )
    agent = LLMIntentAgent(
        deepseek_client=DummyDeepSeekClient(), openai_client=openai_client
    )
    result = asyncio.run(agent.detect_intent(message, user_id=1))
    intent_result = result["metadata"]["intent_result"]
    tx = next(e for e in intent_result.entities if e.entity_type == EntityType.TRANSACTION_TYPE)
    assert tx.normalized_value == expected
