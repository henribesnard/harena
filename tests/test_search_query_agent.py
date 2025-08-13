from conversation_service.agents import base_financial_agent

# Ensure the base agent does not require AutoGen during tests
base_financial_agent.AUTOGEN_AVAILABLE = True

from conversation_service.agents.search_query_agent import SearchQueryAgent
from conversation_service.models.financial_models import (
    FinancialEntity,
    EntityType,
    IntentResult,
    IntentCategory,
    DetectionMethod,
)
import asyncio


class DummyDeepSeekClient:
    api_key = "test-key"
    base_url = "http://api.example.com"


def test_prepare_entity_context_with_string_entity_type():
    agent = SearchQueryAgent(
        deepseek_client=DummyDeepSeekClient(),
        search_service_url="http://search.example.com",
    )
    entity = FinancialEntity(
        entity_type=EntityType.MERCHANT,
        raw_value="Starbucks",
        normalized_value="Starbucks",
        confidence=0.9,
    )
    # Simulate an entity where entity_type is a plain string
    entity.entity_type = "MERCHANT"
    context = agent._prepare_entity_extraction_context("message", [entity])
    assert "MERCHANT" in context


def test_generate_search_contract_deduplicates_terms():
    agent = SearchQueryAgent(
        deepseek_client=DummyDeepSeekClient(),
        search_service_url="http://search.example.com",
    )
    intent_result = IntentResult(
        intent_type="TRANSACTION_SEARCH",
        intent_category=IntentCategory.TRANSACTION_SEARCH,
        confidence=0.9,
        entities=[
            FinancialEntity(
                entity_type=EntityType.MERCHANT,
                raw_value="Carrefour",
                normalized_value="carrefour",
                confidence=0.8,
            )
        ],
        method=DetectionMethod.RULE_BASED,
        processing_time_ms=1.0,
    )

    search_query = asyncio.run(
        agent._generate_search_contract(intent_result, "Carrefour", user_id=1)
    )

    request = search_query.to_search_request()
    assert request["query"].split().count("carrefour") == 1
    assert request["filters"].get("merchants") == ["carrefour"]
