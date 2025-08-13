from conversation_service.agents import base_financial_agent

# Ensure the base agent does not require AutoGen during tests
base_financial_agent.AUTOGEN_AVAILABLE = True

from conversation_service.agents.search_query_agent import SearchQueryAgent
from conversation_service.models.financial_models import FinancialEntity, EntityType


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
