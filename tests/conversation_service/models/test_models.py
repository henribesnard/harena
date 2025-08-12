import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[3]))

import pytest
from datetime import datetime

from conversation_service.models.agent_models import AgentConfig, AgentResponse
from conversation_service.models.financial_models import (
    IntentResult,
    FinancialEntity,
    EntityType,
    IntentCategory,
    DetectionMethod,
)
from conversation_service.models.service_contracts import (
    QueryMetadata,
    SearchParameters,
    SearchFilters,
    SearchServiceQuery,
)


def test_agent_config_defaults_and_dict():
    config = AgentConfig(
        name="test_agent",
        model_client_config={
            "model": "deepseek-chat",
            "api_key": "sk",
            "base_url": "https://api.deepseek.com",
        },
        system_message="You are a test agent.",
    )

    assert config.max_consecutive_auto_reply == 3
    assert config.agent_type == "AssistantAgent"
    assert config.temperature == 0.1
    assert config.max_tokens == 1000
    assert config.timeout_seconds == 30
    assert config.retry_attempts == 2

    data = config.model_dump()
    assert data["name"] == "test_agent"
    assert data["model_client_config"]["model"] == "deepseek-chat"


def test_agent_response_defaults_and_dict():
    resp = AgentResponse(
        agent_name="agent",
        content="ok",
        execution_time_ms=10.5,
    )

    assert resp.success is True
    assert resp.error_message is None
    assert resp.metadata == {}
    assert isinstance(resp.timestamp, datetime)

    data = resp.model_dump()
    assert data["agent_name"] == "agent"
    assert data["content"] == "ok"


def test_agent_response_error_requires_message():
    with pytest.raises(ValueError):
        AgentResponse(
            agent_name="agent",
            content="fail",
            execution_time_ms=1.0,
            success=False,
        )


def test_intent_result_defaults_and_dict():
    entity = FinancialEntity(
        entity_type=EntityType.AMOUNT,
        raw_value="10€",
        normalized_value=10,
        confidence=0.9,
    )

    result = IntentResult(
        intent_type="TRANSACTION_SEARCH_BY_DATE",
        intent_category=IntentCategory.TRANSACTION_SEARCH,
        confidence=0.95,
        entities=[entity],
        method=DetectionMethod.LLM_BASED,
        processing_time_ms=5.2,
    )

    assert result.requires_clarification is False
    assert result.search_required is True

    data = result.model_dump()
    assert data["intent_type"] == "TRANSACTION_SEARCH_BY_DATE"
    assert data["entities"][0]["raw_value"] == "10€"


def test_search_service_query_defaults_and_dict():
    metadata = QueryMetadata(
        conversation_id="conv1",
        user_id=123,
        intent_type="TRANSACTION_SEARCH",
    )
    params = SearchParameters(search_text="test", max_results=10)
    filters = SearchFilters()
    query = SearchServiceQuery(
        query_metadata=metadata,
        search_parameters=params,
        filters=filters,
    )

    assert query.query_metadata.language == "fr"
    assert query.search_parameters.offset == 0
    assert query.aggregations is None

    data = query.model_dump()
    assert data["query_metadata"]["conversation_id"] == "conv1"
    assert data["search_parameters"]["max_results"] == 10
    assert data["filters"] == SearchFilters().model_dump()

    search_req = query.to_search_request()
    assert search_req["user_id"] == 123
    assert search_req["filters"] == {}
    assert search_req["limit"] == 10
