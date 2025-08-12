import pytest
from datetime import datetime

from conversation_service.models.service_contracts import (
    QueryMetadata,
    SearchParameters,
    SearchFilters,
    SearchServiceQuery,
)


def test_query_metadata_defaults_and_validation():
    metadata = QueryMetadata(
        conversation_id="conv1",
        user_id=123,
        intent_type="TRANSACTION_SEARCH",
    )

    assert metadata.language == "fr"
    assert metadata.priority == "normal"
    assert metadata.retry_count == 0
    assert metadata.timeout_ms == 5000
    assert isinstance(metadata.query_id, str)
    assert isinstance(metadata.timestamp, datetime)

    with pytest.raises(ValueError):
        QueryMetadata(conversation_id="conv1", user_id=0, intent_type="X")


def test_search_parameters_defaults():
    params = SearchParameters()

    assert params.max_results == 50
    assert params.offset == 0
    assert params.sort_by == "relevance"
    assert params.sort_order == "desc"
    assert params.search_strategy == "hybrid"
    assert params.include_aggregations is False
    assert params.include_highlights is True


def test_search_filters_validation_and_dump():
    filters = SearchFilters(
        date_range={"start": "2024-01-01", "end": "2024-01-31"},
        amount_range={"min": 10},
    )

    assert filters.date_range["start"] == "2024-01-01"
    data = filters.model_dump()
    assert "date_range" in data

    with pytest.raises(ValueError):
        SearchFilters(date_range={"start": "2024-01-01"})

    with pytest.raises(ValueError):
        SearchFilters(amount_range={"min": 50, "max": 10})


def test_search_service_query_and_to_search_request():
    metadata = QueryMetadata(
        conversation_id="conv2",
        user_id=456,
        intent_type="TRANSACTION_SEARCH",
    )
    params = SearchParameters(search_text="test", max_results=10)
    filters = SearchFilters()

    query = SearchServiceQuery(
        query_metadata=metadata,
        search_parameters=params,
        filters=filters,
    )

    assert query.aggregations is None

    search_req = query.to_search_request()
    assert search_req["user_id"] == 456
    assert search_req["filters"] == {}
    assert search_req["limit"] == 10
    assert search_req["metadata"]["intent_type"] == "TRANSACTION_SEARCH"
