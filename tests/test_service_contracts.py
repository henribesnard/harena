from conversation_service.models.service_contracts import (
    SearchServiceQuery,
    QueryMetadata,
    SearchParameters,
    SearchFilters,
)


def test_to_search_request_minimal():
    query = SearchServiceQuery(
        query_metadata=QueryMetadata(
            conversation_id="conv1",
            user_id=123,
            intent_type="TEST_INTENT",
            source_agent="unit_test",
        ),
        search_parameters=SearchParameters(),
        filters=SearchFilters(),
    )

    expected = {
        "user_id": 123,
        "query": "",
        "filters": {},
        "limit": 50,
        "offset": 0,
        "metadata": {
            "conversation_id": "conv1",
            "intent_type": "TEST_INTENT",
            "source_agent": "unit_test",
        },
    }

    assert query.to_search_request() == expected
