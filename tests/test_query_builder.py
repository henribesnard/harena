import pytest

try:
    from search_service.core.query_builder import QueryBuilder
    from search_service.models.request import SearchRequest
except Exception:  # pragma: no cover - skip if dependencies missing
    QueryBuilder = None
    SearchRequest = None


@pytest.mark.skipif(QueryBuilder is None, reason="search_service not available")
def test_query_builder_includes_user_id_filter():
    builder = QueryBuilder()
    request = SearchRequest(user_id=42, query="coffee")
    query = builder.build_query(request)
    assert {"term": {"user_id": 42}} in query["query"]["bool"]["must"]

