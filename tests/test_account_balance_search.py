from search_service.core.query_builder import QueryBuilder
from search_service.models.request import SearchRequest


def test_search_fields_do_not_include_account_balance():
    qb = QueryBuilder()
    assert "account_balance" not in qb.search_fields


def test_numeric_query_builds_account_balance_filter():
    qb = QueryBuilder()
    req = SearchRequest(user_id=1, query="123.45")
    query = qb.build_query(req)
    must_filters = query["query"]["bool"]["must"]
    assert {"range": {"account_balance": {"gte": 123.45, "lte": 123.45}}} in must_filters
