from search_service.core.query_builder import QueryBuilder
from search_service.models.request import SearchRequest


def test_custom_sort_is_used_in_query():
    qb = QueryBuilder()
    custom_sort = [{"amount": {"order": "asc"}}]
    req = SearchRequest(user_id=1, sort=custom_sort)
    query = qb.build_query(req)
    assert query["sort"] == custom_sort
