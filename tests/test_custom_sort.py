from search_service.core.query_builder import QueryBuilder
from search_service.models.request import SearchRequest


def test_custom_sort_is_used_in_query():
    qb = QueryBuilder()
    custom_sort = [{"amount": {"order": "asc"}}]
    req = SearchRequest(user_id=1, sort=custom_sort)
    query = qb.build_query(req)
    assert query["sort"] == custom_sort


def test_custom_sort_skips_default_sort(monkeypatch):
    qb = QueryBuilder()
    custom_sort = [{"amount": {"order": "asc"}}]
    called = False

    def fake_build_sort(request):
        nonlocal called
        called = True
        return []

    monkeypatch.setattr(qb, "_build_sort_criteria", fake_build_sort)
    req = SearchRequest(user_id=1, sort=custom_sort)
    query = qb.build_query(req)

    assert query["sort"] == custom_sort
    assert called is False
