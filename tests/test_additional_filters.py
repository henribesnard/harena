import pytest
from search_service.core.query_builder import QueryBuilder


def test_additional_filters_exists_wildcard_prefix():
    qb = QueryBuilder()
    filters = {
        "merchant_name": {"wildcard": "Ama*"},
        "category_name": {"exists": True},
        "account_name": {"prefix": "Main"},
    }
    result = qb._build_additional_filters(filters)
    assert {"wildcard": {"merchant_name.keyword": {"value": "Ama*"}}} in result
    assert {"exists": {"field": "category_name"}} in result
    assert {"prefix": {"account_name.keyword": "Main"}} in result


def test_unknown_filter_raises_error():
    qb = QueryBuilder()
    with pytest.raises(ValueError):
        qb._build_additional_filters({"merchant_name": {"foo": "bar"}})
