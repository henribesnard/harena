import pytest
from unittest.mock import AsyncMock, patch

from search_service.core.search_engine import SearchEngine
from search_service.models.request import SearchRequest


def test_cache_key_includes_aggregations_and_flag():
    engine = SearchEngine()
    base_req = SearchRequest(user_id=1, query="", filters={})
    base_key = engine._generate_cache_key(base_req)

    aggs_req = SearchRequest(
        user_id=1,
        query="",
        filters={},
        aggregations={"foo": {"terms": {"field": "bar"}}},
    )
    assert engine._generate_cache_key(aggs_req) != base_key

    flag_req = SearchRequest(user_id=1, query="", filters={}, aggregation_only=True)
    assert engine._generate_cache_key(flag_req) != base_key

    highlight_req = SearchRequest(
        user_id=1,
        query="",
        filters={},
        highlight={"fields": {"primary_description": {}}},
    )
    assert engine._generate_cache_key(highlight_req) != base_key


@pytest.mark.asyncio
async def test_cache_get_set_use_generated_key():
    engine = SearchEngine()
    engine.elasticsearch_client = object()

    req = SearchRequest(
        user_id=1,
        query="test",
        filters={},
        aggregations={"foo": {"terms": {"field": "bar"}}},
    )

    expected_key = engine._generate_cache_key(req)

    engine.cache.get = AsyncMock(return_value=None)
    engine.cache.set = AsyncMock()

    es_resp = {"hits": {"hits": [], "total": {"value": 0}}, "aggregations": {}, "took": 1}
    with patch.object(engine, "_execute_search", AsyncMock(return_value=es_resp)):
        await engine.search(req)

    engine.cache.get.assert_awaited_once()
    engine.cache.set.assert_awaited_once()
    assert engine.cache.get.await_args.args[1] == expected_key
    assert engine.cache.set.await_args.args[1] == expected_key

