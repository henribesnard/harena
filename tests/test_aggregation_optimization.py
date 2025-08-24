import pytest
from unittest.mock import patch

from search_service.core.search_engine import SearchEngine
from search_service.models.request import SearchRequest


def _make_hit(idx: int) -> dict:
    return {
        "_source": {
            "transaction_id": f"tx_{idx}",
            "user_id": 1,
            "amount": float(idx),
            "amount_abs": float(idx),
            "currency_code": "EUR",
            "transaction_type": "debit",
            "date": "2024-01-01",
            "primary_description": f"desc {idx}",
        },
        "_score": 1.0,
    }


def test_aggregation_only_returns_no_results_but_same_aggregations():
    async def _run():
        engine = SearchEngine()
        engine.elasticsearch_client = object()
        engine.cache_enabled = False

        aggs = {"operation_type": {"terms": {"field": "operation_type"}}}

        base_request = SearchRequest(user_id=1, query="", limit=5, aggregations=aggs, filters={}, metadata={})
        agg_only_request = SearchRequest(
            user_id=1, query="", limit=5, aggregations=aggs, aggregation_only=True, filters={}, metadata={}
        )

        hits = [_make_hit(i) for i in range(2)]

        base_response = {
            "hits": {"hits": hits, "total": {"value": len(hits)}},
            "aggregations": {"operation_type": {"buckets": []}},
            "took": 1,
        }
        agg_response = {
            "hits": {"hits": [], "total": {"value": len(hits)}},
            "aggregations": {"operation_type": {"buckets": []}},
            "took": 1,
        }

        async def fake_exec(es_query, request):
            return agg_response if request.aggregation_only else base_response

        with patch.object(engine, "_execute_search", side_effect=fake_exec):
            full_resp = await engine.search(base_request)
            agg_resp = await engine.search(agg_only_request)

        assert agg_resp["results"] == []
        assert full_resp["aggregations"] == agg_resp["aggregations"]

    import asyncio

    asyncio.run(_run())
