import asyncio
import pytest
from unittest.mock import patch

from search_service.core.search_engine import SearchEngine
from search_service.models.request import SearchRequest


def _make_hit(idx: int) -> dict:
    return {
        "_source": {
            "transaction_id": f"tx_{idx}",
            "user_id": 1,
            "account_id": 10 + idx,
            "account_name": f"Account {idx}",
            "account_type": "checking",
            "account_balance": 1000.0 + idx,
            "account_currency": "EUR",
            "amount": float(idx),
            "amount_abs": float(idx),
            "currency_code": "EUR",
            "transaction_type": "debit",
            "date": "2024-01-01",
            "primary_description": f"desc {idx}",
        },
        "_score": 1.0,
    }


@pytest.mark.asyncio
async def test_aggregation_only_returns_no_results_but_same_aggregations():
    engine = SearchEngine()
    engine.elasticsearch_client = object()
    engine.cache_enabled = False
    aggs = {"operation_type": {"terms": {"field": "operation_type"}}}
    base_request = SearchRequest(
        user_id=1, query="", page_size=5, aggregations=aggs, filters={}, metadata={}
    )
    agg_only_request = SearchRequest(
        user_id=1,
        query="",
        page_size=5,
        aggregations=aggs,
        aggregation_only=True,
        filters={},
        metadata={},
    )
    hits = [_make_hit(i) for i in range(2)]
    base_resp = {
        "hits": {"hits": hits, "total": {"value": len(hits)}},
        "aggregations": {"operation_type": {"buckets": []}},
        "took": 1,
    }
    agg_resp = {
        "hits": {"hits": [], "total": {"value": len(hits)}},
        "aggregations": {"operation_type": {"buckets": []}},
        "took": 1,
    }

    async def fake_exec(es_query, request):
        return agg_resp if request.aggregation_only else base_resp

    with patch.object(engine, "_execute_search", side_effect=fake_exec):
        full = await engine.search(base_request)
        agg_only = await engine.search(agg_only_request)

    assert agg_only["results"] == []
    assert full["aggregations"] == agg_only["aggregations"]
    first = full["results"][0]
    assert first["account_name"] == "Account 0"
    assert first["account_type"] == "checking"
    assert first["account_balance"] == 1000.0
    assert first["account_currency"] == "EUR"
    assert first["_score"] == 1.0
    assert "score" not in first
