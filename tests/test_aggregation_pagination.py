import asyncio
import pytest
from unittest.mock import AsyncMock, patch

from search_service.core.search_engine import SearchEngine
from search_service.models.request import SearchRequest


def _make_hit(idx: int) -> dict:
    return {
        "_source": {
            "transaction_id": f"tx_{idx}",
            "user_id": 1,
            "account_id": 20 + idx,
            "account_name": f"Account {idx}",
            "account_type": "checking",
            "account_balance": 2000.0 + idx,
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


def test_pagination_with_aggregations_returns_all_hits():
    async def _run():
        engine = SearchEngine()
        engine.elasticsearch_client = object()

        page_size = 3
        total_hits = 5
        hits_page1 = [_make_hit(i) for i in range(1, page_size + 1)]
        hits_page2 = [_make_hit(i) for i in range(page_size + 1, total_hits + 1)]

        es_resp1 = {
            "hits": {"hits": hits_page1, "total": {"value": total_hits}},
            "aggregations": {"dummy": {"buckets": []}},
            "took": 1,
        }
        es_resp2 = {
            "hits": {"hits": hits_page2, "total": {"value": total_hits}},
            "aggregations": {"dummy": {"buckets": []}},
            "took": 1,
        }

        mock_exec = AsyncMock(side_effect=[es_resp1, es_resp2])

        with patch.object(engine, "_execute_search", mock_exec):
            req = SearchRequest(
                user_id=1,
                query="",
                limit=page_size,
                aggregations={"dummy": {"terms": {"field": "operation_type"}}},
                filters={},
                metadata={},
            )
            resp = await engine.search(req)

        assert mock_exec.await_count == 2
        assert len(resp["results"]) == total_hits
        assert resp["response_metadata"]["total_pages"] > 1
        assert resp["response_metadata"]["returned_results"] == total_hits
        first = resp["results"][0]
        assert first["account_name"] == "Account 1"
        assert first["account_type"] == "checking"
        assert first["account_balance"] == 2001.0
        assert first["account_currency"] == "EUR"
        assert first["_score"] == 1.0
        assert "score" not in first

    asyncio.run(_run())
