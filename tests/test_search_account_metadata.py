import asyncio
from unittest.mock import AsyncMock, patch

from search_service.core.search_engine import SearchEngine
from search_service.models.request import SearchRequest


def _make_es_response():
    return {
        "hits": {
            "hits": [
                {
                    "_source": {
                        "transaction_id": "tx_1",
                        "user_id": 1,
                        "account_id": 42,
                        "account_name": "Main Account",
                        "account_type": "checking",
                        "account_balance": 1234.56,
                        "account_currency": "EUR",
                        "amount": -12.34,
                        "amount_abs": 12.34,
                        "currency_code": "EUR",
                        "transaction_type": "debit",
                        "date": "2024-01-01",
                        "primary_description": "Coffee shop",
                    },
                    "_score": 1.0,
                }
            ],
            "total": {"value": 1},
        },
        "took": 1,
    }


def test_search_returns_account_metadata():
    async def _run():
        engine = SearchEngine()
        engine.elasticsearch_client = object()

        es_resp = _make_es_response()
        with patch.object(engine, "_execute_search", AsyncMock(return_value=es_resp)):
            req = SearchRequest(
                user_id=1,
                query="coffee",
                limit=10,
                filters={},
                metadata={},
            )
            resp = await engine.search(req)

        result = resp["results"][0]
        assert result["account_name"] == "Main Account"
        assert result["account_type"] == "checking"
        assert result["account_balance"] == 1234.56
        assert result["account_currency"] == "EUR"

    asyncio.run(_run())
