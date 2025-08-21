import pytest

from search_service.core.search_engine import SearchEngine
from search_service.models.request import SearchRequest


class DummyESClient:
    async def search(self, index, body, size, from_):
        return {
            "hits": {
                "hits": [
                    {"_source": {
                        "transaction_id": "t1",
                        "user_id": 1,
                        "account_id": 1,
                        "amount": 10.0,
                        "amount_abs": 10.0,
                        "currency_code": "EUR",
                        "transaction_type": "debit",
                        "date": "2024-01-01",
                        "primary_description": "first"
                    }},
                    {"_source": {
                        "transaction_id": "t2",
                        "user_id": 2,
                        "account_id": 1,
                        "amount": 20.0,
                        "amount_abs": 20.0,
                        "currency_code": "EUR",
                        "transaction_type": "debit",
                        "date": "2024-01-02",
                        "primary_description": "second"
                    }}
                ]
            }
        }


@pytest.mark.asyncio
async def test_search_engine_filters_results_by_user_id():
    engine = SearchEngine(DummyESClient(), cache_enabled=False)
    req = SearchRequest(user_id=1, query="")
    resp = await engine.search(req)
    assert len(resp["results"]) == 1
    assert resp["results"][0]["user_id"] == 1
