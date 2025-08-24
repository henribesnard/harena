from unittest.mock import AsyncMock, patch

import pytest

from search_service.core.search_engine import SearchEngine
from search_service.models.request import SearchRequest


def _make_hit() -> dict:
    return {
        "_source": {
            "transaction_id": "tx_1",
            "user_id": 1,
            "amount": -5.0,
            "amount_abs": 5.0,
            "currency_code": "EUR",
            "transaction_type": "debit",
            "date": "2024-01-01",
            "primary_description": "coffee shop purchase",
        },
        "_score": 1.0,
    }


@pytest.mark.asyncio
async def test_two_word_search_returns_results():
    engine = SearchEngine()
    engine.elasticsearch_client = object()
    engine.cache_enabled = False

    req = SearchRequest(user_id=1, query="coffee shop")

    es_response = {
        "hits": {"hits": [_make_hit()], "total": {"value": 1}},
        "took": 1,
    }

    with patch.object(engine, "_execute_search", AsyncMock(return_value=es_response)) as mock_exec:
        resp = await engine.search(req)

    assert len(resp["results"]) >= 1
    es_query = mock_exec.await_args.args[0]
    assert (
        es_query["query"]["bool"]["must"][0]["multi_match"]["minimum_should_match"]
        == "50%"
    )
