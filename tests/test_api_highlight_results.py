from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock

from search_service.api.routes import router, get_search_engine
from search_service.core.search_engine import SearchEngine


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
            "primary_description": "some restaurant",
        },
        "_score": 1.0,
        "highlight": {"primary_description": ["some <em>restaurant</em>"]},
    }


def test_search_endpoint_returns_highlights():
    app = FastAPI()
    app.include_router(router)

    engine = SearchEngine()
    engine.elasticsearch_client = object()
    engine.cache_enabled = False

    es_response = {
        "hits": {"hits": [_make_hit()], "total": {"value": 1}},
        "took": 1,
    }

    engine._execute_search = AsyncMock(return_value=es_response)
    app.dependency_overrides[get_search_engine] = lambda: engine

    client = TestClient(app)
    payload = {
        "user_id": 1,
        "query": "restaurant",
        "highlight": {"fields": {"primary_description": {}}},
    }

    response = client.post("/search", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["results"][0]["highlights"] == {
        "primary_description": ["some <em>restaurant</em>"]
    }

    es_query = engine._execute_search.await_args.args[0]
    assert es_query["highlight"] == payload["highlight"]
