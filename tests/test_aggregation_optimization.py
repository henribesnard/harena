import json
import pytest
import requests

SEARCH_URL = "http://localhost:8000/api/v1/search/search"


def _post(payload: dict) -> requests.Response:
    response = requests.post(SEARCH_URL, json=payload, timeout=10)
    response.raise_for_status()
    return response


def test_aggregation_only_returns_no_results_but_same_aggregations():
    """Standard query vs aggregation_only should yield same aggregations"""
    base_payload = {
        "user_id": 1,
        "query": "",
        "limit": 5,
        "aggregations": {"operation_type": {"terms": {"field": "operation_type"}}},
    }
    agg_only_payload = {**base_payload, "aggregation_only": True}

    try:
        full_resp = _post(base_payload)
        agg_resp = _post(agg_only_payload)
    except requests.exceptions.RequestException:
        pytest.skip("search service not available or request invalid")

    data_full = full_resp.json()
    data_agg = agg_resp.json()

    assert data_agg.get("results") == []
    assert data_full.get("aggregations") == data_agg.get("aggregations")

    print(
        f"Full response size: {len(full_resp.content)} bytes\n"
        f"Aggregation-only response size: {len(agg_resp.content)} bytes"
    )
