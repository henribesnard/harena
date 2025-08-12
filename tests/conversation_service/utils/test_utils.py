import os
import sys
import pytest

# Ensure the package root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from conversation_service.utils import cache, metrics


def test_multilevelcache_l1_l2_interaction(monkeypatch):
    """Ensure L1 and Redis L2 caches cooperate via fakeredis."""
    import asyncio

    fakeredis = pytest.importorskip("fakeredis.aioredis")
    # Patch redis module to use fakeredis
    monkeypatch.setattr(cache, "redis_module", fakeredis)
    monkeypatch.setattr(cache, "RedisConnectionError", Exception)
    monkeypatch.setattr(cache, "RedisTimeoutError", Exception)
    monkeypatch.setattr(cache, "RedisError", Exception)
    monkeypatch.setattr(cache, "REDIS_AVAILABLE", True)

    ml_cache = cache.MultiLevelCache(
        l1_size=10,
        l2_redis_url="redis://localhost",
        l1_ttl=60,
        l2_ttl=3600,
    )

    asyncio.run(ml_cache.set("foo", "bar"))
    assert ml_cache.l1_cache.get("foo") == "bar"

    # Remove from L1 to force fetch from L2
    ml_cache.l1_cache.delete("foo")
    assert ml_cache.l1_cache.get("foo") is None

    value = asyncio.run(ml_cache.get("foo"))
    assert value == "bar"
    # After fetching from L2, value should be back in L1
    assert ml_cache.l1_cache.get("foo") == "bar"

    asyncio.run(ml_cache.clear())


def test_metricscollector_aggregation_and_latency_alert(monkeypatch):
    """MetricsCollector should aggregate and trigger latency alerts."""
    monkeypatch.setenv("PERFORMANCE_ALERT_THRESHOLD_MS", "50")
    mc = metrics.MetricsCollector()
    mc.clear_metrics()

    # Record two response times; one should trigger alert
    mc.record_response_time("endpoint", 30)
    mc.record_response_time("endpoint", 70)

    summary = mc.get_summary()
    key = "response_time_ms{endpoint=endpoint}"
    assert summary["histograms"][key]["count"] == 2
    assert summary["histograms"][key]["avg"] == pytest.approx(50)

    alerts = mc.get_alerts()
    assert any(a.metric_name == "response_time_ms_slow" for a in alerts)


def test_contractvalidator_positive_and_negative():
    """ContractValidator should validate correct contracts and report errors for invalid ones."""
    pytest.importorskip("pydantic")
    from conversation_service.utils import validators
    valid_query = {
        "query_metadata": {
            "conversation_id": "conv-1",
            "user_id": 123,
            "intent_type": "search",
        },
        "search_parameters": {
            "max_results": 10,
            "search_strategy": "lexical",
        },
        "filters": {
            "date_range": {"start": "2023-01-01", "end": "2023-12-31"}
        },
    }

    invalid_query = {
        "query_metadata": {"conversation_id": "conv-1", "user_id": 123},
        "search_parameters": {
            "max_results": 0,
            "search_strategy": "invalid",
        },
        "filters": {
            "date_range": {"start": "2023-12-31", "end": "2023-01-01"}
        },
    }

    assert validators.ContractValidator.validate_search_query(valid_query) == []

    errors = validators.ContractValidator.validate_search_query(invalid_query)
    assert errors
    assert any("intent_type" in e for e in errors)
