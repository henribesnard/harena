import os
from datetime import datetime

from conversation_service.intent_rules import create_rule_engine, MatchingContext


def test_adaptive_ttl_and_metrics(monkeypatch):
    monkeypatch.setenv("RULE_ENGINE_CACHE_TTL", "10")
    engine = create_rule_engine()

    text_high = "bonjour"
    ctx_high = MatchingContext.create(text_high)
    key_high = f"{ctx_high.text_hash}_{0.3}"

    first = engine.match_intent(text_high, confidence_threshold=0.3)
    second = engine.match_intent(text_high, confidence_threshold=0.3)
    assert first == second

    entry_high = engine._result_cache[key_high]
    assert entry_high.semantic_key == ctx_high.text_hash

    metrics = engine.get_cache_metrics()
    assert metrics["hits"] == 1
    assert metrics["misses"] == 1
    assert metrics["cache_size"] == 1

    text_low = "euh"
    ctx_low = MatchingContext.create(text_low)
    key_low = f"{ctx_low.text_hash}_{0.3}"
    engine.match_intent(text_low, confidence_threshold=0.3)
    entry_low = engine._result_cache[key_low]

    ttl_high = (entry_high.expires_at - datetime.now()).total_seconds()
    ttl_low = (entry_low.expires_at - datetime.now()).total_seconds()
    assert ttl_high > ttl_low

    metrics2 = engine.get_cache_metrics()
    assert metrics2["hits"] == 1
    assert metrics2["misses"] == 2
    assert metrics2["cache_size"] >= 2
