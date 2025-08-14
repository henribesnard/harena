import time

from conversation_service.models.financial_models import (
    DetectionMethod,
    IntentCategory,
    IntentResult,
)
from conversation_service.utils.intent_cache import IntentResultCache


def make_result(confidence: float) -> IntentResult:
    return IntentResult(
        intent_type="GREETING",
        intent_category=IntentCategory.GREETING,
        confidence=confidence,
        entities=[],
        method=DetectionMethod.LLM_BASED,
        processing_time_ms=0.0,
    )


def test_cache_hit_miss_and_metrics():
    cache = IntentResultCache(max_size=10)
    assert cache.get("hello") is None
    cache.set("hello", make_result(0.9))
    assert cache.get("hello") is not None
    metrics = cache.get_metrics()
    assert metrics["hits"] == 1
    assert metrics["misses"] == 1


def test_adaptive_ttl_based_on_confidence():
    cache = IntentResultCache(max_size=10)
    cache.set("hi", make_result(0.1))
    cache.set("hello", make_result(0.9))
    import hashlib
    low_key = hashlib.sha256("hi".encode()).hexdigest()
    high_key = hashlib.sha256("hello".encode()).hexdigest()
    low_ttl = cache._store._cache[low_key].ttl  # type: ignore[attr-defined]
    high_ttl = cache._store._cache[high_key].ttl  # type: ignore[attr-defined]
    assert high_ttl > low_ttl


def test_semantic_deduplication():
    cache = IntentResultCache(max_size=10, similarity_threshold=0.8)
    cache.set("check balance", make_result(0.5))
    # semantically similar message should hit the cache
    assert cache.get("check my balance") is not None
    # inserting a similar message should not increase size
    cache.set("check my balance", make_result(0.5))
    assert cache.get_metrics()["size"] == 1
