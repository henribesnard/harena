import sys
import types
import time

import pytest

sys.modules.setdefault("autogen", types.SimpleNamespace(AssistantAgent=object))

agent_module = pytest.importorskip(
    "conversation_service.agents.entity_extractor_agent"
)
EntityExtractionCache = getattr(agent_module, "EntityExtractionCache", None)
if EntityExtractionCache is None:  # pragma: no cover - missing implementation
    pytest.skip("EntityExtractionCache not available", allow_module_level=True)

from conversation_service.models.core_models import FinancialEntity, EntityType


def test_entity_cache_store_and_retrieve():
    cache = EntityExtractionCache()
    entity = FinancialEntity(
        entity_type=EntityType.MERCHANT,
        raw_value="Amazon",
        normalized_value="Amazon",
        confidence=0.95
    )
    cache.set("user1", "MERCHANT_ANALYSIS", "Spent at Amazon", [entity])

    cached = cache.get("user1", "MERCHANT_ANALYSIS", "Spent at Amazon")
    assert cached is not None
    assert len(cached) == 1
    assert cache.hits == 1


def test_entity_cache_ttl_expiry():
    cache = EntityExtractionCache()
    entity = FinancialEntity(
        entity_type=EntityType.MERCHANT,
        raw_value="Amazon",
        normalized_value="Amazon",
        confidence=0.95,
    )
    cache.set("user1", "MERCHANT_ANALYSIS", "Will this expire?", [entity], ttl=1)

    time.sleep(1.1)
    cached = cache.get("user1", "MERCHANT_ANALYSIS", "Will this expire?", ttl=1)
    assert cached is None


def test_cache_clear_resets_state():
    cache = EntityExtractionCache()
    entity = FinancialEntity(
        entity_type=EntityType.MERCHANT,
        raw_value="Amazon",
        normalized_value="Amazon",
        confidence=0.95,
    )
    cache.set("user1", "MERCHANT_ANALYSIS", "Spent at Amazon", [entity])

    # Access to register a cache hit and ensure store populated
    cache.get("user1", "MERCHANT_ANALYSIS", "Spent at Amazon")
    assert cache.hits == 1
    assert cache.store

    cache.clear()
    assert cache.hits == 0
    assert cache.store == {}


def test_repeated_set_updates_timestamp():
    cache = EntityExtractionCache()
    entity = FinancialEntity(
        entity_type=EntityType.MERCHANT,
        raw_value="Amazon",
        normalized_value="Amazon",
        confidence=0.95,
    )

    cache.set("user1", "MERCHANT_ANALYSIS", "Spent at Amazon", [entity])
    key, value = next(iter(cache.store.items()))
    first_timestamp = value[0]

    time.sleep(0.01)
    cache.set("user1", "MERCHANT_ANALYSIS", "Spent at Amazon", [entity])
    second_timestamp = cache.store[key][0]

    assert second_timestamp > first_timestamp
