import sys
import sys
import types
import time

sys.modules.setdefault("autogen", types.SimpleNamespace(AssistantAgent=object))

from conversation_service.agents.entity_extractor_agent import EntityExtractionCache
from conversation_service.models.core_models import FinancialEntity, EntityType


def test_entity_cache_store_and_retrieve():
    cache = EntityExtractionCache()
    entity = FinancialEntity(
        entity_type=EntityType.MERCHANT,
        raw_value="Amazon",
        normalized_value="Amazon",
        confidence=0.95
    )
    cache.set("user1", "Spent at Amazon", "MERCHANT_ANALYSIS", [entity])

    cached = cache.get("user1", "Spent at Amazon", "MERCHANT_ANALYSIS")
    assert cached is not None
    assert cached["cached"] is True
    assert len(cached["entities"]) == 1
    assert cache.hits == 1


def test_entity_cache_ttl_expiry():
    cache = EntityExtractionCache()
    entity = FinancialEntity(
        entity_type=EntityType.MERCHANT,
        raw_value="Amazon",
        normalized_value="Amazon",
        confidence=0.95,
    )
    cache.set("user1", "Will this expire?", "MERCHANT_ANALYSIS", [entity], ttl=1)

    time.sleep(1.1)
    cached = cache.get("user1", "Will this expire?", "MERCHANT_ANALYSIS", ttl=1)
    assert cached is None
