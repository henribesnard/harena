import sys
import types

sys.modules.setdefault("autogen", types.SimpleNamespace(AssistantAgent=object))

from conversation_service.agents.entity_extractor_agent import EntityExtractionCache
from conversation_service.models.financial_models import FinancialEntity
from conversation_service.models.enums import EntityType


def test_entity_cache_store_and_retrieve():
    cache = EntityExtractionCache()
    entity = FinancialEntity(
        entity_type=EntityType.MERCHANT,
        raw_value="Amazon",
        normalized_value="Amazon",
        confidence=0.95
    )
    cache.set("Spent at Amazon", "MERCHANT_ANALYSIS", [entity])

    cached = cache.get("Spent at Amazon", "MERCHANT_ANALYSIS")
    assert cached is not None
    assert cached["cached"] is True
    assert len(cached["entities"]) == 1
    assert cache.hits == 1
