import sys
import sys
import types
from enum import Enum

sys.modules.setdefault("autogen", types.SimpleNamespace(AssistantAgent=object))
sys.modules.setdefault("conversation_service.models.agent_models", types.SimpleNamespace(AgentConfig=object))
sys.modules.setdefault("conversation_service.base_agent", types.SimpleNamespace(BaseFinancialAgent=object))


class DummyLogger:
    def info(self, *args, **kwargs):
        pass

    def exception(self, *args, **kwargs):
        pass


sys.modules.setdefault(
    "conversation_service.utils.logging",
    types.SimpleNamespace(get_structured_logger=lambda name: DummyLogger()),
)

import conversation_service.models.core_models as core_models

class QueryType(str, Enum):
    SIMPLE_SEARCH = "simple_search"

core_models.QueryType = QueryType

from conversation_service.agents.intent_classifier import IntentClassificationCache
from conversation_service.agents.query_generator import QueryOptimizer
from conversation_service.models.core_models import (
    FinancialEntity,
    EntityType,
    IntentResult,
    IntentType,
)


def test_conversation_pipeline():
    cache = IntentClassificationCache()
    intent_result = IntentResult(
        intent=IntentType.MERCHANT_ANALYSIS,
        confidence=0.92,
        reasoning="User asks about spending at a merchant",
    )
    cache.set("user1", "How much did I spend at Amazon?", intent_result)
    cached_intent = cache.get("user1", "How much did I spend at Amazon?")

    entity = FinancialEntity(
        entity_type=EntityType.MERCHANT,
        raw_value="Amazon",
        normalized_value="Amazon",
        confidence=0.95,
    )
    filter_dict = entity.to_search_filter()

    base_query = {"search_parameters": {}, "aggregations": {}}
    optimized_query = QueryOptimizer.optimize_query(base_query, cached_intent.intent)

    assert "merchant_name" in filter_dict["bool"]["should"][0]["match"]
    assert optimized_query["search_parameters"]["limit"] == 15
