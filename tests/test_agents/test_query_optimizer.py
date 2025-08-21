import sys
import types
from enum import Enum

sys.modules.setdefault("autogen", types.SimpleNamespace(AssistantAgent=object))
sys.modules.setdefault("conversation_service.models.agent_models", types.SimpleNamespace(AgentConfig=object))
sys.modules.setdefault("conversation_service.base_agent", types.SimpleNamespace(BaseFinancialAgent=object))
sys.modules.setdefault("conversation_service.core.cache_manager", types.SimpleNamespace(CacheManager=object))
sys.modules.setdefault("conversation_service.core.metrics_collector", types.SimpleNamespace(MetricsCollector=object))
sys.modules.setdefault("conversation_service.utils.logging", types.SimpleNamespace(get_structured_logger=lambda name: None))

import conversation_service.models.core_models as core_models

class QueryType(str, Enum):
    SIMPLE_SEARCH = "simple_search"

core_models.QueryType = QueryType

from conversation_service.agents.query_generator_agent import QueryOptimizer
from conversation_service.models.core_models import IntentType


def test_query_optimizer_applies_merchant_rule():
    base_query = {"search_parameters": {}, "aggregations": {}}
    optimized = QueryOptimizer.optimize_query(base_query, IntentType.MERCHANT_ANALYSIS)

    assert optimized["search_parameters"]["limit"] == 15
    assert "sort" in optimized["search_parameters"]
