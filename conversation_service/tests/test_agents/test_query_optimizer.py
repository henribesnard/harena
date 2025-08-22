import sys
import types
import asyncio
from enum import Enum
import pytest

sys.modules.setdefault("autogen", types.SimpleNamespace(AssistantAgent=object))
sys.modules.setdefault("conversation_service.models.agent_models", types.SimpleNamespace(AgentConfig=object))
sys.modules.setdefault("conversation_service.base_agent", types.SimpleNamespace(BaseFinancialAgent=object))
sys.modules.setdefault("conversation_service.utils.logging", types.SimpleNamespace(get_structured_logger=lambda name: None))

_clients_pkg = types.ModuleType("conversation_service.clients")
_clients_pkg.__path__ = []  # type: ignore[attr-defined]
sys.modules.setdefault("conversation_service.clients", _clients_pkg)

_openai_client_module = types.ModuleType("conversation_service.clients.openai_client")
_openai_client_module.OpenAIClient = object
sys.modules.setdefault("conversation_service.clients.openai_client", _openai_client_module)

_search_client_module = types.ModuleType("conversation_service.clients.search_client")
_search_client_module.SearchClient = object
sys.modules.setdefault("conversation_service.clients.search_client", _search_client_module)

_cache_client_module = types.ModuleType("conversation_service.clients.cache_client")
_cache_client_module.CacheClient = object
sys.modules.setdefault("conversation_service.clients.cache_client", _cache_client_module)

_clients_pkg.OpenAIClient = _openai_client_module.OpenAIClient
_clients_pkg.SearchClient = _search_client_module.SearchClient
_clients_pkg.CacheClient = _cache_client_module.CacheClient

sys.modules.setdefault("openai", types.SimpleNamespace(AsyncOpenAI=object))

import conversation_service.models.core_models as core_models

class QueryType(str, Enum):
    SIMPLE_SEARCH = "simple_search"

core_models.QueryType = QueryType

agent_module = pytest.importorskip("conversation_service.agents.query_generator")
QueryOptimizer = getattr(agent_module, "QueryOptimizer", None)
if QueryOptimizer is None:  # pragma: no cover - missing implementation
    pytest.skip("QueryOptimizer not available", allow_module_level=True)

from conversation_service.models.core_models import IntentType

from conversation_service.agents.query_generator import QueryGeneratorAgent


def test_query_optimizer_applies_merchant_rule():
    base_query = {"search_parameters": {}, "aggregations": {}}
    optimized = QueryOptimizer.optimize_query(base_query, IntentType.MERCHANT_ANALYSIS)

    assert optimized["search_parameters"]["limit"] == 15
    assert "sort" in optimized["search_parameters"]


class _DummySearchClient:
    def __init__(self):
        self.payload = None

    async def search(self, user_id, payload):
        self.payload = payload
        return {}


def test_query_generator_injects_user_id_into_filters():
    search_client = _DummySearchClient()
    agent = QueryGeneratorAgent(search_client=search_client)
    input_data = {
        "intent": "any_intent",
        "entities": {"foo": "bar"},
        "context": {
            "user_id": 99,
            "filters": {"user_id": 1, "other": "value"},
        },
    }

    result = asyncio.run(agent._process_implementation(input_data))

    assert search_client.payload["filters"]["user_id"] == 99
    assert search_client.payload["filters"]["other"] == "value"
    assert result["search_request"]["filters"]["user_id"] == 99
    assert result["search_request"]["filters"]["other"] == "value"
