import sys
import types
from enum import Enum
import asyncio
import pytest


class _DummyAssistantAgent:
    def __init__(self, *args, **kwargs):
        pass


class _DummyLogger:
    def info(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass


# Stub external dependencies required during imports
sys.modules.setdefault("autogen", types.SimpleNamespace(AssistantAgent=_DummyAssistantAgent))
sys.modules.setdefault("conversation_service.base_agent", types.SimpleNamespace(BaseFinancialAgent=object))
sys.modules.setdefault("conversation_service.core.cache_manager", types.SimpleNamespace(CacheManager=object))
sys.modules.setdefault("conversation_service.core.metrics_collector", types.SimpleNamespace(MetricsCollector=object))
sys.modules.setdefault(
    "conversation_service.utils.logging",
    types.SimpleNamespace(get_structured_logger=lambda name: _DummyLogger()),
)

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


class _DummySession:
    async def close(self):
        pass


sys.modules.setdefault(
    "aiohttp",
    types.SimpleNamespace(
        ClientSession=_DummySession,
        ClientTimeout=lambda *args, **kwargs: None,
        ClientError=Exception,
    ),
)


import conversation_service.models.core_models as core_models


class QueryType(str, Enum):
    SIMPLE_SEARCH = "simple_search"


core_models.QueryType = QueryType


from conversation_service.agents.query_generator_agent import QueryOptimizer
from conversation_service.models.core_models import IntentType
from conversation_service.agents.query_generator_agent import QueryGeneratorAgent


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

    assert result["search_request"]["filters"]["user_id"] == 99
    assert result["search_request"]["filters"]["other"] == "value"

