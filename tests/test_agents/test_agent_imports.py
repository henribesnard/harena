import importlib
import sys
import types
import pytest

MODULES = [
    ("conversation_service.agents.entity_extractor_agent", ["EntityExtractionCache"]),
    ("conversation_service.agents.intent_classifier_agent", ["IntentClassificationCache"]),
    ("conversation_service.agents.query_generator_agent", ["QueryOptimizer"]),
    (
        "conversation_service.agents.response_generator_agent",
        ["ResponseGeneratorAgent"],
    ),
    ("conversation_service.utils.streaming", ["stream_response"]),
]


@pytest.mark.parametrize("module_name, symbols", MODULES)
def test_agent_modules_expose_symbols(module_name, symbols):
    if module_name == "conversation_service.agents.query_generator_agent":
        # Provide minimal stubs so that the module can be imported without the
        # real HTTP and OpenAI dependencies present in production.
        clients_pkg = types.ModuleType("conversation_service.clients")
        clients_pkg.__path__ = []  # type: ignore[attr-defined]
        sys.modules.setdefault("conversation_service.clients", clients_pkg)

        openai_client_module = types.ModuleType(
            "conversation_service.clients.openai_client"
        )
        openai_client_module.OpenAIClient = object
        search_client_module = types.ModuleType(
            "conversation_service.clients.search_client"
        )
        search_client_module.SearchClient = object
        cache_client_module = types.ModuleType(
            "conversation_service.clients.cache_client"
        )
        cache_client_module.CacheClient = object

        sys.modules.setdefault(
            "conversation_service.clients.openai_client", openai_client_module
        )
        sys.modules.setdefault(
            "conversation_service.clients.search_client", search_client_module
        )
        sys.modules.setdefault(
            "conversation_service.clients.cache_client", cache_client_module
        )

        clients_pkg.OpenAIClient = openai_client_module.OpenAIClient
        clients_pkg.SearchClient = search_client_module.SearchClient
        clients_pkg.CacheClient = cache_client_module.CacheClient

        sys.modules.setdefault("openai", types.SimpleNamespace(AsyncOpenAI=object))
        sys.modules.setdefault(
            "aiohttp",
            types.SimpleNamespace(
                ClientSession=object,
                ClientTimeout=lambda *args, **kwargs: None,
                ClientError=Exception,
            ),
        )

    module = importlib.import_module(module_name)
    for symbol in symbols:
        assert hasattr(module, symbol)
