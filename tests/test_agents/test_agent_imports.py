import importlib
import pytest

MODULES = [
    ("conversation_service.agents.entity_extractor_agent", ["EntityExtractionCache"]),
    ("conversation_service.agents.intent_classifier_agent", ["IntentClassificationCache"]),
    ("conversation_service.agents.query_generator_agent", ["QueryOptimizer"]),
    (
        "conversation_service.agents.response_generator_agent",
        ["ResponseGeneratorAgent", "stream_response"],
    ),
]


@pytest.mark.parametrize("module_name, symbols", MODULES)
def test_agent_modules_expose_symbols(module_name, symbols):
    module = importlib.import_module(module_name)
    for symbol in symbols:
        assert hasattr(module, symbol)
