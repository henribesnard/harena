import types
import sys

# Preserve original modules so other tests are unaffected
_orig_config_service = sys.modules.get("config_service")
_orig_config_config = sys.modules.get("config_service.config")
_orig_pydantic_settings = sys.modules.get("pydantic_settings")

sys.modules["pydantic_settings"] = types.ModuleType("pydantic_settings")
config_module = types.ModuleType("config_service.config")
setattr(config_module, "settings", types.SimpleNamespace())
config_package = types.ModuleType("config_service")
setattr(config_package, "config", config_module)
sys.modules["config_service"] = config_package
sys.modules["config_service.config"] = config_module

from conversation_service.prompts.search_prompts import format_search_prompt
from conversation_service.prompts.response_prompts import format_response_prompt
from conversation_service.prompts.intent_prompts import format_intent_prompt
from conversation_service.prompts.orchestrator_prompts import (
    format_orchestrator_prompt,
    WorkflowStep,
)

# Restore original modules after imports
if _orig_config_service is not None:
    sys.modules["config_service"] = _orig_config_service
else:
    del sys.modules["config_service"]
if _orig_config_config is not None:
    sys.modules["config_service.config"] = _orig_config_config
else:
    del sys.modules["config_service.config"]
if _orig_pydantic_settings is not None:
    sys.modules["pydantic_settings"] = _orig_pydantic_settings
else:
    del sys.modules["pydantic_settings"]


def test_search_prompt_examples_loaded():
    intent = {"intent": "transaction_query", "confidence": 0.9, "entities": {}}
    prompt = format_search_prompt(intent, "Mes achats")
    assert "EXEMPLES DE GÉNÉRATION DE REQUÊTES" in prompt
    assert "Transaction simple" in prompt


def test_intent_prompt_examples_loaded():
    prompt = format_intent_prompt("Transactions de 50 euros")
    assert "EXEMPLES DE CLASSIFICATION" in prompt
    assert "Transactions de 50 euros" in prompt


def test_response_prompt_examples_loaded():
    search_results = {"results": [], "response_metadata": {}}
    prompt = format_response_prompt("Mes achats", search_results)
    assert "EXEMPLES DE GÉNÉRATION DE RÉPONSES" in prompt
    assert "Transactions simples" in prompt


def test_orchestrator_prompt_examples_loaded():
    prompt = format_orchestrator_prompt(WorkflowStep.INTENT_DETECTION, {})
    assert "EXEMPLES DE DÉCISIONS ORCHESTRATEUR" in prompt
    assert "Détection d'intention" in prompt
