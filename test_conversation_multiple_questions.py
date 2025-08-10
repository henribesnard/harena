"""
Test spécifique pour vérifier le fonctionnement du service de conversation
avec plusieurs échanges successifs.

Ce script envoie une série de messages au service `/conversation/chat`
et vérifie que chaque réponse est retournée avec succès et que l'ID de
conversation reste constant.

Usage:
    python test_conversation_multiple_questions.py
    python test_conversation_multiple_questions.py --base-url http://localhost:8000/api/v1
    python test_conversation_multiple_questions.py --messages "Bonjour" "Comment ça va?" "Merci"
"""

import argparse
import json
import sys
from datetime import datetime
from typing import List
import types

# Stub missing third-party modules for isolated tests
openai_module = types.ModuleType("openai")
openai_module.AsyncOpenAI = type("AsyncOpenAI", (), {})
openai_types = types.ModuleType("openai.types")
openai_chat = types.ModuleType("openai.types.chat")
openai_chat.ChatCompletion = type("ChatCompletion", (), {})
openai_types.chat = openai_chat
sys.modules["openai"] = openai_module
sys.modules["openai.types"] = openai_types
sys.modules["openai.types.chat"] = openai_chat

pydantic_module = types.ModuleType("pydantic")
class BaseModel:
    def __init__(self, **data):
        for k, v in data.items():
            setattr(self, k, v)
def Field(default=None, *args, **kwargs):
    return default
def field_validator(*args, **kwargs):
    def decorator(func):
        return func
    return decorator
def model_validator(*args, **kwargs):
    def decorator(func):
        return func
    return decorator
class ValidationError(Exception):
    pass
pydantic_module.BaseModel = BaseModel
pydantic_module.Field = Field
pydantic_module.field_validator = field_validator
pydantic_module.model_validator = model_validator
pydantic_module.ValidationError = ValidationError
sys.modules["pydantic"] = pydantic_module

# Minimal requests stub for unit tests
requests_module = types.ModuleType("requests")
requests_module.Session = type("Session", (), {})
requests_module.Response = type("Response", (), {})
sys.modules["requests"] = requests_module

# Minimal httpx stub for DeepSeek client
httpx_module = types.ModuleType("httpx")
sys.modules["httpx"] = httpx_module

import requests

DEFAULT_BASE_URL = "http://localhost:8000/api/v1"
REQUEST_TIMEOUT = 30


def _print_step(step: int, message: str) -> None:
    print("\n" + "=" * 60)
    print(f"ÉTAPE {step}: {message}")
    print("=" * 60)


def _print_response(response: requests.Response) -> dict:
    print(f"Status Code: {response.status_code}")
    try:
        data = response.json()
        print("Réponse JSON:")
        print(json.dumps(data, indent=2, ensure_ascii=False))
        return data
    except json.JSONDecodeError:
        print("❌ Réponse non JSON:")
        print(response.text[:500])
        return {}


def run_conversation_test(base_url: str, conversation_id: str, messages: List[str]) -> bool:
    print("🚀 DÉBUT DU TEST MULTI-CONVERSATION HARENA")
    print(f"Base URL: {base_url}")
    print(f"Conversation ID: {conversation_id}")
    print(f"Timestamp: {datetime.now().isoformat()}")

    session = requests.Session()
    session.timeout = REQUEST_TIMEOUT

    successes = 0
    for idx, msg in enumerate(messages, start=1):
        _print_step(idx, f"ENVOI DU MESSAGE: {msg}")
        payload = {"conversation_id": conversation_id, "message": msg}
        try:
            response = session.post(
                f"{base_url.rstrip('/')}/conversation/chat",
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )
        except requests.exceptions.RequestException as e:
            print(f"❌ Erreur de requête: {e}")
            continue

        data = _print_response(response)
        if (
            response.status_code == 200
            and data.get("success") is True
            and data.get("conversation_id") == conversation_id
        ):
            print("✅ Message traité avec succès")
            successes += 1
        else:
            print("❌ Échec du traitement du message")

    print("\n" + "=" * 60)
    print("📊 RÉSUMÉ DU TEST")
    print("=" * 60)
    print(f"Messages réussis: {successes}/{len(messages)}")

    if successes == len(messages):
        print("✅ TOUS LES MESSAGES ONT ÉTÉ TRAITÉS AVEC SUCCÈS")
        return True

    print("❌ Des messages n'ont pas été traités correctement")
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Test de conversation multi-messages")
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="URL de base de l'API Harena",
    )
    parser.add_argument(
        "--conversation-id",
        default="test-conversation-multi",
        help="Identifiant de conversation à utiliser",
    )
    parser.add_argument(
        "--messages",
        nargs="*",
        default=[
            "Bonjour",
            "Peux-tu me donner ton nom ?",
            "Merci et au revoir",
        ],
        help="Liste de messages à envoyer",
    )
    args = parser.parse_args()

    success = run_conversation_test(args.base_url, args.conversation_id, args.messages)
    if not success:
        sys.exit(1)


def test_skips_search_when_not_required():
    from types import SimpleNamespace
    import asyncio
    import conversation_service.agents.base_financial_agent as base_financial_agent
    base_financial_agent.AUTOGEN_AVAILABLE = True
    base_financial_agent.AssistantAgent.__init__ = lambda self, *args, **kwargs: setattr(self, "name", kwargs.get("name"))
    from conversation_service.agents.orchestrator_agent import OrchestratorAgent
    from conversation_service.models.financial_models import IntentResult, IntentCategory, DetectionMethod

    class DummyIntentAgent:
        name = "intent"
        deepseek_client = SimpleNamespace(api_key="test", base_url="http://test")

        async def execute_with_metrics(self, data):
            ir = IntentResult(
                intent_type="TEST_INTENT",
                intent_category=IntentCategory.GENERAL_QUESTION,
                confidence=0.99,
                entities=[],
                method=DetectionMethod.RULE_BASED,
                processing_time_ms=1.0,
                suggested_actions=["No search needed"],
                search_required=False,
            )
            return SimpleNamespace(success=True, metadata={"intent_result": ir})

    class DummySearchAgent:
        name = "search"

        async def execute_with_metrics(self, data):
            raise RuntimeError("search should be skipped")

    class DummyResponseAgent:
        name = "response"

        async def execute_with_metrics(self, data):
            raise RuntimeError("response should be skipped")

    agent = OrchestratorAgent(DummyIntentAgent(), DummySearchAgent(), DummyResponseAgent())
    result = asyncio.run(agent.process_conversation("Bonjour", "conv1"))
    assert result["content"] == "No search needed"
    steps = {s["name"]: s["status"] for s in result["metadata"]["execution_details"]["steps"]}
    assert steps.get("search_query") == "skipped"
    assert steps.get("response_generation") == "skipped"


def test_gratitude_skips_search():
    import types
    import sys
    import asyncio

    import conversation_service.agents.base_financial_agent as base_financial_agent
    base_financial_agent.AUTOGEN_AVAILABLE = True
    base_financial_agent.AssistantAgent.__init__ = lambda self, *args, **kwargs: setattr(self, "name", kwargs.get("name"))

    deepseek_module = types.ModuleType("conversation_service.core.deepseek_client")

    class DeepSeekClient:
        def __init__(self):
            self.api_key = "test"
            self.base_url = "http://test"

        async def generate_response(self, *args, **kwargs):
            raise RuntimeError("AI should not be called")

    deepseek_module.DeepSeekClient = DeepSeekClient
    sys.modules["conversation_service.core.deepseek_client"] = deepseek_module

    from conversation_service.agents.hybrid_intent_agent import HybridIntentAgent

    intent_agent = HybridIntentAgent(DeepSeekClient())
    result = asyncio.run(intent_agent.detect_intent("merci"))
    intent_result = result["metadata"]["intent_result"]

    suggestions = intent_agent.rule_engine.all_rules["GRATITUDE"].suggested_responses
    assert intent_result.intent_type == "GRATITUDE"
    assert intent_result.search_required is False
    assert intent_result.suggested_actions == suggestions


if __name__ == "__main__":
    main()
