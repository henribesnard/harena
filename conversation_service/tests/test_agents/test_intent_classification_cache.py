import sys
import sys
import types
import pytest
import time

sys.modules.setdefault("autogen", types.SimpleNamespace(AssistantAgent=object))

agent_module = pytest.importorskip("conversation_service.agents.intent_classifier")
IntentClassificationCache = getattr(agent_module, "IntentClassificationCache", None)
if IntentClassificationCache is None:  # pragma: no cover - missing implementation
    pytest.skip("IntentClassificationCache not available", allow_module_level=True)

from conversation_service.models.core_models import IntentResult, IntentType


def test_intent_cache_store_and_retrieve():
    cache = IntentClassificationCache()
    result = IntentResult(
        intent=IntentType.BALANCE_INQUIRY,
        confidence=0.9,
        reasoning="Clear request for balance information",
    )
    cache.set("user1", "What's my balance?", result)

    cached = cache.get("user1", "What's my balance?")
    assert cached is not None
    assert cached.intent == IntentType.BALANCE_INQUIRY
    assert cache.hits == 1


def test_intent_cache_ttl_expiry():
    cache = IntentClassificationCache()
    result = IntentResult(
        intent=IntentType.BALANCE_INQUIRY,
        confidence=0.9,
        reasoning="Clear request for balance information",
    )
    cache.set("user1", "Will this expire?", result, ttl=1)

    time.sleep(1.1)
    cached = cache.get("user1", "Will this expire?", ttl=1)
    assert cached is None
