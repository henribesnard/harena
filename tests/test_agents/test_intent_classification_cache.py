import sys
import types

sys.modules.setdefault("autogen", types.SimpleNamespace(AssistantAgent=object))

from conversation_service.agents.intent_classifier_agent import IntentClassificationCache
from conversation_service.models.core_models import IntentResult, IntentType


def test_intent_cache_store_and_retrieve():
    cache = IntentClassificationCache()
    result = IntentResult(
        intent=IntentType.BALANCE_INQUIRY,
        confidence=0.9,
        reasoning="Clear request for balance information"
    )
    cache.set("What's my balance?", result)

    cached = cache.get("What's my balance?")
    assert cached is not None
    assert cached.intent == IntentType.BALANCE_INQUIRY
    assert cache.hits == 1
