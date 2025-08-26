import pytest
from types import SimpleNamespace
from unittest.mock import patch

from conversation_service.api.routes.conversation import _collect_comprehensive_metrics
from conversation_service.prompts.harena_intents import HarenaIntentType


@pytest.mark.asyncio
async def test_collect_comprehensive_metrics_handles_enum_and_string_intent():
    base = {
        "category": "GENERAL",
        "confidence": 0.95,
        "is_supported": True,
        "alternatives": [],
    }
    agent_metrics = SimpleNamespace(cache_hit=True, tokens_consumed=5)

    # Intent provided as enum
    enum_result = SimpleNamespace(intent_type=HarenaIntentType.GREETING, **base)
    with patch("conversation_service.api.routes.conversation.metrics_collector") as mock_metrics:
        await _collect_comprehensive_metrics("req1", enum_result, 100, agent_metrics)
        mock_metrics.increment_counter.assert_any_call("conversation.intent.GREETING")

    # Intent provided as string
    str_result = SimpleNamespace(intent_type="GREETING", **base)
    with patch("conversation_service.api.routes.conversation.metrics_collector") as mock_metrics:
        await _collect_comprehensive_metrics("req2", str_result, 100, agent_metrics)
        mock_metrics.increment_counter.assert_any_call("conversation.intent.GREETING")
