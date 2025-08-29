from datetime import datetime, timezone

from conversation_service.models.responses.autogen_conversation_response import (
    AutogenConversationResponse,
)
from conversation_service.models.responses.conversation_responses import (
    IntentClassificationResult,
    AgentMetrics,
)
from conversation_service.prompts.harena_intents import HarenaIntentType


def _make_intent() -> IntentClassificationResult:
    return IntentClassificationResult(
        intent_type=HarenaIntentType.GREETING,
        confidence=0.9,
        reasoning="salutation",
        original_message="salut",
        category="TEST",
        is_supported=True,
    )


def _make_metrics() -> AgentMetrics:
    return AgentMetrics(
        agent_used="test_agent",
        model_used="test_model",
        tokens_consumed=10,
        processing_time_ms=1,
        confidence_threshold_met=True,
        cache_hit=False,
    )


def test_apply_team_results_updates_metadata_and_returns_self():
    response = AutogenConversationResponse(
        user_id=1,
        message="hello",
        timestamp=datetime.now(timezone.utc),
        intent=_make_intent(),
        agent_metrics=_make_metrics(),
        processing_time_ms=1,
    )

    assert response.autogen_metadata is None

    team_results = {"foo": "bar"}
    returned = response.apply_team_results(team_results)

    assert response.autogen_metadata == team_results
    assert returned is response
