import pytest
from uuid import uuid4
from pydantic import ValidationError

from conversation_service.models import (
    ConversationRequest,
    ConversationResponse,
    ConversationContext,
    ConversationMetadata,
    IntentType,
    DynamicFinancialEntity,
    EntityType,
)


def test_conversation_request_valid():
    ctx = ConversationContext(turn_number=1)
    req = ConversationRequest(
        message="Hello",
        language="en",
        conversation_id=uuid4(),
        context=ctx,
        user_preferences={"tone": "formal"},
    )
    assert req.context.turn_number == 1
    assert req.user_preferences["tone"] == "formal"


def test_user_message_not_empty():
    with pytest.raises(ValidationError):
        ConversationRequest(
            message="",
            language="en",
            conversation_id=uuid4(),
            context={"turn_number": 1},
        )


def test_language_two_letters():
    with pytest.raises(ValidationError):
        ConversationRequest(
            message="Hi",
            language="eng",
            conversation_id=uuid4(),
            context={"turn_number": 1},
        )


def test_conversation_id_uuid():
    with pytest.raises(ValidationError):
        ConversationRequest(
            message="Hi",
            language="en",
            conversation_id="not-a-uuid",
            context={"turn_number": 1},
        )


def test_confidence_score_range():
    ctx = ConversationContext(conversation_id=uuid4(), turn_number=1)
    with pytest.raises(ValidationError):
        ConversationMetadata(
            intent=IntentType.GREETING,
            confidence_score=1.5,
            extraction_mode="auto",
        )
    with pytest.raises(ValidationError):
        ConversationMetadata(
            intent=IntentType.GREETING,
            confidence_score=-0.1,
            extraction_mode="auto",
        )
    with pytest.raises(ValidationError):
        ConversationResponse(
            original_message="Hi",
            response="Hi",
            intent=IntentType.GREETING,
            entities=[],
            confidence_score=1.5,
            language="en",
            context=ctx,
        )
    with pytest.raises(ValidationError):
        ConversationResponse(
            original_message="Hi",
            response="Hi",
            intent=IntentType.GREETING,
            entities=[],
            confidence_score=-0.1,
            language="en",
            context=ctx,
        )


def test_turn_number_positive():
    with pytest.raises(ValidationError):
        ConversationContext(turn_number=0)


def test_conversation_response_valid():
    ctx = ConversationContext(conversation_id=uuid4(), turn_number=2)
    meta = ConversationMetadata(
        intent=IntentType.GREETING,
        confidence_score=0.8,
        extraction_mode="auto",
    )
    entity = DynamicFinancialEntity(
        entity_type=EntityType.ACCOUNT,
        raw_value="123",
        confidence_score=0.9,
    )
    resp = ConversationResponse(
        original_message="Hi",
        response="Hello!",
        intent=IntentType.GREETING,
        entities=[entity],
        confidence_score=0.8,
        language="en",
        context=ctx,
        suggested_actions=["check_balance"],
        user_preferences={"tone": "friendly"},
    )
    assert resp.intent == IntentType.GREETING
    assert resp.entities[0].raw_value == "123"
    assert resp.confidence_score == 0.8
    assert resp.suggested_actions == ["check_balance"]
    assert resp.user_preferences["tone"] == "friendly"

def test_metadata_invalid_intent():
    with pytest.raises(ValidationError):
        ConversationMetadata(
            intent="NOT_AN_INTENT",
            confidence_score=0.5,
            extraction_mode="auto",
        )


def test_metadata_negative_cache_stats():
    with pytest.raises(ValidationError):
        ConversationMetadata(
            intent=IntentType.GREETING,
            cache_stats={"hits": -1},
            extraction_mode="auto",
        )


def test_context_invalid_session_state():
    with pytest.raises(ValidationError):
        ConversationContext(conversation_id=uuid4(), turn_number=1, session_state="old")


def test_auto_summary_not_empty():
    with pytest.raises(ValidationError):
        ConversationContext(
            conversation_id=uuid4(),
            turn_number=1,
            auto_summary="",
        )

