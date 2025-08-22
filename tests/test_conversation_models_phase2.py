import pytest
from uuid import uuid4
from pydantic import ValidationError

from conversation_service.models import (
    ConversationRequest,
    ConversationResponse,
    ConversationMetadata,
    ConversationContext,
)


def test_conversation_request_valid():
    ctx = ConversationContext(conversation_id=uuid4(), turn_number=1)
    req = ConversationRequest(message="Hello", language="en", context=ctx)
    assert req.context.turn_number == 1


def test_user_message_not_empty():
    with pytest.raises(ValidationError):
        ConversationRequest(message="", language="en", context={"turn_number": 1})


def test_language_two_letters():
    with pytest.raises(ValidationError):
        ConversationRequest(message="Hi", language="eng", context={"turn_number": 1})


def test_conversation_id_uuid():
    with pytest.raises(ValidationError):
        ConversationRequest(
            message="Hi",
            language="en",
            context={"conversation_id": "not-a-uuid", "turn_number": 1},
        )


def test_confidence_score_range():
    with pytest.raises(ValidationError):
        ConversationMetadata(intent="greeting", confidence_score=1.5)
    with pytest.raises(ValidationError):
        ConversationMetadata(intent="greeting", confidence_score=-0.1)


def test_turn_number_positive():
    with pytest.raises(ValidationError):
        ConversationContext(conversation_id=uuid4(), turn_number=0)


def test_conversation_response_valid():
    ctx = ConversationContext(conversation_id=uuid4(), turn_number=2)
    meta = ConversationMetadata(intent="greeting", confidence_score=0.8)
    resp = ConversationResponse(
        response="Hello!", language="en", context=ctx, metadata=meta
    )
    assert resp.metadata.confidence_score == 0.8
