import asyncio
import json

from db_service.models import Conversation, ConversationTurn
from conversation_service.models import (
    DynamicFinancialEntity,
    IntentType,
    EntityType,
)

# Import fixtures from phase 0 tests so they're available here
from tests.test_phase_0.conftest import db_session, user  # noqa: F401


def test_full_pipeline(db_session, user, openai_mock, cache):
    """End-to-end integration of phases 0, 1 and 2."""
    # Phase 0: create conversation with metadata
    conv = Conversation(user_id=user.id, conversation_metadata={"topic": "integration"})
    db_session.add(conv)
    db_session.commit()

    stored_conv = db_session.get(Conversation, conv.id)
    assert stored_conv.conversation_metadata == {"topic": "integration"}

    # Phase 1: simulate OpenAI call and cache the intent result
    response = asyncio.run(openai_mock.chat.completions.create(messages=[]))
    intent_data = json.loads(response.choices[0].message.content)
    cache_key = f"{user.id}:{conv.conversation_id}:intent"
    cache.set(cache_key, intent_data)
    assert cache.get(cache_key) == intent_data

    intent_enum = IntentType(intent_data["intent_type"])
    assert intent_enum is IntentType.GREETING

    # Phase 2: add a conversation turn with dynamic entities
    entity = DynamicFinancialEntity(
        entity_type=EntityType.ACCOUNT,
        raw_value="checking",
        confidence_score=0.95,
    )
    turn = ConversationTurn(
        conversation_id=conv.id,
        turn_number=1,
        user_message="What is my account balance?",
        assistant_response="Your balance is 100.",
        intent_result=intent_data,
        entities=[entity.model_dump()],
    )
    db_session.add(turn)
    db_session.commit()

    stored_turn = (
        db_session.query(ConversationTurn)
        .filter_by(conversation_id=conv.id, turn_number=1)
        .one()
    )
    assert stored_turn.intent_result["intent_type"] == intent_enum.value
    assert stored_turn.entities[0]["entity_type"] == EntityType.ACCOUNT.value
    # Ensure enums can be reconstructed from stored values
    assert IntentType(stored_turn.intent_result["intent_type"]) is intent_enum
    assert EntityType(stored_turn.entities[0]["entity_type"]) is EntityType.ACCOUNT
