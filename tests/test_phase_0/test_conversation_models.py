import asyncio
import json

from db_service.models import Conversation, ConversationTurn, User
from conversation_service.models.conversation_db_models import (
    Conversation as ConversationSchema,
    ConversationTurn as ConversationTurnSchema,
)


async def _openai_intent(openai_mock):
    result = await openai_mock.chat.completions.create(messages=[])
    return json.loads(result.choices[0].message.content)


def test_conversation_creation_with_metadata(db_session, user):
    conv = Conversation(user_id=user.id, conversation_metadata={"topic": "finance"})
    db_session.add(conv)
    db_session.commit()

    stored = db_session.query(Conversation).first()
    assert stored.conversation_metadata["topic"] == "finance"


def test_turn_records_agent_chain(db_session, user):
    conv = Conversation(user_id=user.id, conversation_metadata={})
    db_session.add(conv)
    db_session.commit()

    turn = ConversationTurn(
        conversation_id=conv.id,
        turn_number=1,
        user_message="hi",
        assistant_response="hello",
        agent_chain=["detector", "responder"],
    )
    db_session.add(turn)
    db_session.commit()

    stored_turn = db_session.query(ConversationTurn).first()
    assert stored_turn.agent_chain == ["detector", "responder"]


def test_turn_openai_tracking(db_session, user, openai_mock):
    conv = Conversation(user_id=user.id, conversation_metadata={})
    db_session.add(conv)
    db_session.commit()

    intent = asyncio.run(_openai_intent(openai_mock))

    turn = ConversationTurn(
        conversation_id=conv.id,
        turn_number=1,
        user_message="hi",
        assistant_response="hello",
        intent_result=intent,
    )
    db_session.add(turn)
    db_session.commit()

    stored = db_session.query(ConversationTurn).first()
    assert stored.intent_result["intent_type"] == "GREETING"


def test_conversation_cache_hit(cache, db_session, user):
    conv = Conversation(user_id=user.id, conversation_metadata={"lang": "fr"})
    db_session.add(conv)
    db_session.commit()

    key = f"{user.id}:{conv.conversation_id}"
    cache.set(key, conv.conversation_metadata)

    assert cache.get(key) == {"lang": "fr"}
    # Second access should still return the value (cache hit)
    assert cache.get(key) == {"lang": "fr"}


def test_cache_isolation_by_user_id(cache, db_session, user):
    conv = Conversation(user_id=user.id, conversation_metadata={})
    db_session.add(conv)
    db_session.commit()

    key_user1 = f"{user.id}:{conv.conversation_id}"
    cache.set(key_user1, {"foo": "bar"})

    other = User(email="other@example.com", password_hash="h")
    db_session.add(other)
    db_session.commit()

    key_user2 = f"{other.id}:{conv.conversation_id}"
    assert cache.get(key_user2) is None


def test_ai_metadata_persistence(db_session, user):
    conv = Conversation(
        user_id=user.id,
        conversation_metadata={},
        intents=[{"type": "GREETING"}],
        entities=[{"symbol": "AAPL"}],
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
    )
    db_session.add(conv)
    db_session.commit()

    turn = ConversationTurn(
        conversation_id=conv.id,
        turn_number=1,
        user_message="hello",
        assistant_response="hi",
        intent={"type": "GREETING"},
        entities=[{"symbol": "AAPL"}],
        prompt_tokens=3,
        completion_tokens=2,
        total_tokens=5,
    )
    db_session.add(turn)
    db_session.commit()

    stored_conv = db_session.get(Conversation, conv.id)
    assert stored_conv.intents == [{"type": "GREETING"}]
    assert stored_conv.entities == [{"symbol": "AAPL"}]
    assert stored_conv.total_tokens == 15

    stored_turn = (
        db_session.query(ConversationTurn)
        .filter_by(conversation_id=conv.id)
        .first()
    )
    assert stored_turn.intent == {"type": "GREETING"}
    assert stored_turn.entities == [{"symbol": "AAPL"}]
    assert stored_turn.total_tokens == 5

    conv_model = ConversationSchema(
        id=stored_conv.id,
        conversation_id=stored_conv.conversation_id,
        user_id=stored_conv.user_id,
        title=stored_conv.title,
        status=stored_conv.status,
        language=stored_conv.language,
        domain=stored_conv.domain,
        total_turns=stored_conv.total_turns,
        max_turns=stored_conv.max_turns,
        last_activity_at=stored_conv.last_activity_at,
        conversation_metadata=stored_conv.conversation_metadata,
        user_preferences=stored_conv.user_preferences,
        session_metadata=stored_conv.session_metadata,
        intents=stored_conv.intents,
        entities=stored_conv.entities,
        prompt_tokens=stored_conv.prompt_tokens,
        completion_tokens=stored_conv.completion_tokens,
        total_tokens=stored_conv.total_tokens,
        created_at=stored_conv.created_at,
        updated_at=stored_conv.updated_at,
    )
    assert conv_model.intents == [{"type": "GREETING"}]
    assert conv_model.total_tokens == 15

    turn_model = ConversationTurnSchema(
        id=stored_turn.id,
        turn_id=stored_turn.turn_id,
        conversation_id=stored_turn.conversation_id,
        turn_number=stored_turn.turn_number,
        user_message=stored_turn.user_message,
        assistant_response=stored_turn.assistant_response,
        processing_time_ms=stored_turn.processing_time_ms,
        confidence_score=stored_turn.confidence_score,
        error_occurred=stored_turn.error_occurred,
        error_message=stored_turn.error_message,
        intent_result=stored_turn.intent_result,
        agent_chain=stored_turn.agent_chain,
        intent=stored_turn.intent,
        entities=stored_turn.entities,
        prompt_tokens=stored_turn.prompt_tokens,
        completion_tokens=stored_turn.completion_tokens,
        total_tokens=stored_turn.total_tokens,
        search_query_used=stored_turn.search_query_used,
        search_results_count=stored_turn.search_results_count,
        search_execution_time_ms=stored_turn.search_execution_time_ms,
        turn_metadata=stored_turn.turn_metadata,
        created_at=stored_turn.created_at,
        updated_at=stored_turn.updated_at,
    )
    assert turn_model.intent["type"] == "GREETING"
    assert turn_model.total_tokens == 5
