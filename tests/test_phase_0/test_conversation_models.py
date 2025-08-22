import asyncio
import json

from db_service.models import Conversation, ConversationTurn, User


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
