from db_service.models import Conversation


def test_conversation_crud(db_session, user):
    # Create
    conv = Conversation(user_id=user.id, conversation_metadata={"topic": "test"})
    db_session.add(conv)
    db_session.commit()
    conv_id = conv.id

    # Read
    fetched = db_session.get(Conversation, conv_id)
    assert fetched is not None
    assert fetched.conversation_metadata["topic"] == "test"

    # Update
    fetched.title = "updated"
    db_session.commit()
    updated = db_session.get(Conversation, conv_id)
    assert updated.title == "updated"

    # Delete
    db_session.delete(updated)
    db_session.commit()
    assert db_session.query(Conversation).count() == 0
