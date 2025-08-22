from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import pytest

from db_service.base import Base
from db_service.models.conversation import Conversation
from db_service.models.user import User
from conversation_service.message_repository import ConversationMessageRepository


def setup_data(session):
    user = User(email="u@example.com", password_hash="x")
    session.add(user)
    session.commit()
    session.refresh(user)

    conv = Conversation(user_id=user.id, conversation_id="conv1")
    session.add(conv)
    session.commit()
    session.refresh(conv)

    return user.id, conv.id, conv.conversation_id


def test_add_and_list_messages_with_int_conversation_id():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)

    with Session() as session:
        user_id, conv_db_id, conv_id = setup_data(session)

    repo = ConversationMessageRepository(Session())
    repo.add(
        conversation_db_id=conv_db_id,
        user_id=user_id,
        role="user",
        content="hello",
    )

    with Session() as verify_session:
        repo_verify = ConversationMessageRepository(verify_session)
        messages = repo_verify.list_models(conv_id)

    assert len(messages) == 1
    assert messages[0].conversation_id == conv_id
    assert messages[0].content == "hello"


def test_add_validation():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)

    with Session() as session:
        user_id, conv_db_id, _ = setup_data(session)

    with Session() as db:
        repo = ConversationMessageRepository(db)
        with pytest.raises(ValueError):
            repo.add(
                conversation_db_id=-1,
                user_id=user_id,
                role="user",
                content="test",
            )

    with Session() as db:
        repo = ConversationMessageRepository(db)
        with pytest.raises(ValueError):
            repo.add(
                conversation_db_id=conv_db_id,
                user_id=user_id,
                role="user",
                content="",
            )


def test_add_batch_and_rollback_on_error():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)

    with Session() as session:
        user_id, conv_db_id, conv_id = setup_data(session)

    repo = ConversationMessageRepository(Session())
    repo.add_batch(
        [
            {
                "conversation_db_id": conv_db_id,
                "user_id": user_id,
                "role": "user",
                "content": "hello",
            },
            {
                "conversation_db_id": conv_db_id,
                "user_id": user_id,
                "role": "assistant",
                "content": "hi",
            },
        ]
    )

    with Session() as session:
        repo_verify = ConversationMessageRepository(session)
        messages = repo_verify.list_models(conv_id)
        assert [m.content for m in messages] == ["hello", "hi"]

    with Session() as db:
        repo = ConversationMessageRepository(db)
        with pytest.raises(ValueError):
            repo.add_batch(
                [
                    {
                        "conversation_db_id": conv_db_id,
                        "user_id": user_id,
                        "role": "user",
                        "content": "ok",
                    },
                    {
                        "conversation_db_id": -1,
                        "user_id": user_id,
                        "role": "assistant",
                        "content": "bad",
                    },
                ]
            )

    with Session() as session:
        repo_verify = ConversationMessageRepository(session)
        messages = repo_verify.list_models(conv_id)
        # previous batch should not have added any new messages
        assert [m.content for m in messages] == ["hello", "hi"]

