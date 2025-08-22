import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db_service.base import Base
from db_service.models.conversation import Conversation
from db_service.models.user import User
from conversation_service.message_repository import ConversationMessageRepository
from conversation_service.models.conversation_models import MessageCreate


def test_add_and_list_messages_with_int_conversation_id():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)

    with Session() as session:
        user = User(email="u@example.com", password_hash="x")
        session.add(user)
        session.commit()
        session.refresh(user)

        conv = Conversation(user_id=user.id, conversation_id="conv1")
        session.add(conv)
        session.commit()
        session.refresh(conv)

        repo = ConversationMessageRepository(session)
        repo.add(
            conversation_db_id=conv.id,
            user_id=user.id,
            role="user",
            content="hello",
        )

        messages = repo.list_models(conv.conversation_id)
        assert len(messages) == 1
        assert messages[0].conversation_id == conv.conversation_id
        assert messages[0].content == "hello"


def test_add_raises_on_invalid_ids():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)

    with Session() as session:
        user = User(email="u@example.com", password_hash="x")
        session.add(user)
        session.commit()
        session.refresh(user)

        repo = ConversationMessageRepository(session)
        with pytest.raises(ValueError):
            repo.add(
                conversation_db_id=0,
                user_id=user.id,
                role="user",
                content="hi",
            )


def test_add_raises_on_empty_content():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)

    with Session() as session:
        user = User(email="u@example.com", password_hash="x")
        session.add(user)
        session.commit()
        session.refresh(user)

        conv = Conversation(user_id=user.id, conversation_id="c1")
        session.add(conv)
        session.commit()
        session.refresh(conv)

        repo = ConversationMessageRepository(session)
        with pytest.raises(ValueError):
            repo.add(
                conversation_db_id=conv.id,
                user_id=user.id,
                role="user",
                content=" ",
            )


def test_add_batch_inserts_messages_atomically():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)

    with Session() as session:
        user = User(email="u@example.com", password_hash="x")
        session.add(user)
        session.commit()
        session.refresh(user)

        conv = Conversation(user_id=user.id, conversation_id="c1")
        session.add(conv)
        session.commit()
        session.refresh(conv)

        repo = ConversationMessageRepository(session)
        repo.add_batch(
            conversation_db_id=conv.id,
            user_id=user.id,
            messages=[
                MessageCreate(role="user", content="hi"),
                MessageCreate(role="assistant", content="hello"),
            ],
        )

        messages = repo.list_models(conv.conversation_id)
        assert [m.role for m in messages] == ["user", "assistant"]


def test_add_batch_rolls_back_on_failure():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)

    with Session() as session:
        user = User(email="u@example.com", password_hash="x")
        session.add(user)
        session.commit()
        session.refresh(user)

        conv = Conversation(user_id=user.id, conversation_id="c1")
        session.add(conv)
        session.commit()
        session.refresh(conv)

        repo = ConversationMessageRepository(session)
        with pytest.raises(ValueError):
            repo.add_batch(
                conversation_db_id=conv.id,
                user_id=user.id,
                messages=[
                    MessageCreate(role="user", content="hi"),
                    MessageCreate(role="assistant", content=""),
                ],
            )

        messages = repo.list_models(conv.conversation_id)
        assert messages == []
