import logging
from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db_service.base import Base
from db_service.models.conversation import (
    Conversation,
    ConversationMessage as ConversationMessageDB,
)
from db_service.models.user import User
from conversation_service.message_repository import ConversationMessageRepository
from conversation_service.models.conversation_models import MessageCreate


def create_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


def prepare(session):
    user = User(email="u@example.com", password_hash="x")
    session.add(user)
    session.commit()
    session.refresh(user)
    conv = Conversation(user_id=user.id, conversation_id="conv1")
    session.add(conv)
    session.commit()
    session.refresh(conv)
    return user.id, conv.id, conv.conversation_id


def test_add_batch_inserts_messages_atomically():
    Session = create_session()
    with Session() as s:
        user_id, conv_db_id, conv_id = prepare(s)
        repo = ConversationMessageRepository(s)
        repo.add_batch(
            conversation_db_id=conv_db_id,
            user_id=user_id,
            messages=[
                MessageCreate(role="user", content="hi"),
                MessageCreate(role="assistant", content="hello"),
            ],
        )
        msgs = repo.list_models(conv_id)
        assert [m.role for m in msgs] == ["user", "assistant"]


def test_add_batch_rolls_back_on_failure(caplog):
    Session = create_session()
    with Session() as s:
        user_id, conv_db_id, conv_id = prepare(s)
        repo = ConversationMessageRepository(s)
        with pytest.raises(ValueError):
            with caplog.at_level(logging.ERROR):
                repo.add_batch(
                    conversation_db_id=conv_db_id,
                    user_id=user_id,
                    messages=[
                        MessageCreate(role="user", content="hi"),
                        MessageCreate(role="assistant", content=""),
                    ],
                )
        assert repo.list_models(conv_id) == []


def test_list_by_conversation_orders_messages_chronologically():
    Session = create_session()
    with Session() as s:
        user_id, conv_db_id, conv_id = prepare(s)
        repo = ConversationMessageRepository(s)

        later = ConversationMessageDB(
            conversation_id=conv_db_id,
            user_id=user_id,
            role="user",
            content="later",
        )
        later.created_at = datetime(2024, 1, 2, tzinfo=timezone.utc)

        earlier = ConversationMessageDB(
            conversation_id=conv_db_id,
            user_id=user_id,
            role="assistant",
            content="earlier",
        )
        earlier.created_at = datetime(2024, 1, 1, tzinfo=timezone.utc)

        s.add_all([later, earlier])
        s.commit()

        msgs = repo.list_by_conversation(conv_id)
        assert [m.content for m in msgs] == ["earlier", "later"]
