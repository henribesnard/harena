import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db_service.base import Base
from db_service.models.conversation import (
    Conversation,
    ConversationMessage as ConversationMessageDB,
)
from db_service.models.user import User

from conversation_service.core.conversation_service import ConversationService
from conversation_service.message_repository import ConversationMessageRepository


def _setup_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)


def test_save_conversation_turn_persists_all_messages():
    Session = _setup_session()
    session = Session()
    try:
        user = User(email="u@example.com", password_hash="x")
        session.add(user)
        session.commit()

        conv = Conversation(user_id=user.id, conversation_id="c1")
        session.add(conv)
        session.commit()

        svc = ConversationService(session)

        svc.save_conversation_turn_atomic(
            conversation=conv,
            user_message="hi",
            assistant_reply="hello",
        )

        msgs = ConversationMessageRepository(session).list_models(conv.conversation_id)
        assert [m.role for m in msgs] == ["user", "assistant"]
        session.refresh(conv)
        assert conv.total_turns == 1
    finally:
        session.close()


def test_save_conversation_turn_alias_calls_atomic():
    Session = _setup_session()
    session = Session()
    try:
        user = User(email="u@example.com", password_hash="x")
        session.add(user)
        session.commit()

        conv = Conversation(user_id=user.id, conversation_id="c1")
        session.add(conv)
        session.commit()

        svc = ConversationService(session)

        svc.save_conversation_turn(
            conversation=conv,
            user_message="hi",
            assistant_reply="hello",
        )

        msgs = ConversationMessageRepository(session).list_models(conv.conversation_id)
        assert [m.role for m in msgs] == ["user", "assistant"]
        session.refresh(conv)
        assert conv.total_turns == 1
    finally:
        session.close()

def test_save_conversation_turn_rolls_back_on_failure():
    Session = _setup_session()
    session = Session()
    try:
        user = User(email="u@example.com", password_hash="x")
        session.add(user)
        session.commit()

        conv = Conversation(user_id=user.id, conversation_id="c1")
        session.add(conv)
        session.commit()

        svc = ConversationService(session)

        def _fail(*args, **kwargs):
            raise RuntimeError("boom")

        svc._msg_repo.add_batch = _fail

        with pytest.raises(RuntimeError):
            svc.save_conversation_turn_atomic(
                conversation=conv,
                user_message="hi",
                assistant_reply="hello",
            )

        assert ConversationMessageRepository(session).list_models(conv.conversation_id) == []
        session.refresh(conv)
        assert conv.total_turns == 0
    finally:
        session.close()


def test_save_conversation_turn_rolls_back_on_update_failure():
    Session = _setup_session()
    session = Session()
    try:
        user = User(email="u@example.com", password_hash="x")
        session.add(user)
        session.commit()

        conv = Conversation(user_id=user.id, conversation_id="c1")
        session.add(conv)
        session.commit()

        svc = ConversationService(session)

        original_execute = session.execute

        from sqlalchemy.sql.dml import Update

        def failing_execute(statement, *args, **kwargs):
            if isinstance(statement, Update):
                raise RuntimeError("boom")
            return original_execute(statement, *args, **kwargs)

        session.execute = failing_execute  # type: ignore[assignment]

        with pytest.raises(RuntimeError):
            svc.save_conversation_turn_atomic(
                conversation=conv,
                user_message="hi",
                assistant_reply="hello",
            )

        assert ConversationMessageRepository(session).list_models(conv.conversation_id) == []
        session.refresh(conv)
        assert conv.total_turns == 0
    finally:
        session.close()
