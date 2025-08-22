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
    return sessionmaker(bind=engine)


def test_save_conversation_turn_persists_all_messages():
    Session = _setup_session()
    with Session() as session:
        user = User(email="u@example.com", password_hash="x")
        session.add(user)
        session.commit()
        session.refresh(user)

        conv = Conversation(user_id=user.id, conversation_id="c1")
        session.add(conv)
        session.commit()
        session.refresh(conv)

        svc = ConversationService(session)

        def _add_batch(self, *, conversation_db_id, user_id, messages):
            objs = []
            for m in messages:
                msg = ConversationMessageDB(
                    conversation_id=conversation_db_id,
                    user_id=user_id,
                    role=m.role,
                    content=m.content,
                )
                self._db.add(msg)
                self._db.flush()
                self._db.refresh(msg)
                objs.append(msg)
            return objs

        svc._msg_repo.add_batch = _add_batch.__get__(svc._msg_repo, type(svc._msg_repo))

        svc.save_conversation_turn_atomic(
            conversation=conv,
            user_message="hi",
            assistant_reply="hello",
        )

        msgs = ConversationMessageRepository(session).list_models(conv.conversation_id)
        assert [m.role for m in msgs] == ["user", "assistant"]
        session.refresh(conv)
        assert conv.total_turns == 1


def test_save_conversation_turn_alias_calls_atomic():
    Session = _setup_session()
    with Session() as session:
        user = User(email="u@example.com", password_hash="x")
        session.add(user)
        session.commit()
        session.refresh(user)

        conv = Conversation(user_id=user.id, conversation_id="c1")
        session.add(conv)
        session.commit()
        session.refresh(conv)

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

def test_save_conversation_turn_rolls_back_on_failure():
    Session = _setup_session()
    with Session() as session:
        user = User(email="u@example.com", password_hash="x")
        session.add(user)
        session.commit()
        session.refresh(user)

        conv = Conversation(user_id=user.id, conversation_id="c1")
        session.add(conv)
        session.commit()
        session.refresh(conv)

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
