import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db_service.base import Base
from db_service.models.conversation import Conversation
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
        svc.save_conversation_turn_atomic(
            conversation=conv,
            user_message="hi",
            assistant_reply="hello",
        )

        msgs = ConversationMessageRepository(session).list_models(conv.conversation_id)
        assert [m.role for m in msgs] == ["user", "assistant"]


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
        with pytest.raises(ValueError):
            svc.save_conversation_turn_atomic(
                conversation=conv,
                user_message="hi",
                assistant_reply="",
            )

        assert ConversationMessageRepository(session).list_models(conv.conversation_id) == []
