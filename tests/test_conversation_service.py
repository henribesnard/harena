import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db_service.base import Base
from db_service.models.conversation import Conversation
from db_service.models.user import User
from conversation_service.message_repository import ConversationMessageRepository
from conversation_service.models.conversation_models import MessageCreate
from conversation_service.service import ConversationService


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

        repo = ConversationMessageRepository(session)
        svc = ConversationService(repo)
        svc.save_conversation_turn(
            conversation_db_id=conv.id,
            user_id=user.id,
            messages=[
                MessageCreate(role="user", content="hi"),
                MessageCreate(role="agent", content="processing"),
                MessageCreate(role="assistant", content="hello"),
            ],
        )

        messages = repo.list_by_conversation(conv.conversation_id)
        assert [m.role for m in messages] == ["user", "agent", "assistant"]


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

        repo = ConversationMessageRepository(session)
        svc = ConversationService(repo)
        with pytest.raises(ValueError):
            svc.save_conversation_turn(
                conversation_db_id=conv.id,
                user_id=user.id,
                messages=[
                    MessageCreate(role="user", content="hi"),
                    MessageCreate(role="agent", content="processing"),
                    MessageCreate(role="assistant", content=""),
                ],
            )

        assert repo.list_by_conversation(conv.conversation_id) == []
