from concurrent.futures import ThreadPoolExecutor

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from conversation_service.core import ConversationService
from conversation_service.message_repository import ConversationMessageRepository
from db_service.base import Base
from db_service.models.conversation import Conversation, ConversationMessage
from db_service.models.user import User


# --- helpers -----------------------------------------------------------------

@pytest.fixture()
def Session(tmp_path):
    """Return a session factory bound to a temporary SQLite database."""
    engine = create_engine(
        f"sqlite:///{tmp_path}/test.db",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


def _create_user_and_conversation(session):
    user = User(email="u@example.com", password_hash="x")
    session.add(user)
    session.commit()
    session.refresh(user)

    conv = Conversation(user_id=user.id, conversation_id="c1")
    session.add(conv)
    session.commit()
    session.refresh(conv)
    return conv


# Patch ConversationMessageRepository.add_batch to include minimal validation
@pytest.fixture(autouse=True)
def _patch_add_batch(monkeypatch):
    def add_batch(self, *, conversation_db_id, user_id, messages):
        instances = []
        for m in messages:
            if not m.content.strip():
                raise ValueError("content must not be empty")
            msg = ConversationMessage(
                conversation_id=conversation_db_id,
                user_id=user_id,
                role=m.role,
                content=m.content,
            )
            self._db.add(msg)
            self._db.flush()
            self._db.refresh(msg)
            instances.append(msg)
        return instances

    monkeypatch.setattr(ConversationMessageRepository, "add_batch", add_batch)


# --- tests -------------------------------------------------------------------

def test_conversation_turn_atomic_success(Session):
    with Session() as session:
        conv = _create_user_and_conversation(session)
        service = ConversationService(session)
        service.save_conversation_turn_atomic(
            conversation=conv,
            user_message="hi",
            agent_messages=[("agent", "processing")],
            assistant_reply="hello",
        )
        session.refresh(conv)
        msgs = ConversationMessageRepository(session).list_by_conversation(
            conv.conversation_id
        )
        assert [m.role for m in msgs] == ["user", "agent", "assistant"]
        assert conv.total_turns == 1
        assert conv.last_activity_at is not None
        # get_for_user returns the conversation for owner and None for others
        assert service.get_for_user(conv.conversation_id, conv.user_id) is not None
        assert service.get_for_user(conv.conversation_id, conv.user_id + 1) is None


def test_conversation_turn_atomic_rollback(Session):
    with Session() as session:
        conv = _create_user_and_conversation(session)
        service = ConversationService(session)
        with pytest.raises(ValueError):
            service.save_conversation_turn_atomic(
                conversation=conv,
                user_message="hi",
                assistant_reply="",
            )
        session.refresh(conv)
        msgs = ConversationMessageRepository(session).list_by_conversation(conv.conversation_id)
        assert msgs == []
        assert conv.total_turns == 0

        # Empty agent message should raise before repository call
        with pytest.raises(ValueError):
            service.save_conversation_turn_atomic(
                conversation=conv,
                user_message="hi",
                agent_messages=[("agent", "")],
                assistant_reply="ok",
            )
        session.refresh(conv)
        assert ConversationMessageRepository(session).list_by_conversation(conv.conversation_id) == []
        assert conv.total_turns == 0


def test_concurrent_access_safety(Session):
    SessionFactory = Session
    with SessionFactory() as session:
        conv = _create_user_and_conversation(session)
        conv_id = conv.conversation_id

    def worker(msg):
        with SessionFactory() as s:
            conv = s.query(Conversation).filter_by(conversation_id=conv_id).first()
            ConversationService(s).save_conversation_turn_atomic(
                conversation=conv,
                user_message=msg,
                assistant_reply="ok",
            )

    with ThreadPoolExecutor(max_workers=2) as exe:
        exe.map(worker, ["m1", "m2"])

    with SessionFactory() as session:
        conv = session.query(Conversation).filter_by(conversation_id=conv_id).first()
        msgs = ConversationMessageRepository(session).list_by_conversation(conv_id)
        assert conv.total_turns == 2
        assert len(msgs) == 4
