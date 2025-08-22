import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from psycopg2.errors import InFailedSqlTransaction

from db_service.base import Base
from db_service.models.conversation import Conversation, ConversationMessage
from db_service.models.user import User
from conversation_service.core import ConversationService
from conversation_service.message_repository import ConversationMessageRepository
from conversation_service.repository import ConversationRepository
from teams.team_orchestrator import TeamOrchestrator


def _setup_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


def test_start_conversation_rolls_back_on_create_failure(monkeypatch):
    Session = _setup_session()
    with Session() as session:
        user = User(email="u@example.com", password_hash="x")
        session.add(user)
        session.commit()
        session.refresh(user)

        holder = {}

        def failing_create(self, user_id, conv_id):
            conv = Conversation(user_id=user_id, conversation_id=conv_id)
            self._db.add(conv)
            self._db.flush()
            holder["conv"] = conv
            raise RuntimeError("boom")

        monkeypatch.setattr(ConversationRepository, "create", failing_create)

        orchestrator = TeamOrchestrator()
        with pytest.raises(RuntimeError):
            orchestrator.start_conversation(user.id, session)

        assert session.query(Conversation).count() == 0

        # Simuler l'exception InFailedSqlTransaction lors de l'accès à conv.id hors transaction
        def _raise_infailed(self):
            raise InFailedSqlTransaction()

        monkeypatch.setattr(Conversation, "id", property(_raise_infailed))
        with pytest.raises(InFailedSqlTransaction):
            _ = holder["conv"].id


def test_save_conversation_turn_atomic_rolls_back_on_update_failure(monkeypatch):
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

        def add_batch(self, *, conversation_db_id, user_id, messages):
            objs = []
            for m in messages:
                msg = ConversationMessage(
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

        monkeypatch.setattr(ConversationMessageRepository, "add_batch", add_batch)

        from sqlalchemy.sql import expression as sql_expression

        original_execute = session.execute

        def failing_execute(statement, *args, **kwargs):
            if isinstance(statement, sql_expression.Update):
                raise RuntimeError("boom")
            return original_execute(statement, *args, **kwargs)

        monkeypatch.setattr(session, "execute", failing_execute)

        with pytest.raises(RuntimeError):
            svc.save_conversation_turn_atomic(
                conversation=conv,
                user_message="hi",
                assistant_reply="hello",
            )

        assert session.query(ConversationMessage).count() == 0
        session.refresh(conv)
        assert conv.total_turns == 0


def test_access_conv_id_outside_transaction_raises_infailed(monkeypatch):
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

    # La session est maintenant fermée; simuler l'accès hors transaction
    def _raise_infailed(self):
        raise InFailedSqlTransaction()

    monkeypatch.setattr(Conversation, "id", property(_raise_infailed))
    with pytest.raises(InFailedSqlTransaction):
        _ = conv.id
