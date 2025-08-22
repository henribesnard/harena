import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db_service.base import Base
from db_service.models.user import User
from conversation_service.message_repository import ConversationMessageRepository
from teams.team_orchestrator import TeamOrchestrator


class DummyResponder:
    name = "dummy_responder"

    async def process(self, payload):
        return {"response": "pong"}


@pytest.mark.asyncio
async def test_query_agents_saves_full_turn():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)

    with Session() as session:
        user = User(email="u@example.com", password_hash="x")
        session.add(user)
        session.commit()
        session.refresh(user)

        team = TeamOrchestrator(responder=DummyResponder())
        conv_id = team.start_conversation(user.id, session)
        await team.query_agents(conv_id, "ping", user.id, session)

        repo = ConversationMessageRepository(session)
        roles = [m.role for m in repo.list_by_conversation(conv_id)]
        assert roles == ["user", "agent", "assistant"]


@pytest.mark.asyncio
async def test_query_agents_rollback_on_save_failure():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)

    with Session() as session:
        user = User(email="u@example.com", password_hash="x")
        session.add(user)
        session.commit()
        session.refresh(user)

        team = TeamOrchestrator(responder=DummyResponder())
        conv_id = team.start_conversation(user.id, session)
        with pytest.raises(ValueError):
            await team.query_agents(conv_id, "", user.id, session)

        repo = ConversationMessageRepository(session)
        assert repo.list_by_conversation(conv_id) == []
