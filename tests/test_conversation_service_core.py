import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import pytest

from db_service.base import Base
from db_service.models.conversation import Conversation
from db_service.models.user import User

from conversation_service.core.conversation_service import save_conversation_turn
from conversation_service.message_repository import ConversationMessageRepository
from conversation_service.models.conversation_models import MessageCreate
from teams.team_orchestrator import TeamOrchestrator


def _setup_db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Session


def test_save_conversation_turn_persists_messages():
    Session = _setup_db()
    with Session() as session:
        user = User(email="u@example.com", password_hash="x")
        session.add(user)
        session.commit()
        session.refresh(user)

        conv = Conversation(user_id=user.id, conversation_id="conv1")
        session.add(conv)
        session.commit()
        session.refresh(conv)

        save_conversation_turn(
            session,
            conversation_db_id=conv.id,
            user_id=user.id,
            user_message="hello",
            agent_messages=[MessageCreate(role="agent", content="{}")],
            assistant_reply="hi",
        )

        repo = ConversationMessageRepository(session)
        messages = repo.list_by_conversation("conv1")
        assert [m.role for m in messages] == ["user", "agent", "assistant"]
        assert messages[0].content == "hello"
        assert messages[2].content == "hi"


@pytest.mark.asyncio
async def test_query_agents_empty_message_raises_value_error():
    Session = _setup_db()
    with Session() as session:
        user = User(email="u@example.com", password_hash="x")
        session.add(user)
        session.commit()
        session.refresh(user)

        orchestrator = TeamOrchestrator()
        conv_id = orchestrator.start_conversation(user.id, session)

        with pytest.raises(ValueError):
            await orchestrator.query_agents(conv_id, "", user.id, session)

