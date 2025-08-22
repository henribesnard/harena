import os

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from db_service.models.conversation import ConversationMessage as ConversationMessageDB

os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")

from db_service.base import Base
from db_service.models.user import User
from conversation_service.api.routes import router
from conversation_service.repository import ConversationRepository
from db_service.session import get_db
from user_service.api.deps import get_current_active_user


@pytest.fixture
def app_and_session():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)

    with SessionLocal() as session:
        user = User(email="u@example.com", password_hash="x")
        session.add(user)
        session.commit()
        session.refresh(user)

    app = FastAPI()
    app.include_router(router, prefix="/conversation")

    def override_get_db():
        with SessionLocal() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_current_active_user] = lambda: user

    return app, SessionLocal, user


def test_get_history_returns_messages(app_and_session):
    app, SessionLocal, user = app_and_session
    with SessionLocal() as session:
        conv = ConversationRepository(session).create(user.id, "c1")
        session.commit()
        session.add_all(
            [
                ConversationMessageDB(
                    conversation_id=conv.id,
                    user_id=user.id,
                    role="user",
                    content="hi",
                ),
                ConversationMessageDB(
                    conversation_id=conv.id,
                    user_id=user.id,
                    role="assistant",
                    content="hello",
                ),
            ]
        )
        session.commit()
        conv_id = conv.conversation_id

    client = TestClient(app)
    resp = client.get(f"/conversation/{conv_id}/history")
    assert resp.status_code == 200
    data = resp.json()
    assert data["conversation_id"] == conv_id
    assert [m["role"] for m in data["messages"]] == ["user", "assistant"]


def test_get_history_unknown_conversation_returns_404(app_and_session):
    app, _, _ = app_and_session
    client = TestClient(app)
    resp = client.get("/conversation/unknown/history")
    assert resp.status_code == 404
