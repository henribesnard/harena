"""Tests for conversation repository operations."""

from datetime import datetime, timezone
from typing import Any, Dict, List

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from db_service.base import Base
from db_service.models import Conversation, ConversationTurn, ConversationSummary


class ConversationRepository:
    """Minimal repository for testing conversation flow."""

    def __init__(self, db: Session):
        self.db = db

    # Creation complete de conversation
    def create_conversation(self, user_id: int, title: str | None = None) -> Conversation:
        conversation = Conversation(user_id=user_id, title=title or "test")
        self.db.add(conversation)
        self.db.commit()
        self.db.refresh(conversation)
        return conversation

    # Ajout de tour avec resultats d'agents
    def add_turn(
        self,
        conversation_id: str,
        user_id: int,
        user_message: str,
        assistant_response: str,
        agent_results: List[Dict[str, Any]] | None = None,
        processing_time_ms: float | None = None,
        confidence_score: float | None = None,
    ) -> ConversationTurn:
        conversation = (
            self.db.query(Conversation)
            .filter_by(conversation_id=conversation_id, user_id=user_id)
            .first()
        )
        if not conversation:
            raise ValueError("Conversation not found or user mismatch")

        turn_number = conversation.total_turns + 1
        turn = ConversationTurn(
            conversation_id=conversation.id,
            turn_number=turn_number,
            user_message=user_message,
            assistant_response=assistant_response,
            processing_time_ms=processing_time_ms,
            confidence_score=confidence_score,
            agent_chain=agent_results or [],
        )
        self.db.add(turn)
        conversation.total_turns = turn_number
        conversation.last_activity_at = datetime.now(timezone.utc)
        self.db.commit()
        self.db.refresh(turn)
        return turn

    # Aggregation des metriques
    def aggregate_metrics(self, conversation_id: str, user_id: int) -> Dict[str, float]:
        conversation = (
            self.db.query(Conversation)
            .filter_by(conversation_id=conversation_id, user_id=user_id)
            .first()
        )
        turns = conversation.turns
        count = len(turns)
        total_processing = sum(t.processing_time_ms or 0 for t in turns)
        total_confidence = sum(t.confidence_score or 0 for t in turns)
        return {
            "average_processing_time_ms": total_processing / count if count else 0.0,
            "average_confidence": total_confidence / count if count else 0.0,
            "total_turns": count,
        }

    # Generation de resume
    def generate_summary(self, conversation_id: str, user_id: int) -> ConversationSummary:
        conversation = (
            self.db.query(Conversation)
            .filter_by(conversation_id=conversation_id, user_id=user_id)
            .first()
        )
        summary_text = " ".join(t.assistant_response for t in conversation.turns)
        summary = ConversationSummary(
            conversation_id=conversation.id,
            start_turn=1,
            end_turn=conversation.total_turns,
            summary_text=summary_text,
            key_topics=[],
            important_entities=[],
        )
        self.db.add(summary)
        self.db.commit()
        self.db.refresh(summary)
        return summary

    # Helper pour verifier isolation user_id
    def get_conversation(self, conversation_id: str, user_id: int) -> Conversation | None:
        return (
            self.db.query(Conversation)
            .filter_by(conversation_id=conversation_id, user_id=user_id)
            .first()
        )


@pytest.fixture()
def db() -> Session:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    TestingSession = sessionmaker(bind=engine)
    session = TestingSession()
    yield session
    session.close()


def test_full_conversation_flow(db: Session) -> None:
    repo = ConversationRepository(db)

    conversation = repo.create_conversation(user_id=1, title="demo")
    assert conversation.total_turns == 0

    repo.add_turn(
        conversation.conversation_id,
        user_id=1,
        user_message="Salut",
        assistant_response="Bonjour",
        agent_results=[{"agent": "intent", "result": "greeting"}],
        processing_time_ms=50.0,
        confidence_score=0.9,
    )

    metrics = repo.aggregate_metrics(conversation.conversation_id, user_id=1)
    assert metrics["average_processing_time_ms"] == 50.0
    assert metrics["average_confidence"] == 0.9
    assert metrics["total_turns"] == 1

    summary = repo.generate_summary(conversation.conversation_id, user_id=1)
    assert "Bonjour" in summary.summary_text


def test_user_isolation_and_data_consistency(db: Session) -> None:
    repo = ConversationRepository(db)

    conv_user1 = repo.create_conversation(user_id=1, title="user1")
    conv_user2 = repo.create_conversation(user_id=2, title="user2")

    repo.add_turn(
        conv_user1.conversation_id,
        user_id=1,
        user_message="Hello",
        assistant_response="Hi",
        agent_results=[],
        processing_time_ms=10.0,
        confidence_score=0.5,
    )

    assert repo.get_conversation(conv_user1.conversation_id, user_id=2) is None

    repo.add_turn(
        conv_user2.conversation_id,
        user_id=2,
        user_message="Yo",
        assistant_response="Yo!",
        agent_results=[],
        processing_time_ms=20.0,
        confidence_score=0.7,
    )

    metrics_user2 = repo.aggregate_metrics(conv_user2.conversation_id, user_id=2)
    assert metrics_user2["total_turns"] == 1
    assert metrics_user2["average_processing_time_ms"] == 20.0
    assert metrics_user2["average_confidence"] == 0.7
from db_service.models import Conversation


def test_conversation_crud(db_session, user):
    # Create
    conv = Conversation(user_id=user.id, conversation_metadata={"topic": "test"})
    db_session.add(conv)
    db_session.commit()
    conv_id = conv.id

    # Read
    fetched = db_session.get(Conversation, conv_id)
    assert fetched is not None
    assert fetched.conversation_metadata["topic"] == "test"

    # Update
    fetched.title = "updated"
    db_session.commit()
    updated = db_session.get(Conversation, conv_id)
    assert updated.title == "updated"

    # Delete
    db_session.delete(updated)
    db_session.commit()
    assert db_session.query(Conversation).count() == 0
