"""Integration tests for dynamic entity fields in conversation turns."""

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from db_service.base import Base
from db_service.models.conversation import ConversationTurn as ConversationTurnORM
from conversation_service.repositories.conversation_repository import ConversationRepository
from conversation_service.schemas import (
    ConversationCreate,
    ConversationTurnCreate,
    ConversationTurn,
)


def test_add_turn_with_entities() -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine)
    db: Session = SessionLocal()

    repo = ConversationRepository(db)
    conv = repo.create(ConversationCreate(user_id=1, title="demo"))

    turn_in = ConversationTurnCreate(
        user_message="Bonjour",
        assistant_response="Salut",
        entities_extracted=[{"name": "Paris"}],
        intent_classification={"intent": "greet"},
        intent_confidence=0.87,
        openai_usage_stats={"prompt_tokens": 3, "completion_tokens": 2},
        total_tokens_used=5,
        openai_cost_usd=0.01,
    )

    repo.add_turn(conv.conversation_id, turn_in)

    db_turn = db.query(ConversationTurnORM).filter_by(turn_number=1).first()
    stored_turn = ConversationTurn.model_validate(db_turn, from_attributes=True)

    assert stored_turn.entities_extracted == [{"name": "Paris"}]
    assert stored_turn.intent_classification == {"intent": "greet"}
    assert stored_turn.intent_confidence == 0.87
    assert stored_turn.openai_usage_stats == {"prompt_tokens": 3, "completion_tokens": 2}
    assert stored_turn.total_tokens_used == 5
    assert stored_turn.openai_cost_usd == 0.01
