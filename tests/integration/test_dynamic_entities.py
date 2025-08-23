"""Integration tests for dynamic entity fields in conversation turns."""

from sqlalchemy import Column, Float, JSON, create_engine
from sqlalchemy.orm import Session, sessionmaker

from db_service.base import Base
from db_service.models.conversation import ConversationTurn as ConversationTurnORM
from conversation_service.repositories.conversation_repository import ConversationRepository
from conversation_service.schemas import (
    ConversationCreate,
    ConversationTurnCreate,
    ConversationTurn,
)


def _add_dynamic_columns() -> None:
    """Inject dynamic columns into the ORM for testing."""
    if "openai_usage_stats" not in ConversationTurnORM.__table__.columns:
        openai_usage_stats_col = Column("openai_usage_stats", JSON, nullable=True, default=dict)
        ConversationTurnORM.__table__.append_column(openai_usage_stats_col)
        ConversationTurnORM.openai_usage_stats = openai_usage_stats_col
        ConversationTurnORM.__mapper__.add_property("openai_usage_stats", ConversationTurnORM.openai_usage_stats)
    if "openai_cost_usd" not in ConversationTurnORM.__table__.columns:
        openai_cost_usd_col = Column("openai_cost_usd", Float, nullable=True, default=0.0)
        ConversationTurnORM.__table__.append_column(openai_cost_usd_col)
        ConversationTurnORM.openai_cost_usd = openai_cost_usd_col
        ConversationTurnORM.__mapper__.add_property("openai_cost_usd", ConversationTurnORM.openai_cost_usd)


def test_add_turn_with_entities() -> None:
    _add_dynamic_columns()

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
    raw_data = {k: v for k, v in db_turn.__dict__.items() if not k.startswith("_")}
    stored_turn = ConversationTurn.model_validate(raw_data)

    assert stored_turn.entities_extracted == [{"name": "Paris"}]
    assert stored_turn.intent_classification == {"intent": "greet"}
    assert stored_turn.intent_confidence == 0.87
    assert stored_turn.openai_usage_stats == {"prompt_tokens": 3, "completion_tokens": 2}
    assert stored_turn.total_tokens_used == 5
    assert stored_turn.openai_cost_usd == 0.01
