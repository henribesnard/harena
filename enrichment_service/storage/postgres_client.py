"""PostgreSQL client for the enrichment service using shared DB models."""
from typing import Generator

from db_service.base import Base
from db_service.models.sync import (
    SyncAccount as SyncAccountModel,
    Category as CategoryModel,
    RawTransaction as RawTransactionModel,
)  # noqa: F401
from db_service.session import SessionLocal, engine


def init_db() -> None:
    """Create database tables based on shared Base metadata."""
    Base.metadata.create_all(bind=engine)


def get_db() -> Generator:
    """Yield a database session and ensure it is closed afterwards."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
