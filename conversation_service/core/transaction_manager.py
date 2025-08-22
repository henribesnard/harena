"""Simple context manager for SQLAlchemy transactions."""
from __future__ import annotations

from contextlib import contextmanager
from sqlalchemy.orm import Session

@contextmanager
def transaction_manager(db: Session):
    """Provide a transactional scope for database operations.

    Commits the transaction if the enclosed block succeeds, otherwise rolls
    back and re-raises the exception."""
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise

__all__ = ["transaction_manager"]
