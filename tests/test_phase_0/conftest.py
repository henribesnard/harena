import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db_service.base import Base
from db_service.models import User


@pytest.fixture
def db_session():
    """Provide a temporary in-memory database session."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(engine)


@pytest.fixture
def user(db_session):
    """Create a default user for tests."""
    u = User(email="user@example.com", password_hash="hash")
    db_session.add(u)
    db_session.commit()
    return u
