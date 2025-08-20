"""Minimal dependency definitions for the conversation service.

The real project contains a rich set of dependencies for authentication,
metrics, database access and agent management.  For the purposes of the kata we
only implement lightweight placeholders so that the API can be exercised in
isolation and tests can override these dependencies.
"""

from typing import Any, Dict, Generator

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from db_service.session import SessionLocal
from ..models.conversation_models import ConversationRequest
from ..utils.metrics import MetricsCollector

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
_metrics_collector = MetricsCollector()


def get_metrics_collector() -> MetricsCollector:
    """Return a singleton metrics collector."""
    return _metrics_collector


# ---------------------------------------------------------------------------
# Database session management
# ---------------------------------------------------------------------------

def get_db() -> Generator[Session, None, None]:
    """Yield a database session and ensure it is closed afterwards."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Placeholder dependencies used in tests.  They intentionally raise errors so
# that tests must override them with fake implementations.
# ---------------------------------------------------------------------------


async def get_team_manager() -> Any:
    raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Team manager not configured")


async def get_conversation_manager() -> Any:
    raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Conversation manager not configured")


async def get_conversation_service() -> Any:
    class Dummy:
        def get_or_create_conversation(self, user_id: int, conversation_id: str | None):
            class Conv:
                def __init__(self, cid: str):
                    self.conversation_id = cid
            return Conv(conversation_id or "conv-1")

        def add_turn(self, **kwargs: Any) -> None:
            pass

    return Dummy()


async def get_conversation_read_service() -> Any:
    class Dummy:
        def get_conversations(self, user_id: int, limit: int = 10, offset: int = 0):
            return []

    return Dummy()


async def get_current_user() -> Dict[str, Any]:
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")


async def validate_conversation_request(request: ConversationRequest) -> ConversationRequest:
    return request


async def validate_request_rate_limit() -> None:
    return None


async def cleanup_dependencies() -> None:
    """Cleanup hook called on application shutdown."""
    return None
