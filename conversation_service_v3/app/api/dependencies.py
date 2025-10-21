"""
FastAPI dependencies for conversation_service_v3
Provides database sessions and service instances
"""
import logging
from typing import Generator, Optional
from fastapi import Depends, Header, HTTPException
from sqlalchemy.orm import Session

from ..database import get_db
from ..services.conversation_persistence import ConversationPersistenceService

logger = logging.getLogger(__name__)


def get_persistence_service(
    db: Session = Depends(get_db)
) -> ConversationPersistenceService:
    """
    Dependency pour obtenir le service de persistence

    Args:
        db: Session de base de données (injectée par FastAPI)

    Returns:
        Instance de ConversationPersistenceService
    """
    return ConversationPersistenceService(db)


def extract_jwt_token(authorization: Optional[str] = Header(None)) -> Optional[str]:
    """
    Extract JWT token from Authorization header

    Args:
        authorization: Authorization header value

    Returns:
        JWT token without 'Bearer ' prefix, or None if not found
    """
    if authorization and authorization.startswith("Bearer "):
        return authorization.replace("Bearer ", "")
    return None


def get_current_user_id(
    authorization: Optional[str] = Header(None)
) -> Optional[int]:
    """
    Extract user ID from JWT token

    Note: This is a simplified version. In production, you should:
    1. Decode the JWT token
    2. Validate the signature
    3. Check expiration
    4. Extract user_id from claims

    For now, we'll rely on the user_id in the path parameter
    """
    # TODO: Implement proper JWT validation
    return None
