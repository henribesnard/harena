"""
FastAPI dependencies for conversation_service_v3
Provides database sessions and service instances
"""
import logging
from typing import Generator, Optional
from fastapi import Depends, Header, HTTPException
from sqlalchemy.orm import Session
from jose import JWTError, jwt

from ..database import get_db
from ..services.conversation_persistence import ConversationPersistenceService
from ..config.settings import settings

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
    Extract and validate user ID from JWT token

    Args:
        authorization: Authorization header value

    Returns:
        User ID from JWT token, or None if token is invalid/missing

    Raises:
        HTTPException: If token is malformed or invalid
    """
    if not authorization:
        return None

    token = extract_jwt_token(authorization)
    if not token:
        return None

    try:
        # Decode JWT token
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM]
        )

        # Extract user_id from 'sub' claim
        user_id = payload.get("sub")
        if user_id is None:
            logger.warning("JWT token missing 'sub' claim")
            return None

        # Convert to int (sub is usually a string)
        return int(user_id)

    except JWTError as e:
        logger.warning(f"Invalid JWT token: {e}")
        raise HTTPException(
            status_code=401,
            detail=f"Invalid authentication token: {str(e)}"
        )
    except ValueError as e:
        logger.warning(f"Invalid user_id in JWT token: {e}")
        raise HTTPException(
            status_code=401,
            detail="Invalid user_id in token"
        )
