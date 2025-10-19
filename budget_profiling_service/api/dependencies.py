"""
Dépendances FastAPI pour le budget profiling service
"""
from typing import Generator
from sqlalchemy.orm import Session
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from db_service.session import SessionLocal
from budget_profiling_service.api.middleware.auth_middleware import JWTValidator

# Security scheme
security = HTTPBearer()

# JWT Validator global
jwt_validator = JWTValidator()


def get_db() -> Generator[Session, None, None]:
    """
    Dependency pour obtenir une session DB
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_current_user_id(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> int:
    """
    Extrait et valide le user_id depuis le token JWT

    Returns:
        user_id: ID de l'utilisateur authentifié

    Raises:
        HTTPException: Si le token est invalide ou expiré
    """
    token = credentials.credentials

    # Valider le token
    auth_result = jwt_validator.validate_token(token)

    if not auth_result.success:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=auth_result.error_message or "Token invalide",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not auth_result.user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User ID manquant dans le token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return auth_result.user_id
