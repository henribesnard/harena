# search_service/api/deps.py
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError
from sqlalchemy.orm import Session
from typing import Optional, List

from db_service.session import get_db
from db_service.models.user import User
from user_service.services.users import get_user_by_id
from config_service.config import settings
from user_service.core.security import ALGORITHM
from user_service.schemas.user import TokenData

oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{settings.API_V1_STR}/auth/login")


async def get_current_user(
    db: Session = Depends(get_db),
    token: str = Depends(oauth2_scheme)
) -> User:
    """
    Authentifie l'utilisateur et retourne ses informations.
    Utilisé par search_service pour valider les tokens JWT.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(
            token, settings.SECRET_KEY, algorithms=[ALGORITHM]
        )
        user_id: Optional[str] = payload.get("sub")
        permissions: List[str] = payload.get("permissions", [])
        if user_id is None:
            raise credentials_exception
        token_data = TokenData(user_id=int(user_id), permissions=permissions)
    except JWTError:
        raise credentials_exception
    
    user = get_user_by_id(db, user_id=token_data.user_id)
    if user is None:
        raise credentials_exception
    # attach permissions from token
    setattr(user, "permissions", token_data.permissions)
    return user


async def get_current_active_user(
    current_user: User = Depends(get_current_user),
) -> User:
    """
    Vérifie que l'utilisateur authentifié est actif.
    """
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


def validate_user_access(current_user: User, request_user_id: int) -> None:
    """
    Valide que l'utilisateur connecté peut accéder aux données du user_id demandé.
    
    Règles:
    - Un utilisateur normal ne peut accéder qu'à ses propres données (current_user.id == request_user_id)
    - Un admin (is_superuser=True) peut accéder à toutes les données
    
    Args:
        current_user: Utilisateur connecté (depuis le token JWT)
        request_user_id: ID utilisateur demandé dans la requête
        
    Raises:
        HTTPException: Si l'accès n'est pas autorisé
    """
    # Admin peut tout voir
    if current_user.is_superuser:
        return
    
    # Utilisateur normal ne peut voir que ses propres données
    if current_user.id != request_user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Access denied: User {current_user.id} cannot access data for user {request_user_id}"
        )