# user_service/api/deps.py
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError
from sqlalchemy.orm import Session
from typing import Optional, List

from conversation_service.api.dependencies import get_db
from db_service.models.user import User
from user_service.services.users import get_user_by_id
from config.settings import settings
from user_service.core.security import ALGORITHM
from user_service.schemas.user import TokenData

oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{settings.API_V1_STR}/auth/login")


async def get_current_user(
    db: Session = Depends(get_db),
    token: str = Depends(oauth2_scheme)
) -> User:
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
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

def get_current_active_superuser(current_user: User = Depends(get_current_active_user)) -> User:
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=403,
            detail="Insufficient permissions"
        )
    return current_user