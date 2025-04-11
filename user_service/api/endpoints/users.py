# user_service/api/endpoints/users.py
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from typing import Any, List, Optional
from datetime import timedelta

from user_service.schemas.user import (
    User, UserCreate, UserUpdate, Token,
    BridgeConnectionInDB
)
from user_service.api.deps import get_db, get_current_active_user
from user_service.services import users, bridge
from user_service.core.config import settings
from user_service.core.security import create_access_token

router = APIRouter()


@router.post("/register", response_model=User)
async def register_user(
    user_in: UserCreate,
    db: Session = Depends(get_db)
) -> Any:
    """
    Créer un nouvel utilisateur et l'enregistrer auprès de Bridge API.
    """
    # Créer l'utilisateur
    db_user = users.create_user(db, user_in)
    
    # Créer l'utilisateur Bridge
    await bridge.create_bridge_user(db, db_user)
    
    return db_user


@router.post("/auth/login", response_model=Token)
async def login_access_token(
    db: Session = Depends(get_db),
    form_data: OAuth2PasswordRequestForm = Depends()
) -> Any:
    """
    OAuth2 compatible token login, get an access token for future requests.
    """
    user = users.authenticate_user(db, email=form_data.username, password=form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not users.is_active_user(user):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    return {
        "access_token": create_access_token(
            user.id, expires_delta=access_token_expires
        ),
        "token_type": "bearer"
    }


@router.get("/me", response_model=User)
async def read_users_me(
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Get current user information.
    """
    return current_user


@router.put("/me", response_model=User)
async def update_user_me(
    user_in: UserUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Update current user information.
    """
    updated_user = users.update_user(db, current_user.id, user_in)
    return updated_user


@router.get("/bridge/connection", response_model=BridgeConnectionInDB)
async def get_bridge_connection(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Get Bridge API connection information.
    """
    # Vérifier si une connexion existe
    bridge_connection = db.query(bridge.BridgeConnection).filter(
        bridge.BridgeConnection.user_id == current_user.id
    ).first()
    
    if not bridge_connection:
        # Créer une connexion si aucune n'existe
        bridge_connection = await bridge.create_bridge_user(db, current_user)
    
    return bridge_connection


@router.post("/bridge/token", response_model=dict)
async def get_bridge_access_token(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Get Bridge API access token.
    """
    token_data = await bridge.get_bridge_token(db, current_user.id)
    return token_data


@router.post("/bridge/connect-session", response_model=dict)
async def create_bridge_connect_session(
    callback_url: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Create a Bridge API connect session to link bank accounts.
    """
    session_url = await bridge.create_connect_session(db, current_user.id, callback_url)
    return {"connect_url": session_url}