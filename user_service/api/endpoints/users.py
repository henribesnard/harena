from fastapi import APIRouter, Depends, HTTPException, status, Query
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from typing import Any, List, Optional
from datetime import timedelta
import logging

from user_service.schemas.user import (
    User, UserCreate, UserUpdate, Token,
    BridgeConnectionInDB
)
from user_service.api.deps import get_db, get_current_active_user
from user_service.services import users, bridge
from config_service.config import settings
from user_service.core.security import create_access_token

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/register", response_model=User)
async def register_user(
    user_in: UserCreate,
    db: Session = Depends(get_db)
) -> Any:
    """
    Créer un nouvel utilisateur et l'enregistrer auprès de Bridge API.
    
    Note: Les administrateurs (is_superuser=True) ne nécessitent pas de compte Bridge
    car ils n'ont pas besoin d'accéder à des données financières personnelles.
    """
    # Créer l'utilisateur en base
    db_user = users.create_user(db, user_in)
    
    # Créer l'utilisateur Bridge SEULEMENT pour les utilisateurs normaux
    if not db_user.is_superuser:
        try:
            await bridge.create_bridge_user(db, db_user)
        except Exception as e:
            # En cas d'échec Bridge pour un utilisateur normal, on annule la création
            db.delete(db_user)
            db.commit()
            raise HTTPException(
                status_code=500,
                detail=f"Failed to create Bridge user: {str(e)}"
            )
    else:
        # Pour les admins, on log juste qu'on ne crée pas de compte Bridge
        logger.info(f"Admin user {db_user.id} created without Bridge account (not required)")
    
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
    access_token = create_access_token(
        user.id,
        permissions=["chat:write"],
        expires_delta=access_token_expires,
    )
    return {
        "access_token": access_token,
        "token_type": "bearer"
    }


@router.get("/me", response_model=User)
async def read_users_me(
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Get current user information.
    """
    # Ensure permissions field is always present in response
    current_user.permissions = getattr(current_user, "permissions", [])
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
    country_code: Optional[str] = "FR",
    account_types: Optional[str] = "payment",
    context: Optional[str] = None,
    provider_id: Optional[int] = None,
    item_id: Optional[int] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Create a Bridge API connect session to link bank accounts.
    
    - **callback_url**: URL optionnelle pour rediriger l'utilisateur après la session
    - **country_code**: Code pays pour la sélection des banques (défaut: FR)
    - **account_types**: Types de comptes requis (défaut: payment)
    - **context**: Contexte à ajouter à l'URL de callback (max 100 caractères)
    - **provider_id**: ID du fournisseur pour aller directement à la page d'authentification
    - **item_id**: ID de l'item pour gérer une connexion existante
    """
    session_url = await bridge.create_connect_session(
        db, 
        current_user.id, 
        callback_url=callback_url,
        country_code=country_code,
        account_types=account_types,
        context=context,
        provider_id=provider_id,
        item_id=item_id
    )
    
    # Dans le cas d'une nouvelle connexion (pas item_id), préparer les structures de suivi
    # Cette partie sera principalement gérée par les webhooks, mais on peut initialiser ici
    # si nécessaire pour une meilleure expérience utilisateur
    if not item_id:
        try:
            # Import uniquement ici pour éviter les dépendances circulaires
            from sync_service.services import sync_manager
            # On pourrait pré-initialiser des structures si nécessaire
            # Mais généralement, cela sera géré par le webhook item.created
        except ImportError:
            # Si le module sync_service n'est pas encore disponible, on ignore simplement
            pass
            
    return {"connect_url": session_url}