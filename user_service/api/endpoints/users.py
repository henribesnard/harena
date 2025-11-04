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
    Créer un nouvel utilisateur.

    La connexion à Bridge API est désormais optionnelle et peut être effectuée
    ultérieurement via l'endpoint /bridge/connect.

    Args:
        user_in: Données d'inscription (email, password, confirm_password, nom/prénom)
        db: Session de base de données (injecté automatiquement)

    Returns:
        User: Utilisateur créé avec ses préférences par défaut

    Raises:
        HTTPException 400: Email déjà enregistré
        HTTPException 422: Validation échouée (mots de passe ne correspondent pas, etc.)
    """
    # Créer l'utilisateur en base
    db_user = users.create_user(db, user_in)

    # L'utilisateur est créé SANS connexion Bridge par défaut
    # La connexion peut être ajoutée plus tard via /bridge/connect
    logger.info(f"User {db_user.id} created without Bridge connection (can be added later)")

    return db_user


@router.post("/auth/login", response_model=Token)
async def login_access_token(
    db: Session = Depends(get_db),
    form_data: OAuth2PasswordRequestForm = Depends()
) -> Any:
    """
    OAuth2 compatible token login, get an access token for future requests.

    Args:
        form_data: Formulaire OAuth2 (username=email, password)
        db: Session de base de données (injecté automatiquement)

    Returns:
        Token: Access token JWT et type (bearer)

    Raises:
        HTTPException 401: Email ou mot de passe incorrect
        HTTPException 400: Utilisateur inactif
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

    Args:
        current_user: Utilisateur authentifié (injecté depuis JWT)

    Returns:
        User: Informations complètes de l'utilisateur avec préférences et connexions Bridge

    Raises:
        HTTPException 401: Token invalide ou expiré
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

    Args:
        user_in: Données à mettre à jour (email, nom/prénom, password, preferences)
        db: Session de base de données (injecté automatiquement)
        current_user: Utilisateur authentifié (injecté depuis JWT)

    Returns:
        User: Utilisateur mis à jour

    Raises:
        HTTPException 401: Token invalide ou expiré
        HTTPException 404: Utilisateur non trouvé
    """
    updated_user = users.update_user(db, current_user.id, user_in)
    return updated_user


@router.post("/bridge/connect", response_model=BridgeConnectionInDB)
async def connect_bridge_account(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Connecter le compte utilisateur à Bridge API.

    Cette étape est optionnelle et peut être effectuée après l'inscription.
    Elle permet à l'utilisateur de synchroniser ses comptes bancaires.

    Raises:
        HTTPException 409: Bridge connection already exists
        HTTPException 500: Failed to connect to Bridge API
        HTTPException 503: Bridge API unavailable
    """
    # Vérifier si une connexion existe déjà
    existing_connection = bridge.get_bridge_connection_by_user(db, current_user.id)

    if existing_connection:
        logger.info(f"Bridge connection already exists for user {current_user.id}")
        return existing_connection

    try:
        bridge_connection = await bridge.create_bridge_user(db, current_user)
        logger.info(f"Bridge connection created successfully for user {current_user.id}")
        return bridge_connection
    except HTTPException:
        # Renvoyer les exceptions HTTP déjà formatées
        raise
    except Exception as e:
        # Erreurs inattendues
        logger.error(f"Failed to connect Bridge for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to connect Bridge: {str(e)}"
        )


@router.get("/bridge/connection", response_model=BridgeConnectionInDB)
async def get_bridge_connection(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Get Bridge API connection information.

    Retourne une erreur 404 si aucune connexion n'existe.
    Utilisez /bridge/connect pour créer une connexion.

    Raises:
        HTTPException 404: No Bridge connection found
    """
    # Vérifier si une connexion existe
    bridge_connection = bridge.get_bridge_connection_by_user(db, current_user.id)

    if not bridge_connection:
        # NE PAS créer automatiquement
        # Retourner un message indiquant que l'utilisateur doit se connecter
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No Bridge connection found. Please connect your bank account first using POST /bridge/connect"
        )

    return bridge_connection


@router.post("/bridge/token", response_model=dict)
async def get_bridge_access_token(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Get Bridge API access token.

    Récupère ou génère un token d'accès Bridge API pour l'utilisateur.
    Les tokens sont mis en cache et réutilisés s'ils sont encore valides.

    Args:
        db: Session de base de données (injecté automatiquement)
        current_user: Utilisateur authentifié (injecté depuis JWT)

    Returns:
        dict: Token Bridge avec access_token et expires_at

    Raises:
        HTTPException 401: Token JWT invalide ou expiré
        HTTPException 404: Aucune connexion Bridge trouvée
        HTTPException 503: Bridge API indisponible
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