# user_service/services/bridge.py
from sqlalchemy.orm import Session
from fastapi import HTTPException, status
import httpx
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
import logging

from user_service.models.user import User, BridgeConnection
from user_service.core.config import settings

# Configuration des logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def create_bridge_user(db: Session, user: User) -> BridgeConnection:
    """Crée un utilisateur dans Bridge API et enregistre sa connexion"""
    
    # Générer un external_user_id basé sur l'ID de l'utilisateur Harena
    external_user_id = f"harena-user-{user.id}"
    logger.info(f"Creating Bridge user with external_user_id: {external_user_id}")
    
    # Vérifier si une connexion existe déjà
    existing_connection = db.query(BridgeConnection).filter(
        BridgeConnection.user_id == user.id
    ).first()
    
    if existing_connection:
        logger.info(f"Found existing Bridge connection for user {user.id}")
        return existing_connection
    
    # Préparation des headers
    headers = {
        "accept": "application/json",
        "Bridge-Version": settings.BRIDGE_API_VERSION,
        "content-type": "application/json",
        "Client-Id": settings.BRIDGE_CLIENT_ID,
        "Client-Secret": settings.BRIDGE_CLIENT_SECRET
    }
    logger.info(f"API URL: {settings.BRIDGE_API_URL}/aggregation/users")
    logger.info(f"Headers prepared (credentials hidden)")
    
    # Préparation du payload
    payload = {"external_user_id": external_user_id}
    logger.info(f"Payload: {payload}")
    
    # Appel à l'API Bridge pour créer l'utilisateur
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{settings.BRIDGE_API_URL}/aggregation/users",
                json=payload,
                headers=headers
            )
            
            logger.info(f"Bridge API response status: {response.status_code}")
            logger.info(f"Bridge API response: {response.text}")
            
            if response.status_code not in [200, 201]:
                logger.error(f"Failed to create Bridge user: {response.text}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to create Bridge user: {response.text}"
                )
            
            bridge_user = response.json()
            logger.info(f"Bridge user created successfully: {bridge_user}")
            
            # Vérifier que la réponse contient les champs attendus
            if "uuid" not in bridge_user:
                logger.error(f"Bridge API response missing uuid field: {bridge_user}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Invalid Bridge user response: missing uuid"
                )
            
            # Créer la connexion Bridge en BDD
            bridge_connection = BridgeConnection(
                user_id=user.id,
                bridge_user_uuid=bridge_user["uuid"],
                external_user_id=external_user_id
            )
            
            logger.info(f"Saving Bridge connection to database")
            try:
                db.add(bridge_connection)
                db.commit()
                db.refresh(bridge_connection)
                logger.info(f"Bridge connection saved successfully: {bridge_connection.id}")
                return bridge_connection
            except Exception as db_error:
                logger.error(f"Database error: {str(db_error)}")
                db.rollback()
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Database error: {str(db_error)}"
                )
    except httpx.RequestError as req_error:
        logger.error(f"HTTP request error: {str(req_error)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to connect to Bridge API: {str(req_error)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create Bridge user: {str(e)}"
        )


async def get_bridge_token(db: Session, user_id: int) -> Dict[str, Any]:
    """Récupère ou génère un token d'authentification Bridge"""
    
    # Récupérer la connexion Bridge
    bridge_connection = db.query(BridgeConnection).filter(
        BridgeConnection.user_id == user_id
    ).first()
    
    if not bridge_connection:
        logger.error(f"No Bridge connection found for user {user_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No Bridge connection found for this user"
        )
    
    # Vérifier si le token existant est valide
    now = datetime.now(timezone.utc)
    if bridge_connection.token_expires_at and bridge_connection.token_expires_at > now and bridge_connection.last_token:
        logger.info(f"Using existing valid token for user {user_id}")
        return {
            "access_token": bridge_connection.last_token,
            "expires_at": bridge_connection.token_expires_at
        }
    
    # Sinon, demander un nouveau token
    logger.info(f"Requesting new token for user {user_id}")
    
    headers = {
        "accept": "application/json",
        "Bridge-Version": settings.BRIDGE_API_VERSION,
        "content-type": "application/json",
        "Client-Id": settings.BRIDGE_CLIENT_ID,
        "Client-Secret": settings.BRIDGE_CLIENT_SECRET
    }
    
    payload = {"external_user_id": bridge_connection.external_user_id}
    logger.info(f"Token payload: {payload}")
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{settings.BRIDGE_API_URL}/aggregation/authorization/token",
                json=payload,
                headers=headers
            )
            
            logger.info(f"Bridge token API response status: {response.status_code}")
            
            if response.status_code not in [200, 201]:
                logger.error(f"Failed to get Bridge token: {response.text}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to get Bridge token: {response.text}"
                )
            
            token_data = response.json()
            logger.info(f"Bridge token received successfully")
            
            # Vérifier que la réponse contient les champs attendus
            if "access_token" not in token_data or "expires_at" not in token_data:
                logger.error(f"Bridge token API response missing required fields")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Invalid Bridge token response: missing required fields"
                )
            
            # Mettre à jour la connexion avec le nouveau token
            try:
                bridge_connection.last_token = token_data["access_token"]
                bridge_connection.token_expires_at = datetime.fromisoformat(token_data["expires_at"].replace('Z', '+00:00'))
                
                db.add(bridge_connection)
                db.commit()
                db.refresh(bridge_connection)
                logger.info(f"Bridge token saved to database for user {user_id}")
                
                return {
                    "access_token": bridge_connection.last_token,
                    "expires_at": bridge_connection.token_expires_at
                }
            except Exception as db_error:
                logger.error(f"Database error when saving token: {str(db_error)}")
                db.rollback()
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Database error: {str(db_error)}"
                )
    except httpx.RequestError as req_error:
        logger.error(f"HTTP request error when getting token: {str(req_error)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to connect to Bridge API: {str(req_error)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error when getting token: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get Bridge token: {str(e)}"
        )


async def create_connect_session(
    db: Session, 
    user_id: int, 
    callback_url: Optional[str] = None,
    country_code: Optional[str] = "FR",
    account_types: Optional[str] = "payment",
    context: Optional[str] = None,
    provider_id: Optional[int] = None,
    item_id: Optional[int] = None
) -> str:
    """
    Crée une session de connexion Bridge pour connecter un compte bancaire
    
    Trois modes de fonctionnement selon les paramètres:
    1. Connexion standard: user_email obligatoire (récupéré automatiquement)
    2. Connexion avec fournisseur présélectionné: provider_id + user_email
    3. Gestion d'un item existant: item_id obligatoire
    """
    
    logger.info(f"Creating connect session for user {user_id}")
    
    # Récupérer l'utilisateur pour obtenir son email
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        logger.error(f"User not found: {user_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Récupérer le token Bridge
    token_data = await get_bridge_token(db, user_id)
    
    # Créer le payload selon le mode de fonctionnement
    payload = {}
    
    # Mode 3: Gestion d'un item existant
    if item_id:
        payload["item_id"] = item_id
    # Mode 2: Connexion avec fournisseur présélectionné
    elif provider_id:
        payload["provider_id"] = provider_id
        payload["user_email"] = user.email
    # Mode 1: Connexion standard
    else:
        payload["user_email"] = user.email
    
    # Paramètres optionnels communs
    if callback_url:
        payload["callback_url"] = callback_url
    if country_code:
        payload["country_code"] = country_code
    if account_types:
        payload["account_types"] = account_types
    if context and len(context) <= 100:
        payload["context"] = context
    
    logger.info(f"Connect session payload: {payload}")
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{settings.BRIDGE_API_URL}/aggregation/connect-sessions",
                json=payload,
                headers={
                    "accept": "application/json",
                    "Bridge-Version": settings.BRIDGE_API_VERSION,
                    "content-type": "application/json",
                    "Client-Id": settings.BRIDGE_CLIENT_ID,
                    "Client-Secret": settings.BRIDGE_CLIENT_SECRET,
                    "authorization": f"Bearer {token_data['access_token']}"
                }
            )
            
            logger.info(f"Connect session API response status: {response.status_code}")
            
            if response.status_code not in [200, 201]:
                logger.error(f"Failed to create connect session: {response.text}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to create connect session: {response.text}"
                )
            
            session_data = response.json()
            logger.info(f"Connect session created successfully")
            
            # Vérifier que la réponse contient les champs attendus
            if "url" not in session_data:
                logger.error(f"Connect session API response missing url field")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Invalid connect session response: missing url"
                )
            
            return session_data["url"]
    except httpx.RequestError as req_error:
        logger.error(f"HTTP request error when creating session: {str(req_error)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to connect to Bridge API: {str(req_error)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error when creating session: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create connect session: {str(e)}"
        )