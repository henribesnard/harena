# user_service/services/bridge.py
from sqlalchemy.orm import Session
from fastapi import HTTPException, status
import httpx
from datetime import datetime, timezone
from typing import Optional, Dict, Any

from user_service.models.user import User, BridgeConnection
from user_service.core.config import settings


async def create_bridge_user(db: Session, user: User) -> BridgeConnection:
    """Crée un utilisateur dans Bridge API et enregistre sa connexion"""
    
    # Générer un external_user_id basé sur l'ID de l'utilisateur Harena
    external_user_id = f"harena-user-{user.id}"
    
    # Vérifier si une connexion existe déjà
    existing_connection = db.query(BridgeConnection).filter(
        BridgeConnection.user_id == user.id
    ).first()
    
    if existing_connection:
        return existing_connection
    
    # Appel à l'API Bridge pour créer l'utilisateur
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{settings.BRIDGE_API_URL}/aggregation/users",
            json={"external_user_id": external_user_id},
            headers={
                "accept": "application/json",
                "Bridge-Version": settings.BRIDGE_API_VERSION,
                "content-type": "application/json",
                "Client-Id": settings.BRIDGE_CLIENT_ID,
                "Client-Secret": settings.BRIDGE_CLIENT_SECRET
            }
        )
        
        if response.status_code != 200:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to create Bridge user: {response.text}"
            )
        
        bridge_user = response.json()
        
        # Créer la connexion Bridge en BDD
        bridge_connection = BridgeConnection(
            user_id=user.id,
            bridge_user_uuid=bridge_user["uuid"],
            external_user_id=external_user_id
        )
        
        db.add(bridge_connection)
        db.commit()
        db.refresh(bridge_connection)
        
        return bridge_connection


async def get_bridge_token(db: Session, user_id: int) -> Dict[str, Any]:
    """Récupère ou génère un token d'authentification Bridge"""
    
    # Récupérer la connexion Bridge
    bridge_connection = db.query(BridgeConnection).filter(
        BridgeConnection.user_id == user_id
    ).first()
    
    if not bridge_connection:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No Bridge connection found for this user"
        )
    
    # Vérifier si le token existant est valide
    now = datetime.now(timezone.utc)
    if bridge_connection.token_expires_at and bridge_connection.token_expires_at > now:
        return {
            "access_token": bridge_connection.last_token,
            "expires_at": bridge_connection.token_expires_at
        }
    
    # Sinon, demander un nouveau token
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{settings.BRIDGE_API_URL}/aggregation/authorization/token",
            json={"external_user_id": bridge_connection.external_user_id},
            headers={
                "accept": "application/json",
                "Bridge-Version": settings.BRIDGE_API_VERSION,
                "content-type": "application/json",
                "Client-Id": settings.BRIDGE_CLIENT_ID,
                "Client-Secret": settings.BRIDGE_CLIENT_SECRET
            }
        )
        
        if response.status_code != 200:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get Bridge token: {response.text}"
            )
        
        token_data = response.json()
        
        # Mettre à jour la connexion avec le nouveau token
        bridge_connection.last_token = token_data["access_token"]
        bridge_connection.token_expires_at = datetime.fromisoformat(token_data["expires_at"].replace('Z', '+00:00'))
        
        db.add(bridge_connection)
        db.commit()
        db.refresh(bridge_connection)
        
        return {
            "access_token": bridge_connection.last_token,
            "expires_at": bridge_connection.token_expires_at
        }


async def create_connect_session(db: Session, user_id: int, callback_url: Optional[str] = None) -> str:
    """Crée une session de connexion Bridge pour connecter un compte bancaire"""
    
    # Récupérer le token Bridge
    token_data = await get_bridge_token(db, user_id)
    
    # Créer une session de connexion
    async with httpx.AsyncClient() as client:
        payload = {}
        if callback_url:
            payload["callback_url"] = callback_url
        
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
        
        if response.status_code != 200:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to create connect session: {response.text}"
            )
        
        session_data = response.json()
        return session_data["url"]