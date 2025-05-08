"""
Endpoints pour la réception des webhooks Bridge.

Ce module définit les routes pour recevoir et traiter les webhooks envoyés par Bridge API.
"""
from fastapi import APIRouter, Depends, HTTPException, Header, Request, status
from sqlalchemy.orm import Session
from typing import Dict, Any, Optional

from db_service.session import get_db
from config_service.config import settings
from sync_service.webhook_handler.processor import process_webhook, validate_webhook

router = APIRouter()

# Récupérer le secret webhook de la configuration
WEBHOOK_SECRET = settings.BRIDGE_WEBHOOK_SECRET

@router.post("/bridge", status_code=status.HTTP_200_OK)
async def receive_bridge_webhook(
    request: Request,
    db: Session = Depends(get_db),
    bridge_signature: Optional[str] = Header(None, alias="BridgeApi-Signature")
):
    """
    Endpoint pour recevoir les webhooks de Bridge API.
    Vérifie la signature et traite les événements.
    
    Args:
        request: La requête HTTP entrante
        db: Session de base de données
        bridge_signature: Signature du webhook Bridge pour vérification
        
    Returns:
        Dict: Statut de réception du webhook
    """
    # Récupérer le corps de la requête
    raw_payload = await request.body()
    
    try:
        payload = await request.json()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid JSON payload: {str(e)}"
        )
    
    # Vérifier la signature si on est en production
    is_production = settings.ENVIRONMENT.lower() == "production"
    
    if is_production:
        if not bridge_signature:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing webhook signature in production environment"
            )
            
        if not validate_webhook(raw_payload.decode(), bridge_signature, WEBHOOK_SECRET):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid webhook signature"
            )
    else:
        # En environnement de développement, on accepte les webhooks sans signature
        # mais on les vérifie quand même si la signature est fournie
        if bridge_signature and not validate_webhook(raw_payload.decode(), bridge_signature, WEBHOOK_SECRET):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid webhook signature"
            )
    
    # Traiter l'événement
    try:
        webhook_event = await process_webhook(db, payload, bridge_signature)
        return {"status": "received", "event_id": webhook_event.id, "type": webhook_event.event_type}
    except Exception as e:
        # Log l'erreur mais retourner quand même un statut 200
        # pour ne pas déclencher de réessais côté Bridge
        # Les erreurs sont déjà enregistrées dans la table WebhookEvent
        return {"status": "received_with_errors", "message": str(e)}