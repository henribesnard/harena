"""
Endpoints pour la réception des webhooks Bridge.

Ce module définit les routes pour recevoir et traiter les webhooks envoyés par Bridge API.
"""
import logging
from fastapi import APIRouter, Depends, HTTPException, Header, Request, status, Body
from sqlalchemy.orm import Session
from typing import Dict, Any, Optional

# Configuration du logger
logger = logging.getLogger(__name__)

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
    # Récupérer le corps de la requête sous forme de bytes brutes
    raw_payload = await request.body()
    
    # Log de débogage pour le payload brut reçu
    payload_debug = raw_payload[:200].decode('utf-8', errors='replace') if raw_payload else "Empty"
    logger.debug(f"Payload brut reçu (début): {payload_debug}...")
    
    try:
        # Conversion en dict pour traitement
        payload_dict = await request.json()
    except Exception as e:
        logger.error(f"Impossible de parser le JSON du webhook: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid JSON payload: {str(e)}"
        )
    
    # Vérifier la signature si on est en production
    is_production = settings.ENVIRONMENT.lower() == "production"
    signature_valid = False
    
    if bridge_signature and WEBHOOK_SECRET:
        # Vérifier la signature avec le payload brut (bytes)
        signature_valid = validate_webhook(raw_payload, bridge_signature, WEBHOOK_SECRET)
        
        if is_production and not signature_valid:
            logger.warning(f"Signature webhook invalide en production: {bridge_signature[:20]}...")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid webhook signature"
            )
    elif is_production:
        if not bridge_signature:
            logger.warning("En-tête de signature manquant en production")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing webhook signature in production environment"
            )
        if not WEBHOOK_SECRET:
            logger.error("Secret webhook non configuré en production")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
                detail="Webhook secret not configured"
            )
    else:
        # En mode développement, on accepte les webhooks sans signature
        logger.info("Vérification de signature ignorée en environnement de développement")
    
    # Traiter l'événement en passant le payload brut pour référence future
    try:
        webhook_event = await process_webhook(db, payload_dict, bridge_signature, raw_payload)
        return {
            "status": "received", 
            "event_id": webhook_event.id, 
            "type": webhook_event.event_type,
            "signature_verified": signature_valid
        }
    except Exception as e:
        # Log l'erreur mais retourner quand même un statut 200
        # pour ne pas déclencher de réessais côté Bridge
        logger.error(f"Erreur lors du traitement du webhook: {str(e)}", exc_info=True)
        return {
            "status": "received_with_errors", 
            "message": str(e),
            "signature_verified": signature_valid
        }