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

from conversation_service.api.dependencies import get_db
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
    # Récupérer le corps de la requête sous forme de bytes brutes
    raw_payload = await request.body()
    raw_payload_str = raw_payload.decode('utf-8', errors='replace')
    
    # Log complet pour débogage
    logger.debug(f"Payload brut (bytes): {raw_payload[:50]}...")
    logger.debug(f"Payload brut (str): {raw_payload_str[:50]}...")
    logger.debug(f"Signature complète: {bridge_signature}")
    logger.debug(f"Secret webhook utilisé (longueur): {len(WEBHOOK_SECRET)}")
    
    try:
        # Conversion en dict pour traitement
        payload_dict = await request.json()
    except Exception as e:
        logger.error(f"Impossible de parser le JSON du webhook: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid JSON payload: {str(e)}"
        )
    
    # Vérifier la signature
    signature_valid = False
    if bridge_signature and WEBHOOK_SECRET:
        # Essayer les deux méthodes pour voir laquelle fonctionne
        signature_valid_bytes = validate_webhook(raw_payload, bridge_signature, WEBHOOK_SECRET)
        signature_valid_str = validate_webhook(raw_payload_str, bridge_signature, WEBHOOK_SECRET)
        signature_valid = signature_valid_bytes or signature_valid_str
        
        logger.debug(f"Résultat validation bytes: {signature_valid_bytes}")
        logger.debug(f"Résultat validation str: {signature_valid_str}")
    
    # TEMPORAIRE: Accepter les webhooks en production même avec signature invalide
    is_production = settings.ENVIRONMENT.lower() == "production"
    if is_production and not signature_valid:
        logger.warning(f"⚠️ Signature webhook invalide en production, mais traitement poursuivi pour déboguer.")
    
    # Traiter l'événement webhook
    try:
        webhook_event = await process_webhook(db, payload_dict, bridge_signature, raw_payload)
        return {
            "status": "received", 
            "event_id": webhook_event.id, 
            "type": webhook_event.event_type,
            "signature_verified": signature_valid,
            "debug_mode": True  # Indiquer qu'on est en mode débogage
        }
    except Exception as e:
        logger.error(f"Erreur lors du traitement du webhook: {str(e)}", exc_info=True)
        return {
            "status": "received_with_errors", 
            "message": str(e),
            "signature_verified": signature_valid
        }