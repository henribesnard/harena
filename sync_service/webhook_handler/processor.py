"""
Processeur de webhooks Bridge.

Ce module contient la logique principale pour valider et traiter
les webhooks reçus de l'API Bridge.
"""

import json
import hmac
import hashlib
import logging
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from typing import Dict, Any, Optional

from db_service.models.sync import WebhookEvent
from sync_service.utils.logging import get_contextual_logger
from sync_service.webhook_handler.handlers import handle_event

logger = logging.getLogger(__name__)

def validate_webhook(payload: str, signature_header: str, secret: str) -> bool:
    """
    Vérifie la signature du webhook Bridge.
    
    Args:
        payload: Corps du webhook en format texte
        signature_header: En-tête de signature fourni par Bridge
        secret: Clé secrète pour vérifier la signature
        
    Returns:
        bool: True si la signature est valide, False sinon
    """
    logger.debug(f"Vérification de signature webhook: payload length={len(payload)}, signature={signature_header[:20]}...")

    if not signature_header or not payload or not secret:
        logger.warning("Données manquantes pour vérifier la signature")
        return False

    signatures = signature_header.split(',')
    v1_signatures = [s.split('=')[1] for s in signatures if s.startswith('v1=')]

    if not v1_signatures:
        logger.warning("Aucune signature v1 trouvée dans l'en-tête: %s", signature_header)
        return False

    computed_signature = hmac.new(
        secret.encode(),
        payload.encode(),
        hashlib.sha256
    ).hexdigest().upper()

    result = computed_signature in [s.upper() for s in v1_signatures]
    if not result:
        logger.warning("Signature webhook INVALIDE. Reçue: %s (...), Calculée: %s (...)", signature_header[:10], computed_signature[:10])
    else:
        logger.debug(f"Signature webhook valide.")
    return result

async def process_webhook(db: Session, webhook_data: Dict[str, Any], signature: str = None) -> WebhookEvent:
    """
    Traite un événement webhook reçu de Bridge API.
    
    Args:
        db: Session de base de données
        webhook_data: Données du webhook reçu
        signature: Signature du webhook pour référence
        
    Returns:
        WebhookEvent: Événement webhook enregistré en base de données
    """
    payload_str = json.dumps(webhook_data)  # Sérialiser pour stockage et logs
    event_type = webhook_data.get("type", "UNKNOWN")
    content_summary = str(webhook_data.get("content", {}))[:200]  # Résumé du contenu pour log
    logger.info(f"Traitement du webhook entrant: type={event_type}, signature={signature is not None}, content_summary='{content_summary}...'.")

    # Enregistrer l'événement brut en BDD
    event = WebhookEvent(
        event_type=event_type,
        event_content=webhook_data.get("content", {}),
        raw_payload=payload_str,
        signature=signature
    )
    db.add(event)
    db.commit()
    db.refresh(event)
    logger.debug(f"Webhook enregistré en base: id={event.id}")

    # Traiter l'événement via le gestionnaire approprié
    try:
        await handle_event(db, event)
        event.processed = True
        event.processing_timestamp = datetime.now(timezone.utc)
        logger.info(f"Webhook id={event.id} de type {event_type} traité avec succès")
    except Exception as e:
        logger.error(f"Erreur lors du traitement du webhook id={event.id} de type {event_type}: {e}", exc_info=True)
        event.error_message = str(e)
        event.processed = True  # Marquer comme traité pour éviter les retentatives infinies
        event.processing_timestamp = datetime.now(timezone.utc)
    
    # Mise à jour finale de l'événement
    db.add(event)
    db.commit()
    db.refresh(event)
    
    return event