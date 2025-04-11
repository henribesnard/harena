# sync_service/services/webhook_handler.py
import json
import hmac
import hashlib
import logging
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from fastapi import HTTPException, status
from typing import Dict, Any

from sync_service.models.sync import WebhookEvent, SyncItem, SyncAccount
from user_service.models.user import User, BridgeConnection
from sync_service.services import sync_manager, transaction_sync
from user_service.core.config import settings

logger = logging.getLogger(__name__)

def verify_webhook_signature(payload: str, signature_header: str, secret: str) -> bool:
    """Vérifier la signature du webhook Bridge."""
    if not signature_header or not payload or not secret:
        return False
        
    # Extraire la signature v1
    signatures = signature_header.split(',')
    v1_signatures = [s.split('=')[1] for s in signatures if s.startswith('v1=')]
    
    if not v1_signatures:
        return False
    
    # Calculer la signature attendue
    computed_signature = hmac.new(
        secret.encode(),
        payload.encode(),
        hashlib.sha256
    ).hexdigest().upper()
    
    # Vérifier si la signature calculée correspond à l'une des signatures reçues
    return computed_signature in [s.upper() for s in v1_signatures]

async def process_webhook(db: Session, webhook_data: Dict[str, Any], signature: str = None) -> WebhookEvent:
    """Traiter un événement webhook reçu de Bridge API."""
    # Enregistrer l'événement webhook
    payload_str = json.dumps(webhook_data)
    event = WebhookEvent(
        event_type=webhook_data.get("type", "UNKNOWN"),
        event_content=webhook_data.get("content", {}),
        raw_payload=payload_str,
        signature=signature
    )
    db.add(event)
    db.commit()
    db.refresh(event)
    
    # Traiter selon le type d'événement
    try:
        if event.event_type == "item.created":
            await handle_item_created(db, event)
        elif event.event_type == "item.refreshed":
            await handle_item_refreshed(db, event)
        elif event.event_type == "item.account.updated":
            await handle_account_updated(db, event)
        elif event.event_type == "TEST_EVENT":
            logger.info("Test webhook received and processed successfully")
        else:
            logger.warning(f"Unhandled webhook event type: {event.event_type}")
        
        # Marquer comme traité
        event.processed = True
        event.processing_timestamp = datetime.now(timezone.utc)
        db.add(event)
        db.commit()
        
        return event
    except Exception as e:
        logger.error(f"Error processing webhook: {str(e)}")
        event.error_message = str(e)
        db.add(event)
        db.commit()
        raise

async def handle_item_created(db: Session, event: WebhookEvent) -> None:
    """Gérer la création d'un nouvel item."""
    content = event.event_content
    item_id = content.get("item_id")
    user_uuid = content.get("user_uuid")
    
    if not item_id or not user_uuid:
        logger.error(f"Missing item_id or user_uuid in webhook: {content}")
        return
    
    # Trouver l'utilisateur correspondant
    bridge_connection = db.query(BridgeConnection).filter(
        BridgeConnection.bridge_user_uuid == user_uuid
    ).first()
    
    if not bridge_connection:
        logger.error(f"No bridge connection found for user_uuid: {user_uuid}")
        return
    
    # Créer ou mettre à jour le SyncItem
    await sync_manager.create_or_update_sync_item(db, bridge_connection.user_id, item_id, content)

async def handle_item_refreshed(db: Session, event: WebhookEvent) -> None:
    """Gérer le rafraîchissement d'un item."""
    content = event.event_content
    item_id = content.get("item_id")
    status = content.get("status", 0)
    status_code_info = content.get("status_code_info")
    full_refresh = content.get("full_refresh", False)
    
    if not item_id:
        logger.error(f"Missing item_id in webhook: {content}")
        return
    
    # Mettre à jour le statut de l'item
    sync_item = db.query(SyncItem).filter(SyncItem.bridge_item_id == item_id).first()
    if not sync_item:
        logger.warning(f"SyncItem not found for item_id: {item_id}")
        return
    
    await sync_manager.update_item_status(db, sync_item, status, status_code_info)
    
    # Si rafraîchissement complet et statut OK, déclencher la synchronisation
    if full_refresh and status == 0:
        await sync_manager.trigger_full_sync_for_item(db, sync_item)

async def handle_account_updated(db: Session, event: WebhookEvent) -> None:
    """Gérer la mise à jour d'un compte."""
    content = event.event_content
    account_id = content.get("account_id")
    
    if not account_id:
        logger.error(f"Missing account_id in webhook: {content}")
        return
    
    # Trouver le compte synchronisé
    sync_account = db.query(SyncAccount).filter(
        SyncAccount.bridge_account_id == account_id
    ).first()
    
    if not sync_account:
        logger.warning(f"SyncAccount not found for account_id: {account_id}")
        return
    
    # Déclencher la synchronisation des transactions
    await transaction_sync.sync_account_transactions(db, sync_account)