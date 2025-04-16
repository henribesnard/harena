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
    logger.debug(f"Vérification de signature webhook: payload length={len(payload)}, signature={signature_header[:20]}...")
    
    if not signature_header or not payload or not secret:
        logger.warning("Données manquantes pour vérifier la signature")
        return False
        
    # Extraire la signature v1
    signatures = signature_header.split(',')
    v1_signatures = [s.split('=')[1] for s in signatures if s.startswith('v1=')]
    
    if not v1_signatures:
        logger.warning("Aucune signature v1 trouvée")
        return False
    
    # Calculer la signature attendue
    computed_signature = hmac.new(
        secret.encode(),
        payload.encode(),
        hashlib.sha256
    ).hexdigest().upper()
    
    # Vérifier si la signature calculée correspond à l'une des signatures reçues
    result = computed_signature in [s.upper() for s in v1_signatures]
    logger.debug(f"Résultat de vérification de signature: {result}")
    return result

async def process_webhook(db: Session, webhook_data: Dict[str, Any], signature: str = None) -> WebhookEvent:
    """Traiter un événement webhook reçu de Bridge API."""
    # Enregistrer l'événement webhook
    payload_str = json.dumps(webhook_data)
    event_type = webhook_data.get("type", "UNKNOWN")
    logger.info(f"Traitement du webhook: type={event_type}")
    
    event = WebhookEvent(
        event_type=event_type,
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
            logger.info(f"Traitement d'un événement item.created: {event.event_content}")
            await handle_item_created(db, event)
        elif event.event_type == "item.refreshed":
            logger.info(f"Traitement d'un événement item.refreshed: {event.event_content}")
            await handle_item_refreshed(db, event)
        elif event.event_type == "item.account.updated":
            logger.info(f"Traitement d'un événement item.account.updated: {event.event_content}")
            await handle_account_updated(db, event)
        elif event.event_type == "item.account.created":
            logger.info(f"Traitement d'un événement item.account.created: {event.event_content}")
            await handle_account_created(db, event)
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
        logger.error(f"Error processing webhook: {str(e)}", exc_info=True)
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
    
    logger.info(f"Traitement de la création de l'item {item_id} pour l'utilisateur {user_uuid}")
    
    # Trouver l'utilisateur correspondant
    bridge_connection = db.query(BridgeConnection).filter(
        BridgeConnection.bridge_user_uuid == user_uuid
    ).first()
    
    if not bridge_connection:
        logger.error(f"No bridge connection found for user_uuid: {user_uuid}")
        return
    
    logger.info(f"Bridge connection trouvée: user_id={bridge_connection.user_id}")
    
    # Créer ou mettre à jour le SyncItem
    sync_item = await sync_manager.create_or_update_sync_item(db, bridge_connection.user_id, item_id, content)
    logger.info(f"SyncItem créé/mis à jour: id={sync_item.id}, bridge_item_id={sync_item.bridge_item_id}")

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
    
    logger.info(f"Traitement du rafraîchissement de l'item {item_id} (status={status}, full_refresh={full_refresh})")
    
    # Mettre à jour le statut de l'item
    sync_item = db.query(SyncItem).filter(SyncItem.bridge_item_id == item_id).first()
    if not sync_item:
        logger.warning(f"SyncItem not found for item_id: {item_id}")
        return
    
    logger.info(f"SyncItem trouvé: id={sync_item.id}, user_id={sync_item.user_id}")
    
    await sync_manager.update_item_status(db, sync_item, status, status_code_info)
    
    # Si rafraîchissement complet et statut OK, déclencher la synchronisation
    if status == 0:
        logger.info(f"Item {item_id} a un statut OK, déclenchement de la synchronisation")
        await sync_manager.trigger_full_sync_for_item(db, sync_item)
    else:
        logger.warning(f"Item {item_id} a un statut non-OK ({status}), pas de synchronisation déclenchée")

async def handle_account_updated(db: Session, event: WebhookEvent) -> None:
    """Gérer la mise à jour d'un compte."""
    content = event.event_content
    account_id = content.get("account_id")
    
    if not account_id:
        logger.error(f"Missing account_id in webhook: {content}")
        return
    
    logger.info(f"Traitement de la mise à jour du compte {account_id}")
    
    # Trouver le compte synchronisé
    sync_account = db.query(SyncAccount).filter(
        SyncAccount.bridge_account_id == account_id
    ).first()
    
    if not sync_account:
        logger.warning(f"SyncAccount not found for account_id: {account_id}")
        return
    
    logger.info(f"SyncAccount trouvé: id={sync_account.id}, item_id={sync_account.item_id}")
    
    # Déclencher la synchronisation des transactions
    result = await transaction_sync.sync_account_transactions(db, sync_account)
    logger.info(f"Résultat de la synchronisation du compte {account_id}: {result}")

async def handle_account_created(db: Session, event: WebhookEvent) -> None:
    """Gérer la création d'un nouveau compte."""
    content = event.event_content
    account_id = content.get("account_id")
    item_id = content.get("item_id")
    user_uuid = content.get("user_uuid")
    
    if not account_id or not item_id:
        logger.error(f"Missing account_id or item_id in webhook: {content}")
        return
    
    logger.info(f"Traitement de la création du compte {account_id} pour l'item {item_id}")
    
    # Trouver l'item correspondant
    sync_item = db.query(SyncItem).filter(SyncItem.bridge_item_id == item_id).first()
    if not sync_item:
        logger.warning(f"SyncItem not found for item_id: {item_id}")
        # Essayer de trouver l'utilisateur et créer l'item si besoin
        if user_uuid:
            bridge_connection = db.query(BridgeConnection).filter(
                BridgeConnection.bridge_user_uuid == user_uuid
            ).first()
            if bridge_connection:
                logger.info(f"Création d'un nouvel item pour le compte: item_id={item_id}, user_id={bridge_connection.user_id}")
                sync_item = await sync_manager.create_or_update_sync_item(db, bridge_connection.user_id, item_id, content)
            else:
                logger.error(f"No bridge connection found for user_uuid: {user_uuid}")
                return
        else:
            logger.error("Impossible de créer l'item: user_uuid manquant")
            return
    
    logger.info(f"SyncItem trouvé/créé: id={sync_item.id}, user_id={sync_item.user_id}")
    
    # Vérifier si le compte existe déjà
    existing_account = db.query(SyncAccount).filter(
        SyncAccount.bridge_account_id == account_id
    ).first()
    
    if existing_account:
        logger.info(f"Le compte {account_id} existe déjà, mise à jour")
        existing_account.account_name = content.get("name", existing_account.account_name)
        db.add(existing_account)
        db.commit()
        db.refresh(existing_account)
        sync_account = existing_account
    else:
        # Créer un nouveau compte
        logger.info(f"Création d'un nouveau SyncAccount pour bridge_account_id={account_id}")
        sync_account = SyncAccount(
            item_id=sync_item.id,
            bridge_account_id=account_id,
            account_name=content.get("name", f"Account {account_id}")
        )
        db.add(sync_account)
        db.commit()
        db.refresh(sync_account)
    
    # Déclencher la synchronisation des transactions pour ce nouveau compte
    logger.info(f"Déclenchement de la synchronisation pour le nouveau compte: id={sync_account.id}")
    result = await transaction_sync.sync_account_transactions(db, sync_account)
    logger.info(f"Résultat de la synchronisation du compte {account_id}: {result}")