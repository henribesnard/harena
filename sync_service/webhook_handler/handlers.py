"""
Gestionnaires spécifiques pour les différents types d'événements webhook.

Ce module contient les fonctions de traitement pour chaque type d'événement
envoyé par l'API Bridge.
"""

import logging
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from typing import Dict, List, Any, Optional

from db_service.models.sync import WebhookEvent, SyncItem, SyncAccount
from db_service.models.user import BridgeConnection
from sync_service.utils.logging import get_contextual_logger

logger = logging.getLogger(__name__)

async def handle_event(db: Session, event: WebhookEvent) -> None:
    """
    Sélectionne et exécute le gestionnaire approprié pour un événement webhook.
    
    Args:
        db: Session de base de données
        event: Événement webhook à traiter
    """
    event_type = event.event_type
    ctx_logger = get_contextual_logger("webhook_handler", event_id=event.id, event_type=event_type)
    
    # Tests et événements spéciaux
    if event_type == "TEST_EVENT":
        ctx_logger.info(f"Événement de test reçu et traité avec succès (id={event.id}).")
        return
    
    # Sélectionner le gestionnaire approprié selon le type d'événement
    handler_map = {
        "item.created": handle_item_created,
        "item.refreshed": handle_item_refreshed,
        "item.account.updated": handle_account_updated,
        "item.account.created": handle_account_created,
    }
    
    handler = handler_map.get(event_type)
    if handler:
        ctx_logger.info(f"Traitement de l'événement {event_type} avec le gestionnaire {handler.__name__}")
        await handler(db, event)
    else:
        ctx_logger.warning(f"Aucun gestionnaire défini pour le type d'événement {event_type}")
        raise ValueError(f"Unsupported event type: {event_type}")

async def _find_user_id_from_uuid(db: Session, user_uuid: str) -> Optional[int]:
    """
    Trouve l'ID utilisateur interne à partir de l'UUID Bridge.
    
    Args:
        db: Session de base de données
        user_uuid: UUID utilisateur fourni par Bridge
        
    Returns:
        int: ID utilisateur interne si trouvé, None sinon
    """
    bridge_connection = db.query(BridgeConnection).filter(
        BridgeConnection.bridge_user_uuid == user_uuid
    ).first()
    
    if bridge_connection:
        return bridge_connection.user_id
    
    logger.error(f"Aucune connexion Bridge trouvée pour user_uuid: {user_uuid}")
    return None

async def handle_item_created(db: Session, event: WebhookEvent) -> None:
    """
    Gère la création d'un nouvel item.
    
    Args:
        db: Session de base de données
        event: Événement webhook de type item.created
    """
    content = event.event_content
    item_id = content.get("item_id")
    user_uuid = content.get("user_uuid")
    ctx_logger = get_contextual_logger("webhook_handler", event_id=event.id, event_type=event.event_type, bridge_item_id=item_id, user_uuid=user_uuid)

    if not item_id or not user_uuid:
        ctx_logger.error(f"Données manquantes (item_id ou user_uuid) dans webhook item.created: {content}")
        raise ValueError("Missing item_id or user_uuid in item.created event")

    ctx_logger.info(f"Traitement item.created: item {item_id} pour user {user_uuid}")

    user_id = await _find_user_id_from_uuid(db, user_uuid)
    if not user_id:
        raise ValueError(f"User not found for user_uuid: {user_uuid}")

    ctx_logger.info(f"Utilisateur interne trouvé: user_id={user_id}")
    
    # Créer ou mettre à jour le SyncItem en SQL
    from sync_service.sync_manager.orchestrator import create_or_update_sync_item
    sync_item = await create_or_update_sync_item(db, user_id, item_id, content)
    ctx_logger.info(f"SyncItem SQL créé/mis à jour: id={sync_item.id}")

async def handle_item_refreshed(db: Session, event: WebhookEvent) -> None:
    """
    Gère le rafraîchissement d'un item.
    
    Args:
        db: Session de base de données
        event: Événement webhook de type item.refreshed
    """
    content = event.event_content
    item_id = content.get("item_id")
    status_code = content.get("status", 0)
    status_code_info = content.get("status_code_info")
    status_description = content.get("status_code_description")
    full_refresh = content.get("full_refresh", False)
    user_uuid = content.get("user_uuid")
    ctx_logger = get_contextual_logger("webhook_handler", event_id=event.id, event_type=event.event_type, bridge_item_id=item_id, user_uuid=user_uuid)

    if not item_id:
        ctx_logger.error(f"Missing item_id in item.refreshed webhook: {content}")
        raise ValueError("Missing item_id in item.refreshed event")

    ctx_logger.info(f"Traitement item.refreshed: item {item_id}, status={status_code}, full_refresh={full_refresh}")

    # Trouver l'item SQL correspondant
    sync_item = db.query(SyncItem).filter(SyncItem.bridge_item_id == item_id).first()

    if not sync_item:
        # Si l'item n'existe pas, essayer de le créer
        ctx_logger.warning(f"SyncItem non trouvé pour item_id={item_id}. Tentative de création.")
        if user_uuid:
            user_id = await _find_user_id_from_uuid(db, user_uuid)
            if user_id:
                # Ajouter des informations pour la création
                item_data_for_creation = {
                    "status": status_code,
                    "status_code_info": status_code_info,
                    "status_code_description": status_description,
                }
                from sync_service.sync_manager.orchestrator import create_or_update_sync_item
                sync_item = await create_or_update_sync_item(db, user_id, item_id, item_data_for_creation)
                ctx_logger.info(f"SyncItem créé dynamiquement suite à item.refreshed: id={sync_item.id}")
            else:
                raise ValueError(f"User not found for user_uuid {user_uuid} while handling refreshed event for unknown item {item_id}")
        else:
            raise ValueError(f"Cannot create SyncItem for {item_id}: user_uuid missing in refreshed event")

    # Mettre à jour le statut de l'item en SQL
    from sync_service.sync_manager.item_handler import update_item_status
    sync_item = await update_item_status(db, sync_item, status_code, status_code_info, status_description)
    ctx_logger.info(f"Statut SQL de l'item mis à jour.")

    # Si le rafraîchissement est réussi (status OK), déclencher une synchronisation complète
    if status_code == 0:
        ctx_logger.info(f"Item {item_id} rafraîchi avec succès (status=0). Déclenchement de la synchronisation complète.")
        from sync_service.sync_manager.orchestrator import trigger_full_sync_for_item
        sync_result = await trigger_full_sync_for_item(db, sync_item)
        ctx_logger.info(f"Résultat de la synchronisation complète déclenchée par item.refreshed: {sync_result.get('status')}")
    else:
        ctx_logger.warning(f"Item {item_id} rafraîchi avec un statut non-OK ({status_code}). Pas de synchronisation complète déclenchée.")

async def handle_account_updated(db: Session, event: WebhookEvent) -> None:
    """
    Gère la mise à jour d'un compte.
    
    Args:
        db: Session de base de données
        event: Événement webhook de type item.account.updated
    """
    content = event.event_content
    account_id = content.get("account_id")
    user_uuid = content.get("user_uuid")
    item_id = content.get("item_id")
    ctx_logger = get_contextual_logger("webhook_handler", event_id=event.id, event_type=event.event_type, 
                                       bridge_account_id=account_id, bridge_item_id=item_id, user_uuid=user_uuid)

    if not account_id or not item_id or not user_uuid:
        ctx_logger.error(f"Données manquantes (account_id, item_id ou user_uuid) dans webhook account.updated: {content}")
        raise ValueError("Missing account_id, item_id or user_uuid in account.updated event")

    ctx_logger.info(f"Traitement account.updated: compte {account_id}, item {item_id}")

    # 1. Trouver l'utilisateur interne
    user_id = await _find_user_id_from_uuid(db, user_uuid)
    if not user_id:
        raise ValueError(f"User not found for user_uuid: {user_uuid}")
    ctx_logger.debug(f"Utilisateur interne trouvé: {user_id}")

    # 2. Trouver l'item SQL correspondant
    sync_item = db.query(SyncItem).filter(SyncItem.bridge_item_id == item_id, SyncItem.user_id == user_id).first()
    if not sync_item:
        ctx_logger.warning(f"SyncItem {item_id} non trouvé pour user {user_id} lors de account.updated.")
        raise ValueError(f"SyncItem {item_id} not found for user {user_id} when handling account update")
    ctx_logger.debug(f"SyncItem SQL trouvé: id={sync_item.id}")

    # 3. Trouver ou créer le compte SQL
    sync_account = db.query(SyncAccount).filter(SyncAccount.bridge_account_id == account_id).first()
    if not sync_account:
        ctx_logger.warning(f"SyncAccount SQL non trouvé pour {account_id}. Création.")
        sync_account = SyncAccount(
            item_id=sync_item.id,
            bridge_account_id=account_id,
            account_name=f"Compte {account_id}",  # Nom temporaire
            account_type="unknown"  # Type temporaire
        )
        db.add(sync_account)
        db.commit()
        db.refresh(sync_account)
        ctx_logger.info(f"SyncAccount SQL créé: id={sync_account.id}")
    else:
        ctx_logger.debug(f"SyncAccount SQL trouvé: id={sync_account.id}")

    # 4. Déclencher la synchronisation des transactions pour CE compte
    ctx_logger.info(f"Déclenchement de la synchronisation des transactions pour le compte {account_id}")
    
    try:
        from sync_service.sync_manager.transaction_handler import sync_account_transactions
        tx_sync_result = await sync_account_transactions(db, sync_account)
        ctx_logger.info(f"Résultat de la synchronisation des transactions du compte {account_id}: {tx_sync_result.get('status')}")
    except Exception as tx_error:
        ctx_logger.error(f"Erreur lors de la synchronisation des transactions pour le compte {account_id}: {tx_error}", exc_info=True)

async def handle_account_created(db: Session, event: WebhookEvent) -> None:
    """
    Gère la création d'un nouveau compte.
    
    Args:
        db: Session de base de données
        event: Événement webhook de type item.account.created
    """
    content = event.event_content
    account_id = content.get("account_id")
    item_id = content.get("item_id")
    user_uuid = content.get("user_uuid")
    ctx_logger = get_contextual_logger("webhook_handler", event_id=event.id, event_type=event.event_type, 
                                      bridge_account_id=account_id, bridge_item_id=item_id, user_uuid=user_uuid)

    if not account_id or not item_id or not user_uuid:
        ctx_logger.error(f"Données manquantes (account_id, item_id ou user_uuid) dans webhook account.created")
        raise ValueError("Missing account_id, item_id or user_uuid in account.created event")

    ctx_logger.info(f"Traitement account.created: compte {account_id} pour item {item_id}")

    # 1. Trouver l'utilisateur interne
    user_id = await _find_user_id_from_uuid(db, user_uuid)
    if not user_id:
        raise ValueError(f"User not found for user_uuid: {user_uuid}")
    ctx_logger.debug(f"Utilisateur interne trouvé: {user_id}")

    # 2. Trouver l'item SQL correspondant
    sync_item = db.query(SyncItem).filter(SyncItem.bridge_item_id == item_id, SyncItem.user_id == user_id).first()
    if not sync_item:
        # Créer l'item s'il n'existe pas encore
        ctx_logger.warning(f"SyncItem {item_id} non trouvé pour user {user_id} lors de account.created. Tentative de création.")
        item_data_for_creation = {"status": -2}  # Statut initial "en cours"
        from sync_service.sync_manager.orchestrator import create_or_update_sync_item
        sync_item = await create_or_update_sync_item(db, user_id, item_id, item_data_for_creation)
        ctx_logger.info(f"SyncItem créé dynamiquement suite à account.created: id={sync_item.id}")

    # 3. Vérifier si le compte SQL existe déjà
    sync_account = db.query(SyncAccount).filter(SyncAccount.bridge_account_id == account_id).first()
    if sync_account:
        ctx_logger.warning(f"SyncAccount SQL {account_id} existe déjà lors d'un événement account.created. Mise à jour.")
        # Mettre à jour les infos de base si fournies dans le webhook
        sync_account.account_name = content.get("name", sync_account.account_name)
        sync_account.account_type = content.get("type", sync_account.account_type)
        db.add(sync_account)
    else:
        # Créer le nouveau compte SQL
        ctx_logger.info(f"Création du nouveau SyncAccount SQL pour {account_id}")
        sync_account = SyncAccount(
            item_id=sync_item.id,
            bridge_account_id=account_id,
            account_name=content.get("name", f"Compte {account_id}"),
            account_type=content.get("type", "unknown")
        )
        db.add(sync_account)

    db.commit()
    db.refresh(sync_account)
    ctx_logger.info(f"SyncAccount SQL créé/mis à jour: id={sync_account.id}")

    # 5. Déclencher la synchronisation des transactions pour ce nouveau compte
    ctx_logger.info(f"Déclenchement de la synchronisation des transactions pour le nouveau compte {account_id}")
    
    try:
        from sync_service.sync_manager.transaction_handler import sync_account_transactions
        tx_sync_result = await sync_account_transactions(db, sync_account)
        ctx_logger.info(f"Résultat de la synchronisation initiale des transactions du compte {account_id}: {tx_sync_result.get('status')}")
    except Exception as tx_error:
        ctx_logger.error(f"Erreur lors de la synchronisation initiale des transactions pour compte {account_id}: {tx_error}", exc_info=True)