"""
Gestionnaire de webhooks Bridge pour Harena.

Ce module gère la réception et le traitement des événements webhook envoyés par Bridge API.
"""

import json
import hmac
import hashlib
import logging
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from typing import Dict, Any, Optional

from fastapi import HTTPException, status

# Imports Models
from sync_service.models.sync import WebhookEvent, SyncItem, SyncAccount
from user_service.models.user import User, BridgeConnection

# Imports Services (éviter les importations circulaires)
# sync_manager et transaction_sync seront importés au besoin dans les fonctions
from config_service.config import settings

# Import Vector Storage Service (avec gestion d'erreur)
try:
    from sync_service.services.vector_storage import VectorStorageService
    VECTOR_STORAGE_AVAILABLE = True
except ImportError as e:
    VECTOR_STORAGE_AVAILABLE = False
    # Classe factice si le service est indisponible
    class VectorStorageService:
        async def store_account(self, *args, **kwargs): 
            logging.getLogger(__name__).warning("VectorStorage non dispo: store_account ignoré.")
            return False
        async def check_user_storage_initialized(self, *args, **kwargs): return False
        async def initialize_user_storage(self, *args, **kwargs): pass

# Configuration du logger
logger = logging.getLogger(__name__)

# --- Fonctions Utilitaires ---

def verify_webhook_signature(payload: str, signature_header: str, secret: str) -> bool:
    """Vérifier la signature du webhook Bridge."""
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


async def _find_user_id_from_uuid(db: Session, user_uuid: str) -> Optional[int]:
    """Trouve l'ID utilisateur interne à partir de l'UUID Bridge."""
    bridge_connection = db.query(BridgeConnection).filter(
        BridgeConnection.bridge_user_uuid == user_uuid
    ).first()
    if bridge_connection:
        return bridge_connection.user_id
    logger.error(f"Aucune connexion Bridge trouvée pour user_uuid: {user_uuid}")
    return None


async def _update_vector_account(db: Session, user_id: int, bridge_account_id: int, item_id: int) -> bool:
    """Helper pour récupérer les détails d'un compte et le stocker/mettre à jour dans Qdrant."""
    if not VECTOR_STORAGE_AVAILABLE:
        logger.warning(f"_update_vector_account: Vector storage indisponible pour compte {bridge_account_id}.")
        return False

    logger.debug(f"Mise à jour vectorielle demandée pour compte {bridge_account_id}, user {user_id}")
    try:
        # Importer bridge uniquement au besoin pour éviter les dépendances circulaires
        from user_service.services.bridge import get_bridge_accounts
        
        # 1. Récupérer les détails du compte depuis Bridge
        all_item_accounts = await get_bridge_accounts(db, user_id, item_id=item_id)
        
        account_details = None
        for acc in all_item_accounts:
            if acc.get("id") == bridge_account_id:
                account_details = acc
                break

        if not account_details:
            logger.error(f"Impossible de récupérer les détails du compte {bridge_account_id} depuis Bridge pour l'item {item_id}.")
            return False

        # 2. Préparer les données pour le stockage vectoriel
        account_to_store = {
            **account_details,
            "user_id": user_id,
            "item_id": item_id,
            "bridge_account_id": bridge_account_id,
            "bridge_updated_at": account_details.get("updated_at")
        }

        # 3. Stocker/Mettre à jour dans Qdrant
        vector_storage = VectorStorageService()
        success = await vector_storage.store_account(account_to_store)
        if success:
            logger.info(f"Compte {bridge_account_id} stocké/mis à jour dans le Vector Store.")
        else:
            logger.error(f"Échec du stockage/mise à jour vectoriel pour le compte {bridge_account_id}.")
        return success

    except HTTPException as http_exc:
        logger.error(f"Erreur HTTP lors de la récupération/mise à jour vectorielle du compte {bridge_account_id}: {http_exc.detail}")
        return False
    except Exception as e:
        logger.error(f"Erreur inattendue lors de la mise à jour vectorielle du compte {bridge_account_id}: {e}", exc_info=True)
        return False


# --- Handlers de Webhook ---

async def handle_item_created(db: Session, event: WebhookEvent) -> None:
    """Gérer la création d'un nouvel item."""
    content = event.event_content
    item_id = content.get("item_id")
    user_uuid = content.get("user_uuid")
    ctx_logger = logging.LoggerAdapter(logger, {"event_id": event.id, "event_type": event.event_type, "bridge_item_id": item_id, "user_uuid": user_uuid})

    if not item_id or not user_uuid:
        ctx_logger.error(f"Données manquantes (item_id ou user_uuid) dans webhook item.created: {content}")
        raise ValueError("Missing item_id or user_uuid in item.created event")

    ctx_logger.info(f"Traitement item.created: item {item_id} pour user {user_uuid}")

    user_id = await _find_user_id_from_uuid(db, user_uuid)
    if not user_id:
        raise ValueError(f"User not found for user_uuid: {user_uuid}")

    ctx_logger.info(f"Utilisateur interne trouvé: user_id={user_id}")

    # Importer sync_manager ici pour éviter les dépendances circulaires
    from sync_service.services.sync_manager import create_or_update_sync_item
    
    # Créer ou mettre à jour le SyncItem en SQL
    sync_item = await create_or_update_sync_item(db, user_id, item_id, content)
    ctx_logger.info(f"SyncItem SQL créé/mis à jour: id={sync_item.id}")


async def handle_item_refreshed(db: Session, event: WebhookEvent) -> None:
    """Gérer le rafraîchissement d'un item."""
    content = event.event_content
    item_id = content.get("item_id")
    status_code = content.get("status", 0)
    status_code_info = content.get("status_code_info")
    status_description = content.get("status_code_description")
    full_refresh = content.get("full_refresh", False)
    user_uuid = content.get("user_uuid")
    ctx_logger = logging.LoggerAdapter(logger, {"event_id": event.id, "event_type": event.event_type, "bridge_item_id": item_id, "user_uuid": user_uuid})

    if not item_id:
        ctx_logger.error(f"Missing item_id in item.refreshed webhook: {content}")
        raise ValueError("Missing item_id in item.refreshed event")

    ctx_logger.info(f"Traitement item.refreshed: item {item_id}, status={status_code}, full_refresh={full_refresh}")

    # Importer sync_manager ici pour éviter les dépendances circulaires
    from sync_service.services.sync_manager import update_item_status, create_or_update_sync_item, trigger_full_sync_for_item

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
                sync_item = await create_or_update_sync_item(db, user_id, item_id, item_data_for_creation)
                ctx_logger.info(f"SyncItem créé dynamiquement suite à item.refreshed: id={sync_item.id}")
            else:
                raise ValueError(f"User not found for user_uuid {user_uuid} while handling refreshed event for unknown item {item_id}")
        else:
            raise ValueError(f"Cannot create SyncItem for {item_id}: user_uuid missing in refreshed event")

    # Mettre à jour le statut de l'item en SQL
    sync_item = await update_item_status(db, sync_item, status_code, status_code_info, status_description)
    ctx_logger.info(f"Statut SQL de l'item mis à jour.")

    # Si le rafraîchissement est réussi (status OK), déclencher une synchronisation complète
    if status_code == 0:
        ctx_logger.info(f"Item {item_id} rafraîchi avec succès (status=0). Déclenchement de la synchronisation complète.")
        sync_result = await trigger_full_sync_for_item(db, sync_item)
        ctx_logger.info(f"Résultat de la synchronisation complète déclenchée par item.refreshed: {sync_result.get('status')}")
    else:
        ctx_logger.warning(f"Item {item_id} rafraîchi avec un statut non-OK ({status_code}). Pas de synchronisation complète déclenchée.")
        # Envisager ici des actions de retry ou de nettoyage suivant le code d'erreur


async def handle_account_updated(db: Session, event: WebhookEvent) -> None:
    """Gérer la mise à jour d'un compte."""
    content = event.event_content
    account_id = content.get("account_id")
    user_uuid = content.get("user_uuid")
    item_id = content.get("item_id")
    ctx_logger = logging.LoggerAdapter(logger, {"event_id": event.id, "event_type": event.event_type, "bridge_account_id": account_id, "bridge_item_id": item_id, "user_uuid": user_uuid})

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

    # 4. Mettre à jour le compte dans le Vector Store
    vector_update_success = await _update_vector_account(db, user_id, account_id, item_id)
    if not vector_update_success:
        ctx_logger.error(f"Échec de la mise à jour vectorielle pour le compte {account_id}.")

    # 5. Déclencher la synchronisation des transactions pour CE compte
    ctx_logger.info(f"Déclenchement de la synchronisation des transactions pour le compte {account_id}")
    
    # Importer transaction_sync ici pour éviter les dépendances circulaires
    from sync_service.services.transaction_sync import sync_account_transactions
    
    try:
        tx_sync_result = await sync_account_transactions(db, sync_account)
        ctx_logger.info(f"Résultat de la synchronisation des transactions du compte {account_id}: {tx_sync_result.get('status')}")
    except Exception as tx_error:
        ctx_logger.error(f"Erreur lors de la synchronisation des transactions pour le compte {account_id}: {tx_error}", exc_info=True)
        # Ne pas relancer l'exception ici pour que le webhook soit acquitté quand même


async def handle_account_created(db: Session, event: WebhookEvent) -> None:
    """Gérer la création d'un nouveau compte."""
    content = event.event_content
    account_id = content.get("account_id")
    item_id = content.get("item_id")
    user_uuid = content.get("user_uuid")
    ctx_logger = logging.LoggerAdapter(logger, {"event_id": event.id, "event_type": event.event_type, "bridge_account_id": account_id, "bridge_item_id": item_id, "user_uuid": user_uuid})

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
        # Importer sync_manager ici pour éviter les dépendances circulaires
        from sync_service.services.sync_manager import create_or_update_sync_item
        
        ctx_logger.warning(f"SyncItem {item_id} non trouvé pour user {user_id} lors de account.created. Tentative de création.")
        item_data_for_creation = {"status": -2}  # Statut initial "en cours"
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

    # 4. Stocker le nouveau compte dans le Vector Store
    vector_store_success = await _update_vector_account(db, user_id, account_id, item_id)
    if not vector_store_success:
        ctx_logger.error(f"Échec du stockage vectoriel initial pour le nouveau compte {account_id}.")

    # 5. Déclencher la synchronisation des transactions pour ce nouveau compte
    ctx_logger.info(f"Déclenchement de la synchronisation des transactions pour le nouveau compte {account_id}")
    
    # Importer transaction_sync ici pour éviter les dépendances circulaires
    from sync_service.services.transaction_sync import sync_account_transactions
    
    try:
        tx_sync_result = await sync_account_transactions(db, sync_account)
        ctx_logger.info(f"Résultat de la synchronisation initiale des transactions du compte {account_id}: {tx_sync_result.get('status')}")
    except Exception as tx_error:
        ctx_logger.error(f"Erreur lors de la synchronisation initiale des transactions pour compte {account_id}: {tx_error}", exc_info=True)


# --- Fonction Principale de Traitement ---

async def process_webhook(db: Session, webhook_data: Dict[str, Any], signature: str = None) -> WebhookEvent:
    """Traiter un événement webhook reçu de Bridge API."""
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

    # Traiter selon le type d'événement
    handler = None
    if event.event_type == "item.created":
        handler = handle_item_created
    elif event.event_type == "item.refreshed":
        handler = handle_item_refreshed
    elif event.event_type == "item.account.updated":
        handler = handle_account_updated
    elif event.event_type == "item.account.created":
        handler = handle_account_created
    elif event.event_type == "TEST_EVENT":
        logger.info(f"Webhook de test reçu et traité avec succès (id={event.id}).")
        # Marquer comme traité directement
        event.processed = True
        event.processing_timestamp = datetime.now(timezone.utc)
        db.add(event)
        db.commit()
        return event  # Pas besoin d'appeler de handler spécifique
    else:
        logger.warning(f"Type d'événement webhook non géré: {event.event_type} (id={event.id})")
        # Marquer comme traité pour ne pas le réessayer indéfiniment
        event.processed = True
        event.error_message = f"Unhandled event type: {event.event_type}"
        event.processing_timestamp = datetime.now(timezone.utc)
        db.add(event)
        db.commit()
        return event  # Pas de handler à appeler

    # Appeler le handler approprié
    if handler:
        try:
            logger.info(f"Appel du handler '{handler.__name__}' pour l'événement id={event.id}")
            await handler(db, event)
            event.processed = True
            logger.info(f"Traitement réussi pour l'événement id={event.id} par {handler.__name__}")
        except ValueError as ve:  # Erreurs de données manquantes
            logger.error(f"Erreur de données lors du traitement du webhook id={event.id} par {handler.__name__}: {ve}")
            event.error_message = f"Data error: {ve}"
            event.processed = True  # Marquer comme traité car l'erreur est liée aux données
        except HTTPException as he:  # Erreurs venant des appels API
            logger.error(f"Erreur HTTP {he.status_code} lors du traitement du webhook id={event.id} par {handler.__name__}: {he.detail}")
            event.error_message = f"HTTP error {he.status_code}: {he.detail}"
            # Pour les erreurs serveur (5xx), marquer comme non traité pour retry possible
            event.processed = he.status_code < 500  # Marquer comme traité si erreur 4xx
        except Exception as e:
            logger.error(f"Erreur inattendue lors du traitement du webhook id={event.id} par {handler.__name__}: {e}", exc_info=True)
            event.error_message = f"Unexpected error: {str(e)}"
            # Marquer comme traité pour éviter les boucles d'erreur infinies
            event.processed = True
        finally:
            # Toujours enregistrer l'état final du traitement
            if not event.processing_timestamp:
                event.processing_timestamp = datetime.now(timezone.utc)
            db.add(event)
            db.commit()

    return event