"""
Gestionnaire des items de connexion Bridge.

Ce module gère la récupération et la mise à jour des items Bridge
dans la base de données SQL.
"""

import logging
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from typing import Dict, List, Any, Optional

from db_service.models.sync import SyncItem
from sync_service.utils.logging import get_contextual_logger

logger = logging.getLogger(__name__)

async def get_bridge_item_info(
    db: Session,
    sync_item: SyncItem,
    access_token: str
) -> Optional[Dict[str, Any]]:
    """
    Récupère les informations à jour d'un item depuis Bridge API.
    
    Args:
        db: Session de base de données
        sync_item: Item de synchronisation
        access_token: Token d'accès Bridge
        
    Returns:
        Dict: Informations de l'item ou None si erreur
    """
    user_id = sync_item.user_id
    bridge_item_id = sync_item.bridge_item_id
    ctx_logger = get_contextual_logger("sync_service.item_handler", user_id=user_id, bridge_item_id=bridge_item_id)
    ctx_logger.info(f"Récupération des informations de l'item {bridge_item_id}")
    
    try:
        # Récupérer les informations de l'item depuis Bridge API
        from user_service.services.bridge import get_bridge_item
        item_info = await get_bridge_item(db, user_id, access_token, bridge_item_id)
        
        if not item_info:
            ctx_logger.warning(f"Aucune information récupérée pour l'item {bridge_item_id}")
            return None
            
        ctx_logger.info(f"Informations de l'item {bridge_item_id} récupérées avec succès")
        return item_info
    except Exception as e:
        ctx_logger.error(f"Erreur lors de la récupération des informations de l'item {bridge_item_id}: {e}", exc_info=True)
        return None

async def update_item_status(
    db: Session,
    sync_item: SyncItem,
    status: int,
    status_code_info: Optional[str] = None,
    status_description: Optional[str] = None
) -> SyncItem:
    """
    Met à jour le statut d'un item de synchronisation.
    
    Args:
        db: Session de base de données
        sync_item: Item de synchronisation
        status: Nouveau statut
        status_code_info: Information sur le code de statut
        status_description: Description du statut
        
    Returns:
        SyncItem: Item de synchronisation mis à jour
    """
    user_id = sync_item.user_id
    bridge_item_id = sync_item.bridge_item_id
    ctx_logger = get_contextual_logger("sync_service.item_handler", user_id=user_id, bridge_item_id=bridge_item_id)
    
    try:
        # Déterminer si le statut nécessite une action utilisateur
        needs_action = status in [402, 429, 1010]  # Statuts nécessitant une action utilisateur
        
        # Mettre à jour les champs
        sync_item.status = status
        sync_item.status_code_info = status_code_info or sync_item.status_code_info
        sync_item.status_description = status_description or sync_item.status_description
        sync_item.needs_user_action = needs_action
        
        # Mise à jour des timestamps
        now = datetime.now(timezone.utc)
        sync_item.last_try_refresh = now
        
        # Si le statut est OK (0), mettre à jour le timestamp de succès
        if status == 0:
            sync_item.last_successful_refresh = now
            
        # Sauvegarder les changements
        db.add(sync_item)
        db.commit()
        db.refresh(sync_item)
        
        ctx_logger.info(f"Statut de l'item mis à jour: {status} (needs_action={needs_action})")
        return sync_item
    except Exception as e:
        db.rollback()
        ctx_logger.error(f"Erreur lors de la mise à jour du statut de l'item: {e}", exc_info=True)
        raise

async def refresh_bridge_item(
    db: Session,
    user_id: int,
    bridge_item_id: int
) -> Dict[str, Any]:
    """
    Rafraîchit un item Bridge pour récupérer de nouvelles données.
    
    Args:
        db: Session de base de données
        user_id: ID de l'utilisateur
        bridge_item_id: ID de l'item Bridge
        
    Returns:
        Dict: Résultat du rafraîchissement
    """
    ctx_logger = get_contextual_logger("sync_service.item_handler", user_id=user_id, bridge_item_id=bridge_item_id)
    ctx_logger.info(f"Rafraîchissement de l'item {bridge_item_id}")
    
    try:
        # Récupérer le token Bridge
        from user_service.services.bridge import get_bridge_token
        token_data = await get_bridge_token(db, user_id)
        access_token = token_data["access_token"]
        
        # Rafraîchir l'item via Bridge API
        from user_service.services.bridge import refresh_bridge_item
        refresh_result = await refresh_bridge_item(db, user_id, access_token, bridge_item_id)
        
        if not refresh_result:
            ctx_logger.warning(f"Aucun résultat de rafraîchissement pour l'item {bridge_item_id}")
            return {"status": "error", "message": "No refresh result returned"}
            
        # Mise à jour du SyncItem
        sync_item = db.query(SyncItem).filter(
            SyncItem.user_id == user_id,
            SyncItem.bridge_item_id == bridge_item_id
        ).first()
        
        if sync_item:
            # Extraire le statut du résultat Bridge
            status = refresh_result.get("status", -1)
            status_code_info = refresh_result.get("status_code_info")
            status_description = refresh_result.get("status_code_description")
            
            # Mettre à jour le statut
            sync_item = await update_item_status(
                db, 
                sync_item, 
                status, 
                status_code_info, 
                status_description
            )
            
            ctx_logger.info(f"Item {bridge_item_id} rafraîchi avec statut {status}")
            
            return {
                "status": "success",
                "item_status": status,
                "needs_action": sync_item.needs_user_action,
                "refresh_result": refresh_result
            }
        else:
            ctx_logger.warning(f"SyncItem {bridge_item_id} non trouvé lors du rafraîchissement")
            return {"status": "error", "message": "SyncItem not found"}
    except Exception as e:
        ctx_logger.error(f"Erreur lors du rafraîchissement de l'item {bridge_item_id}: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}

async def create_reconnect_session(
    db: Session,
    user_id: int,
    bridge_item_id: int
) -> str:
    """
    Crée une session de reconnexion pour un item en erreur.
    
    Args:
        db: Session de base de données
        user_id: ID de l'utilisateur
        bridge_item_id: ID de l'item Bridge
        
    Returns:
        str: URL de reconnexion
    """
    ctx_logger = get_contextual_logger("sync_service.item_handler", user_id=user_id, bridge_item_id=bridge_item_id)
    ctx_logger.info(f"Création d'une session de reconnexion pour l'item {bridge_item_id}")
    
    # Vérifier que l'item existe et appartient à l'utilisateur
    sync_item = db.query(SyncItem).filter(
        SyncItem.user_id == user_id,
        SyncItem.bridge_item_id == bridge_item_id
    ).first()
    
    if not sync_item:
        ctx_logger.error(f"SyncItem {bridge_item_id} non trouvé pour user {user_id}")
        raise ValueError(f"Item {bridge_item_id} not found for user {user_id}")
    
    try:
        # Récupérer le token Bridge
        from user_service.services.bridge import get_bridge_token
        token_data = await get_bridge_token(db, user_id)
        access_token = token_data["access_token"]
        
        # Créer la session de reconnexion via Bridge API
        from user_service.services.bridge import create_bridge_reconnect_session
        reconnect_result = await create_bridge_reconnect_session(db, user_id, access_token, bridge_item_id)
        
        if not reconnect_result or "redirect_url" not in reconnect_result:
            ctx_logger.error(f"Résultat de reconnexion invalide pour l'item {bridge_item_id}")
            raise ValueError("Invalid reconnect result")
            
        redirect_url = reconnect_result["redirect_url"]
        ctx_logger.info(f"Session de reconnexion créée pour l'item {bridge_item_id}")
        
        return redirect_url
    except Exception as e:
        ctx_logger.error(f"Erreur lors de la création de la session de reconnexion: {e}", exc_info=True)
        raise