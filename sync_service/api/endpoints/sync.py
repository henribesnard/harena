"""
Endpoints pour la gestion de la synchronisation des données.

Ce module expose les endpoints pour démarrer, surveiller et gérer
la synchronisation des données financières depuis Bridge API.
"""

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.orm import Session
from typing import Dict, Any, List
from datetime import datetime

from user_service.db.session import get_db
from user_service.api.deps import get_current_active_user
from user_service.models.user import User
from sync_service.models.sync import SyncItem
import logging

# Import des services de gestion de la synchronisation
from sync_service.sync_manager.orchestrator import trigger_full_sync_for_item
from sync_service.sync_manager.utils import get_global_sync_status
from sync_service.sync_manager.item_handler import create_reconnect_session

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/status")
async def get_sync_status(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Récupère l'état actuel de synchronisation pour l'utilisateur.
    
    Returns:
        Dict: État global de synchronisation incluant statuts SQL et vectoriels
    """
    return await get_global_sync_status(db, current_user.id)

@router.post("/refresh")
async def refresh_sync(
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Déclenche une nouvelle synchronisation pour tous les items de l'utilisateur.
    La synchronisation s'exécute en arrière-plan pour éviter les timeouts.
    
    Returns:
        Dict: Statut de démarrage de la synchronisation
    """
    # Récupérer tous les items actifs de l'utilisateur
    items = db.query(SyncItem).filter(
        SyncItem.user_id == current_user.id,
        SyncItem.status == 0  # Seulement les items sans erreur
    ).all()
    
    if not items:
        return {
            "status": "warning",
            "message": "No active items found to synchronize"
        }
    
    # Démarrer la synchronisation en arrière-plan
    background_tasks.add_task(
        process_sync_background,
        user_id=current_user.id,
        item_ids=[item.id for item in items]
    )
    
    return {
        "status": "success",
        "message": f"Sync initiated for {len(items)} items",
        "items_count": len(items),
        "item_ids": [item.bridge_item_id for item in items],
        "info": "The sync is now running in the background and will continue after this response is sent."
    }

@router.post("/reconnect/{bridge_item_id}")
async def reconnect_item(
    bridge_item_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Crée une session de reconnexion pour un item en erreur.
    
    Args:
        bridge_item_id: ID de l'item Bridge à reconnecter
        
    Returns:
        Dict: URL de reconnexion à utiliser par le frontend
    """
    try:
        connect_url = await create_reconnect_session(
            db, current_user.id, bridge_item_id
        )
        
        return {
            "status": "success",
            "connect_url": connect_url
        }
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Failed to create reconnect session: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create reconnect session: {str(e)}"
        )

@router.post("/sync-item/{bridge_item_id}")
async def sync_single_item(
    bridge_item_id: int,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Déclenche une synchronisation pour un seul item spécifique.
    
    Args:
        bridge_item_id: ID de l'item Bridge à synchroniser
        
    Returns:
        Dict: Statut de démarrage de la synchronisation
    """
    # Récupérer l'item spécifié
    item = db.query(SyncItem).filter(
        SyncItem.user_id == current_user.id,
        SyncItem.bridge_item_id == bridge_item_id
    ).first()
    
    if not item:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Item with id {bridge_item_id} not found"
        )
    
    # Démarrer la synchronisation en arrière-plan
    background_tasks.add_task(
        process_sync_background,
        user_id=current_user.id,
        item_ids=[item.id]
    )
    
    return {
        "status": "success",
        "message": f"Sync initiated for item {bridge_item_id}",
        "info": "The sync is now running in the background and will continue after this response is sent."
    }

# Fonction auxiliaire pour le traitement en arrière-plan
async def process_sync_background(user_id: int, item_ids: List[int]):
    """
    Fonction exécutée en arrière-plan pour traiter la synchronisation complète
    sans bloquer la réponse HTTP.
    
    Args:
        user_id: ID de l'utilisateur
        item_ids: Liste des IDs d'items à synchroniser
    """
    # Créer une nouvelle session car nous sommes dans un thread/coroutine distinct
    from user_service.db.session import SessionLocal
    db = SessionLocal()
    
    try:
        logger.info(f"Starting background sync for user {user_id} with {len(item_ids)} items")
        
        for item_id in item_ids:
            try:
                item = db.query(SyncItem).filter(SyncItem.id == item_id).first()
                if not item:
                    logger.warning(f"Item with ID {item_id} not found during background sync")
                    continue
                
                logger.info(f"Background sync: processing item {item.bridge_item_id} for user {user_id}")
                
                # Exécuter la synchronisation complète
                sync_result = await trigger_full_sync_for_item(db, item)
                
                logger.info(f"Background sync completed for item {item.bridge_item_id}: status={sync_result.get('status')}")
                
                # Log détaillé des différentes étapes de la synchronisation
                for step_name, step_result in sync_result.get("steps", {}).items():
                    logger.info(f"Step {step_name}: {step_result.get('status', 'unknown')}")
                
            except Exception as e:
                logger.error(f"Error during background sync for item {item_id}: {str(e)}", exc_info=True)
        
        logger.info(f"All background syncs completed for user {user_id}")
    except Exception as e:
        logger.error(f"Unexpected error in background sync for user {user_id}: {str(e)}", exc_info=True)
    finally:
        db.close()