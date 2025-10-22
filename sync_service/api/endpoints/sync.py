"""
Endpoints pour la gestion de la synchronisation des données.

Ce module expose les endpoints pour démarrer, surveiller et gérer
la synchronisation des données financières depuis Bridge API.
"""

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.orm import Session
from typing import Dict, Any, List
from datetime import datetime

from db_service.session import get_db
from user_service.api.deps import get_current_active_user
from db_service.models.user import User, BridgeConnection
from db_service.models.sync import SyncItem
import logging

# Import des services de gestion de la synchronisation
from sync_service.sync_manager.orchestrator import trigger_full_sync_for_item
from sync_service.sync_manager.utils import get_global_sync_status
from sync_service.sync_manager.item_handler import create_reconnect_session

router = APIRouter()
logger = logging.getLogger(__name__)


async def require_bridge_connection(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
) -> User:
    """
    Dépendance pour vérifier qu'une connexion Bridge existe.

    Lève une HTTPException 428 (Precondition Required) si aucune connexion n'existe.
    L'utilisateur doit d'abord connecter son compte Bridge via POST /users/bridge/connect.
    """
    bridge_connection = db.query(BridgeConnection).filter(
        BridgeConnection.user_id == current_user.id
    ).first()

    if not bridge_connection:
        logger.warning(f"User {current_user.id} attempted sync operation without Bridge connection")
        raise HTTPException(
            status_code=status.HTTP_428_PRECONDITION_REQUIRED,
            detail="Bridge connection required. Please connect your bank account first using POST /users/bridge/connect"
        )

    return current_user

@router.get("/status")
async def get_sync_status(
    current_user: User = Depends(require_bridge_connection),
    db: Session = Depends(get_db)
):
    """
    Récupère l'état actuel de synchronisation pour l'utilisateur.

    Nécessite une connexion Bridge active.

    Returns:
        Dict: État global de synchronisation incluant statuts SQL et vectoriels
    """
    return await get_global_sync_status(db, current_user.id)

@router.post("/refresh")
async def refresh_sync(
    background_tasks: BackgroundTasks,
    current_user: User = Depends(require_bridge_connection),
    db: Session = Depends(get_db)
):
    """
    Déclenche une nouvelle synchronisation pour tous les items de l'utilisateur.
    Vérifie d'abord si les items existent, les crée si nécessaire, puis lance la synchronisation.

    Nécessite une connexion Bridge active.

    Returns:
        Dict: Statut de démarrage de la synchronisation
    """
    # Récupérer les items actifs existants de l'utilisateur
    items = db.query(SyncItem).filter(
        SyncItem.user_id == current_user.id,
        SyncItem.status == 0  # Seulement les items sans erreur
    ).all()
    
    # Si aucun item actif n'est trouvé, vérifier les items Bridge et les créer
    if not items:
        try:
            # Obtenir le token Bridge
            from user_service.services.bridge import get_bridge_token, get_bridge_items
            from sync_service.sync_manager.orchestrator import create_or_update_sync_item
            
            token_data = await get_bridge_token(db, current_user.id)
            access_token = token_data["access_token"]
            
            # Récupérer les items depuis Bridge API
            bridge_items = await get_bridge_items(db, current_user.id, access_token)
            
            if not bridge_items:
                return {
                    "status": "warning",
                    "message": "No items found in Bridge API. Please connect a bank account first."
                }
            
            # Créer les items dans la base de données locale
            created_items = []
            for item_data in bridge_items:
                item = await create_or_update_sync_item(db, current_user.id, item_data["id"], item_data)
                if item.status == 0:  # Seulement les items actifs
                    created_items.append(item)
            
            if not created_items:
                return {
                    "status": "warning",
                    "message": "Items found in Bridge API, but none are in active state."
                }
            
            items = created_items
            
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Error synchronizing items from Bridge: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "message": f"Failed to synchronize items from Bridge API: {str(e)}"
            }
    
    # Démarrer la synchronisation en arrière-plan pour tous les items récupérés/créés
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
    current_user: User = Depends(require_bridge_connection),
    db: Session = Depends(get_db)
):
    """
    Crée une session de reconnexion pour un item en erreur.

    Nécessite une connexion Bridge active.

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
    current_user: User = Depends(require_bridge_connection),
    db: Session = Depends(get_db)
):
    """
    Déclenche une synchronisation pour un seul item spécifique.

    Nécessite une connexion Bridge active.

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