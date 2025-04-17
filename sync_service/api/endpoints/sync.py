from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.orm import Session
from typing import Dict, Any, List
from datetime import datetime

from user_service.db.session import get_db
from user_service.api.deps import get_current_active_user
from user_service.models.user import User
from sync_service.models.sync import SyncItem
from sync_service.services import sync_manager
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/status")
async def get_sync_status(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Récupère l'état actuel de synchronisation pour l'utilisateur.
    """
    return await sync_manager.get_user_sync_status(db, current_user.id)

@router.post("/refresh")
async def refresh_sync(
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Déclenche une nouvelle synchronisation pour tous les items de l'utilisateur.
    La synchronisation s'exécute en arrière-plan pour éviter les timeouts.
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
    """
    try:
        connect_url = await sync_manager.create_reconnect_session(
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
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create reconnect session: {str(e)}"
        )

# Fonction auxiliaire pour le traitement en arrière-plan
async def process_sync_background(user_id: int, item_ids: List[int]):
    """
    Fonction exécutée en arrière-plan pour traiter la synchronisation complète
    sans bloquer la réponse HTTP.
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
                
                # Activer le logging détaillé pour voir toutes les collections
                logger.info(f"Background sync: processing item {item.bridge_item_id} for user {user_id}")
                
                # Exécuter la synchronisation complète
                sync_result = await sync_manager.trigger_full_sync_for_item(db, item)
                
                # Log des résultats détaillés de la synchronisation
                logger.info(f"Background sync completed for item {item.bridge_item_id}: status={sync_result.get('status')}")
                
                # Log détaillé pour chaque type de collection
                log_collection_details(sync_result)
                
            except Exception as e:
                logger.error(f"Error during background sync for item {item_id}: {str(e)}", exc_info=True)
        
        logger.info(f"All background syncs completed for user {user_id}")
    except Exception as e:
        logger.error(f"Unexpected error in background sync for user {user_id}: {str(e)}", exc_info=True)
    finally:
        db.close()

def log_collection_details(sync_result: Dict[str, Any]):
    """Log des détails de synchronisation pour chaque type de collection."""
    steps = sync_result.get("steps", {})
    
    # Log pour les comptes
    if "store_vector_accounts" in steps:
        accounts_result = steps["store_vector_accounts"]
        logger.info(f"Accounts sync result: {accounts_result}")
    
    # Log pour les catégories
    if "store_vector_categories" in steps:
        categories_result = steps["store_vector_categories"]
        logger.info(f"Categories sync result: {categories_result}")
    
    # Log pour les insights
    if "store_vector_insights" in steps:
        insights_result = steps["store_vector_insights"]
        logger.info(f"Insights sync result: {insights_result}")
    
    # Log pour les stocks
    if "store_vector_stocks" in steps:
        stocks_result = steps["store_vector_stocks"]
        logger.info(f"Stocks sync result: {stocks_result}")
    
    # Log pour les transactions
    if "sync_transactions" in steps:
        tx_result = steps["sync_transactions"]
        logger.info(f"Transactions sync stats: total_new={tx_result.get('total_new_transactions_stored', 0)}, accounts={tx_result.get('accounts_processed', 0)}")
    
    # Log des statistiques vectorielles finales
    if "final_vector_stats" in sync_result:
        stats = sync_result["final_vector_stats"]
        logger.info(f"Final vector statistics after sync: {stats}")