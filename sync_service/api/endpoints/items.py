"""
Endpoints pour la gestion des items de connexion Bridge.

Ce module expose les endpoints pour gérer les items/connexions bancaires
stockés dans la base de données.
"""

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.orm import Session
from typing import Dict, Any, List, Optional

from conversation_service.api.dependencies import get_db
from user_service.api.deps import get_current_active_user
from db_service.models.user import User
from db_service.models.sync import SyncItem
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/")
async def get_items(
    status: Optional[int] = None,
    needs_action: Optional[bool] = None,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Récupère tous les items/connexions bancaires de l'utilisateur avec filtres optionnels.
    
    Args:
        status: Filtrer par statut de l'item
        needs_action: Filtrer par besoin d'action utilisateur
        
    Returns:
        List: Liste des items
    """
    # Construire la requête de base
    query = db.query(SyncItem).filter(SyncItem.user_id == current_user.id)
    
    # Appliquer les filtres supplémentaires
    if status is not None:
        query = query.filter(SyncItem.status == status)
    
    if needs_action is not None:
        query = query.filter(SyncItem.needs_action == needs_action)
    
    # Exécuter la requête
    items = query.all()
    
    # Formater les résultats
    return [format_item(item) for item in items]

@router.get("/{item_id}")
async def get_item(
    item_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Récupère un item spécifique par son ID.
    
    Args:
        item_id: ID de l'item Bridge
        
    Returns:
        Dict: Détails de l'item
    """
    item = db.query(SyncItem).filter(
        SyncItem.bridge_item_id == item_id,
        SyncItem.user_id == current_user.id
    ).first()
    
    if not item:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Item {item_id} not found"
        )
    
    return format_item(item)

@router.post("/{item_id}/refresh")
async def refresh_item(
    item_id: int,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Rafraîchit un item spécifique.
    
    Args:
        item_id: ID de l'item Bridge à rafraîchir
        
    Returns:
        Dict: Résultat du rafraîchissement
    """
    # Vérifier que l'item appartient à l'utilisateur
    item = db.query(SyncItem).filter(
        SyncItem.bridge_item_id == item_id,
        SyncItem.user_id == current_user.id
    ).first()
    
    if not item:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Item {item_id} not found"
        )
    
    # Rafraîchir l'item via Bridge API
    from sync_service.sync_manager.item_handler import refresh_bridge_item
    result = await refresh_bridge_item(db, current_user.id, item_id)
    
    # Si le rafraîchissement a réussi et que le statut est OK, déclencher une synchro complète
    if result.get("status") == "success" and result.get("item_status") == 0:
        # Lancer la synchronisation complète en arrière-plan
        background_tasks.add_task(
            process_sync_background,
            user_id=current_user.id,
            item_ids=[item.id]
        )
        
        result["sync_initiated"] = True
        result["message"] = "Item refreshed successfully and sync initiated"
    
    return result

@router.delete("/{item_id}")
async def delete_item(
    item_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Supprime un item et toutes ses données associées.
    
    Args:
        item_id: ID de l'item Bridge à supprimer
        
    Returns:
        Dict: Résultat de la suppression
    """
    # Vérifier que l'item appartient à l'utilisateur
    item = db.query(SyncItem).filter(
        SyncItem.bridge_item_id == item_id,
        SyncItem.user_id == current_user.id
    ).first()
    
    if not item:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Item {item_id} not found"
        )
    
    try:
        # Supprimer l'item dans Bridge API
        from user_service.services.bridge import delete_bridge_item
        from user_service.services.bridge import get_bridge_token
        
        token_data = await get_bridge_token(db, current_user.id)
        await delete_bridge_item(db, current_user.id, token_data["access_token"], item_id)
        
        # Supprimer l'item en base de données
        # Grâce aux cascades SQL, cela supprimera aussi les comptes, transactions, etc.
        db.delete(item)
        db.commit()
        
        return {
            "status": "success",
            "message": f"Item {item_id} and all associated data deleted successfully"
        }
    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting item {item_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete item: {str(e)}"
        )

# Fonction auxiliaire pour le traitement en arrière-plan (déjà définie dans sync.py)
async def process_sync_background(user_id: int, item_ids: List[int]):
    """
    Fonction exécutée en arrière-plan pour traiter la synchronisation complète.
    
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
                from sync_service.sync_manager.orchestrator import trigger_full_sync_for_item
                sync_result = await trigger_full_sync_for_item(db, item)
                
                logger.info(f"Background sync completed for item {item.bridge_item_id}: status={sync_result.get('status')}")
                
            except Exception as e:
                logger.error(f"Error during background sync for item {item_id}: {str(e)}", exc_info=True)
        
        logger.info(f"All background syncs completed for user {user_id}")
    except Exception as e:
        logger.error(f"Unexpected error in background sync for user {user_id}: {str(e)}", exc_info=True)
    finally:
        db.close()

# Fonctions utilitaires pour formater les réponses
def format_item(item: SyncItem) -> Dict[str, Any]:
    """
    Formate un item pour la réponse API.
    
    Args:
        item: Item depuis la BDD
        
    Returns:
        Dict: Item formaté
    """
    return {
        "id": item.bridge_item_id,
        "status": item.status,
        "status_code_info": item.status_code_info,
        "status_description": item.status_description,
        "provider_id": item.provider_id,
        "account_types": item.account_types,
        "needs_user_action": item.needs_user_action,
        "last_successful_refresh": item.last_successful_refresh.isoformat() if item.last_successful_refresh else None,
        "last_try_refresh": item.last_try_refresh.isoformat() if item.last_try_refresh else None,
        "created_at": item.created_at.isoformat(),
        "updated_at": item.updated_at.isoformat(),
        "accounts_count": len(item.accounts)
    }