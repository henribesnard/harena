# sync_service/api/endpoints/sync.py
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import Dict, Any, List

from user_service.db.session import get_db
from user_service.api.deps import get_current_active_user
from user_service.models.user import User
from sync_service.models.sync import SyncItem, SyncAccount
from sync_service.services import sync_manager, transaction_sync

router = APIRouter()

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
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Déclenche une nouvelle synchronisation pour tous les items de l'utilisateur.
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
    
    # Déclencher la synchronisation pour chaque item
    for item in items:
        await sync_manager.trigger_full_sync_for_item(db, item)
    
    return {
        "status": "success",
        "message": f"Sync initiated for {len(items)} items"
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