# sync_service/services/sync_manager.py
import logging
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from typing import Dict, Any, List, Optional

from sync_service.models.sync import SyncItem, SyncAccount
from user_service.models.user import User, BridgeConnection
from user_service.services import bridge as bridge_service

logger = logging.getLogger(__name__)

# Liste des statuts qui requièrent une action utilisateur
ACTION_REQUIRED_STATUSES = [402, 429, 1010]

async def create_or_update_sync_item(db: Session, user_id: int, bridge_item_id: int, item_data: Dict[str, Any]) -> SyncItem:
    """Créer ou mettre à jour un item de synchronisation."""
    # Vérifier si l'item existe déjà
    sync_item = db.query(SyncItem).filter(SyncItem.bridge_item_id == bridge_item_id).first()
    
    status = item_data.get("status", 0)
    status_code_info = item_data.get("status_code_info")
    
    if sync_item:
        # Mettre à jour l'item existant
        sync_item.status = status
        sync_item.status_code_info = status_code_info
        sync_item.needs_user_action = status in ACTION_REQUIRED_STATUSES
        
        if "status_code_description" in item_data:
            sync_item.status_description = item_data["status_code_description"]
        
        if "last_successful_refresh" in item_data:
            sync_item.last_successful_refresh = datetime.fromisoformat(
                item_data["last_successful_refresh"].replace('Z', '+00:00')
            )
        
        if "last_try_refresh" in item_data:
            sync_item.last_try_refresh = datetime.fromisoformat(
                item_data["last_try_refresh"].replace('Z', '+00:00')
            )
            
        if "provider_id" in item_data:
            sync_item.provider_id = item_data["provider_id"]
            
        if "account_types" in item_data:
            sync_item.account_types = item_data["account_types"]
    else:
        # Créer un nouvel item
        sync_item = SyncItem(
            user_id=user_id,
            bridge_item_id=bridge_item_id,
            status=status,
            status_code_info=status_code_info,
            status_description=item_data.get("status_code_description"),
            provider_id=item_data.get("provider_id"),
            account_types=item_data.get("account_types"),
            needs_user_action=status in ACTION_REQUIRED_STATUSES
        )
        
        if "last_successful_refresh" in item_data:
            sync_item.last_successful_refresh = datetime.fromisoformat(
                item_data["last_successful_refresh"].replace('Z', '+00:00')
            )
            
        if "last_try_refresh" in item_data:
            sync_item.last_try_refresh = datetime.fromisoformat(
                item_data["last_try_refresh"].replace('Z', '+00:00')
            )
    
    db.add(sync_item)
    db.commit()
    db.refresh(sync_item)
    
    return sync_item

async def update_item_status(db: Session, sync_item: SyncItem, status: int, status_code_info: Optional[str] = None) -> SyncItem:
    """Mettre à jour le statut d'un item."""
    # Mettre à jour les informations de statut
    sync_item.status = status
    if status_code_info:
        sync_item.status_code_info = status_code_info
    
    # Déterminer si une action utilisateur est requise
    sync_item.needs_user_action = status in ACTION_REQUIRED_STATUSES
    
    # Si le statut est OK, mettre à jour la date de dernière synchronisation réussie
    if status == 0:
        sync_item.last_successful_refresh = datetime.now(timezone.utc)
    
    # Dans tous les cas, mettre à jour la date de dernière tentative
    sync_item.last_try_refresh = datetime.now(timezone.utc)
    
    db.add(sync_item)
    db.commit()
    db.refresh(sync_item)
    
    return sync_item

async def trigger_full_sync_for_item(db: Session, sync_item: SyncItem) -> None:
    """Déclencher une synchronisation complète pour un item."""
    # Récupérer les informations d'authentification Bridge
    user_id = sync_item.user_id
    token_data = await bridge_service.get_bridge_token(db, user_id)
    
    # Récupérer les comptes de l'item
    accounts = await fetch_item_accounts(db, sync_item, token_data["access_token"])
    
    # Synchroniser les transactions pour chaque compte
    from sync_service.services.transaction_sync import sync_account_transactions
    
    for account in accounts:
        await sync_account_transactions(db, account)

async def fetch_item_accounts(db: Session, sync_item: SyncItem, access_token: str) -> List[SyncAccount]:
    """Récupérer et mettre à jour les comptes d'un item."""
    import httpx
    from user_service.core.config import settings
    
    # Appel à l'API Bridge pour récupérer les comptes
    url = f"{settings.BRIDGE_API_URL}/aggregation/accounts?item_id={sync_item.bridge_item_id}"
    headers = {
        "accept": "application/json",
        "Bridge-Version": settings.BRIDGE_API_VERSION,
        "Client-Id": settings.BRIDGE_CLIENT_ID,       # Ajout de Client-Id
        "Client-Secret": settings.BRIDGE_CLIENT_SECRET,
        "authorization": f"Bearer {access_token}"
    }
    
    accounts = []
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers)
            
            if response.status_code != 200:
                logger.error(f"Failed to fetch accounts: {response.text}")
                return accounts
                
            accounts_data = response.json()
            
            if "resources" not in accounts_data:
                logger.error(f"Invalid accounts response: {accounts_data}")
                return accounts
                
            # Traiter chaque compte
            for account_data in accounts_data["resources"]:
                bridge_account_id = account_data["id"]
                
                # Vérifier si le compte existe déjà
                sync_account = db.query(SyncAccount).filter(
                    SyncAccount.bridge_account_id == bridge_account_id
                ).first()
                
                if sync_account:
                    # Mettre à jour le compte existant
                    sync_account.account_name = account_data.get("name")
                    sync_account.account_type = account_data.get("type")
                else:
                    # Créer un nouveau compte
                    sync_account = SyncAccount(
                        item_id=sync_item.id,
                        bridge_account_id=bridge_account_id,
                        account_name=account_data.get("name"),
                        account_type=account_data.get("type")
                    )
                
                db.add(sync_account)
                accounts.append(sync_account)
            
            db.commit()
            for account in accounts:
                db.refresh(account)
                
            return accounts
    except Exception as e:
        logger.error(f"Error fetching accounts: {str(e)}")
        return accounts

async def get_user_sync_status(db: Session, user_id: int) -> Dict[str, Any]:
    """Obtenir l'état de synchronisation pour un utilisateur."""
    # Récupérer tous les items de l'utilisateur
    sync_items = db.query(SyncItem).filter(SyncItem.user_id == user_id).all()
    
    # Récupérer tous les comptes
    account_ids = []
    for item in sync_items:
        account_ids.extend([account.id for account in item.accounts])
    
    sync_accounts = db.query(SyncAccount).filter(SyncAccount.id.in_(account_ids)).all() if account_ids else []
    
    # Déterminer l'état global
    needs_action = any(item.needs_user_action for item in sync_items)
    last_sync = max([item.last_successful_refresh for item in sync_items if item.last_successful_refresh], default=None)
    
    return {
        "total_items": len(sync_items),
        "total_accounts": len(sync_accounts),
        "needs_user_action": needs_action,
        "items_needing_action": [
            {
                "id": item.id,
                "bridge_item_id": item.bridge_item_id,
                "status": item.status,
                "status_code_info": item.status_code_info,
                "status_description": item.status_description
            }
            for item in sync_items if item.needs_user_action
        ],
        "last_successful_sync": last_sync,
    }

async def create_reconnect_session(db: Session, user_id: int, bridge_item_id: int) -> str:
    """Créer une session de reconnexion pour un item."""
    # Vérifier que l'item appartient bien à l'utilisateur
    sync_item = db.query(SyncItem).filter(
        SyncItem.user_id == user_id,
        SyncItem.bridge_item_id == bridge_item_id
    ).first()
    
    if not sync_item:
        raise ValueError(f"Item {bridge_item_id} not found for user {user_id}")
    
    # Créer une session Connect avec l'item_id
    return await bridge_service.create_connect_session(
        db,
        user_id,
        item_id=bridge_item_id,
        callback_url="https://app.harena.io/reconnect-callback"
    )