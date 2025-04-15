# sync_service/services/transaction_sync.py
import logging
import httpx
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from typing import Dict, Any, List, Optional

from sync_service.models.sync import SyncAccount, SyncItem
from user_service.services import bridge as bridge_service
from user_service.core.config import settings

logger = logging.getLogger(__name__)

async def sync_account_transactions(db: Session, sync_account: SyncAccount) -> Dict[str, Any]:
    """Synchroniser les transactions d'un compte depuis la dernière mise à jour."""
    try:
        # Récupérer l'utilisateur via l'item de manière robuste
        item = db.query(SyncItem).filter(SyncItem.id == sync_account.item_id).first()
        
        if not item:
            logger.error(f"Item not found for account {sync_account.id}")
            return {"errors": "Item not found for account"}
        
        user_id = item.user_id
        
        # Récupérer le token Bridge
        token_data = await bridge_service.get_bridge_token(db, user_id)
        access_token = token_data["access_token"]
        
        # Construire la requête avec le paramètre since
        url = f"{settings.BRIDGE_API_URL}/aggregation/transactions?account_id={sync_account.bridge_account_id}"
        
        # Ajouter le paramètre since si on a une date de dernière synchronisation
        if sync_account.last_sync_timestamp:
            since_param = sync_account.last_sync_timestamp.isoformat()
            url += f"&since={since_param}"
        
        headers = {
            "accept": "application/json",
            "Bridge-Version": settings.BRIDGE_API_VERSION,
            "authorization": f"Bearer {access_token}",
            "Client-Id": settings.BRIDGE_CLIENT_ID,    
            "Client-Secret": settings.BRIDGE_CLIENT_SECRET,
        }
        
        result = {
            "new_transactions": 0,
            "updated_transactions": 0,
            "deleted_transactions": 0,
            "errors": None
        }
        
        # Récupérer les transactions
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers)
            
            if response.status_code != 200:
                logger.error(f"Failed to fetch transactions: {response.text}")
                result["errors"] = f"API error: {response.status_code}"
                return result
            
            transactions_data = response.json()
            
            if "resources" not in transactions_data:
                logger.error(f"Invalid transactions response: {transactions_data}")
                result["errors"] = "Invalid API response format"
                return result
            
            # Traiter les transactions
            transactions = transactions_data["resources"]
            
            for transaction in transactions:
                # À implémenter: logique de mise à jour des transactions dans la base
                if transaction.get("deleted", False):
                    result["deleted_transactions"] += 1
                else:
                    # Déterminer s'il s'agit d'une nouvelle transaction ou d'une mise à jour
                    # selon votre logique de gestion des transactions
                    result["new_transactions"] += 1
            
            # Mettre à jour le timestamp de dernière synchronisation
            if transactions:
                last_updated_dates = [
                    datetime.fromisoformat(t["updated_at"].replace('Z', '+00:00')) 
                    for t in transactions if "updated_at" in t
                ]
                
                if last_updated_dates:
                    last_updated = max(last_updated_dates)
                    sync_account.last_transaction_date = last_updated
            
            sync_account.last_sync_timestamp = datetime.now(timezone.utc)
            db.add(sync_account)
            db.commit()
            db.refresh(sync_account)
            
            return result
    except Exception as e:
        logger.error(f"Error syncing transactions: {str(e)}")
        return {
            "new_transactions": 0,
            "updated_transactions": 0,
            "deleted_transactions": 0,
            "errors": str(e)
        }