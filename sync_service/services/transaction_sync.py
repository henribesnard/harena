# sync_service/services/transaction_sync.py
import logging
import httpx
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from typing import Dict, Any, List, Optional

from sync_service.models.sync import SyncAccount
from user_service.services import bridge as bridge_service
from user_service.core.config import settings

logger = logging.getLogger(__name__)

async def sync_account_transactions(db: Session, sync_account: SyncAccount) -> Dict[str, Any]:
    """Synchroniser les transactions d'un compte depuis la dernière mise à jour."""
    # Récupérer l'utilisateur via l'item
    user_id = db.query(SyncAccount.item).filter(
        SyncAccount.id == sync_account.id
    ).join(SyncAccount.item).scalar().user_id
    
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
        "authorization": f"Bearer {access_token}"
    }
    
    result = {
        "new_transactions": 0,
        "updated_transactions": 0,
        "deleted_transactions": 0,
        "errors": None
    }
    
    try:
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
            # Ici, on simplifie en comptant simplement les transactions
            # Dans une implémentation réelle, il faudrait les stocker dans la base
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
            last_updated = max(
                [datetime.fromisoformat(t["updated_at"].replace('Z', '+00:00')) 
                 for t in transactions if "updated_at" in t],
                default=None
            )
            
            if last_updated:
                sync_account.last_transaction_date = last_updated
            
            sync_account.last_sync_timestamp = datetime.now(timezone.utc)
            db.add(sync_account)
            db.commit()
            db.refresh(sync_account)
            
            return result
    except Exception as e:
        logger.error(f"Error syncing transactions: {str(e)}")
        result["errors"] = str(e)
        return result