import logging
import httpx
import json
from datetime import datetime, timezone, timedelta
from sqlalchemy.orm import Session
from typing import Dict, Any, List, Optional, Union
import traceback

from sync_service.models.sync import SyncAccount, SyncItem
from user_service.models.user import User, BridgeConnection  
from user_service.services import bridge as bridge_service
from user_service.core.config import settings
from sync_service.services.vector_storage import VectorStorageService

# Configuration du logger
logger = logging.getLogger(__name__)

async def sync_account_transactions(db: Session, sync_account: SyncAccount) -> Dict[str, Any]:
    """Synchroniser les transactions d'un compte depuis la dernière mise à jour."""
    logger.info(f"Début de synchronisation des transactions pour le compte {sync_account.bridge_account_id}")
    
    try:
        # Récupérer l'utilisateur via l'item de manière robuste
        item = db.query(SyncItem).filter(SyncItem.id == sync_account.item_id).first()
        
        if not item:
            logger.error(f"Item not found for account {sync_account.id}")
            return {"status": "error", "errors": "Item not found for account"}
        
        user_id = item.user_id
        logger.info(f"Item trouvé: bridge_item_id={item.bridge_item_id}, user_id={user_id}")
        
        # Récupérer l'utilisateur pour Bridge API
        user_bridge_connection = db.query(BridgeConnection).filter(
            BridgeConnection.user_id == user_id
        ).first()
        
        if not user_bridge_connection:
            logger.error(f"No bridge connection found for user_id {user_id}")
            return {"status": "error", "errors": "Bridge connection not found"}
        
        logger.info(f"Bridge connection trouvée: bridge_user_uuid={user_bridge_connection.bridge_user_uuid}")
        
        # Récupérer le token Bridge
        try:
            token_data = await bridge_service.get_bridge_token(db, user_id)
            access_token = token_data["access_token"]
            logger.info(f"Token récupéré pour user_id={user_id}")
        except Exception as token_error:
            logger.error(f"Failed to get Bridge token: {str(token_error)}")
            return {"status": "error", "errors": f"Token error: {str(token_error)}"}
        
        # Construire la requête avec le paramètre since
        url = f"{settings.BRIDGE_API_URL}/aggregation/transactions?account_id={sync_account.bridge_account_id}"
        
        # Déterminer la date de début pour la synchronisation
        since_date = None
        if sync_account.last_sync_timestamp:
            # Synchroniser depuis la dernière synchro moins un jour pour prendre en compte les éventuelles modifications
            since_date = sync_account.last_sync_timestamp - timedelta(days=1)
            logger.info(f"Dernière synchronisation le {sync_account.last_sync_timestamp}, synchronisation depuis {since_date}")
        else:
            # Pour une première synchronisation, récupérer les 90 derniers jours
            since_date = datetime.now(timezone.utc) - timedelta(days=90)
            logger.info(f"Première synchronisation, récupération depuis {since_date}")
        
        # Ajouter le paramètre since
        since_param = since_date.strftime("%Y-%m-%d")
        url += f"&since={since_param}"
        
        headers = {
            "accept": "application/json",
            "Bridge-Version": settings.BRIDGE_API_VERSION,
            "authorization": f"Bearer {access_token}",
            "Client-Id": settings.BRIDGE_CLIENT_ID,    
            "Client-Secret": settings.BRIDGE_CLIENT_SECRET,
        }
        
        logger.info(f"Appel API Bridge pour les transactions: URL={url}, depuis={since_param}")
        
        result = {
            "status": "success",
            "new_transactions": 0,
            "updated_transactions": 0,
            "deleted_transactions": 0,
            "processed_transactions": 0,
            "errors": None
        }
        
        # Récupérer les transactions
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers)
            
            if response.status_code != 200:
                logger.error(f"Failed to fetch transactions: {response.text}")
                result["status"] = "error"
                result["errors"] = f"API error: {response.status_code}"
                return result
            
            response_data = response.json()
            logger.debug(f"Réponse API Bridge: {json.dumps(response_data)[:500]}...")
            
            if "resources" not in response_data:
                logger.error(f"Invalid transactions response structure: {response_data}")
                result["status"] = "error"
                result["errors"] = "Invalid API response format"
                return result
            
            # Traiter les transactions
            transactions = response_data["resources"]
            total_transactions = len(transactions)
            logger.info(f"Nombre de transactions récupérées: {total_transactions}")
            
            # Si des transactions ont été trouvées, les stocker dans la base vectorielle
            if total_transactions > 0:
                # Initialiser le service de stockage vectoriel
                vector_storage = VectorStorageService()
                
                # Préparer les transactions pour le stockage vectoriel
                vector_transactions = []
                for tx in transactions:
                    # Extraire les dates de la transaction
                    tx_date = None
                    if "date" in tx:
                        try:
                            if isinstance(tx["date"], str):
                                tx_date = tx["date"]
                            else:
                                tx_date = tx["date"].isoformat()
                        except (ValueError, AttributeError):
                            tx_date = datetime.now().date().isoformat()
                    else:
                        tx_date = datetime.now().date().isoformat()
                    
                    # Préparer la transaction pour la vectorisation
                    vector_tx = {
                        "user_id": user_id,
                        "account_id": sync_account.bridge_account_id,
                        "bridge_transaction_id": tx.get("id"),
                        "amount": tx.get("amount", 0.0),
                        "currency_code": tx.get("currency_code", "EUR"),
                        "description": tx.get("description", ""),
                        "clean_description": tx.get("clean_description", ""),
                        "transaction_date": tx_date,
                        "category_id": tx.get("category_id"),
                        "operation_type": tx.get("operation_type"),
                        "is_recurring": False  # Valeur par défaut
                    }
                    
                    vector_transactions.append(vector_tx)
                
                # Stocker les transactions dans la base vectorielle
                logger.info(f"Stockage de {len(vector_transactions)} transactions dans la base vectorielle")
                vector_result = await vector_storage.batch_store_transactions(vector_transactions)
                
                # Mettre à jour le résultat avec les informations du stockage vectoriel
                result["new_transactions"] = vector_result.get("successful", 0)
                result["processed_transactions"] = vector_result.get("total", 0)
                
                if vector_result.get("status") != "success":
                    result["status"] = "partial"
                    result["errors"] = f"Erreurs pendant le stockage vectoriel: {vector_result.get('failed', 0)} échecs"
                
                logger.info(f"Résultat du stockage vectoriel: {vector_result}")
            
            # Mettre à jour le timestamp de dernière synchronisation, même en cas d'erreur partielle
            current_time = datetime.now(timezone.utc)
            sync_account.last_sync_timestamp = current_time
            
            # Mettre à jour la date de dernière transaction si disponible
            if transactions:
                for transaction in transactions:
                    if "updated_at" in transaction:
                        try:
                            updated_at = datetime.fromisoformat(transaction["updated_at"].replace('Z', '+00:00'))
                            if not sync_account.last_transaction_date or updated_at > sync_account.last_transaction_date:
                                sync_account.last_transaction_date = updated_at
                        except (ValueError, TypeError) as e:
                            logger.warning(f"Error parsing transaction date: {e}")
            
            db.add(sync_account)
            db.commit()
            db.refresh(sync_account)
            
            logger.info(f"Synchronisation terminée pour le compte {sync_account.bridge_account_id}: {result}")
            return result
            
    except Exception as e:
        error_details = traceback.format_exc()
        logger.error(f"Error syncing transactions: {str(e)}\n{error_details}")
        return {
            "status": "error",
            "new_transactions": 0,
            "updated_transactions": 0,
            "deleted_transactions": 0,
            "processed_transactions": 0,
            "errors": str(e)
        }

async def force_sync_all_accounts(db: Session, user_id: int) -> Dict[str, Any]:
    """Force la synchronisation de tous les comptes d'un utilisateur."""
    logger.info(f"Forçage de la synchronisation de tous les comptes pour l'utilisateur {user_id}")
    
    try:
        # Récupérer tous les items de l'utilisateur
        items = db.query(SyncItem).filter(SyncItem.user_id == user_id).all()
        
        if not items:
            logger.warning(f"Aucun item trouvé pour l'utilisateur {user_id}")
            return {"status": "warning", "message": "No items found for this user"}
        
        logger.info(f"{len(items)} items trouvés pour l'utilisateur {user_id}")
        
        results = []
        
        # Pour chaque item, synchroniser tous ses comptes
        for item in items:
            # Récupérer les comptes de cet item
            accounts = db.query(SyncAccount).filter(SyncAccount.item_id == item.id).all()
            
            if not accounts:
                logger.warning(f"Aucun compte trouvé pour l'item {item.id}")
                continue
            
            logger.info(f"{len(accounts)} comptes trouvés pour l'item {item.id}")
            
            # Synchroniser chaque compte
            for account in accounts:
                try:
                    logger.info(f"Synchronisation du compte {account.bridge_account_id}")
                    result = await sync_account_transactions(db, account)
                    results.append({
                        "account_id": account.bridge_account_id,
                        "result": result
                    })
                except Exception as e:
                    logger.error(f"Error syncing account {account.bridge_account_id}: {str(e)}")
                    results.append({
                        "account_id": account.bridge_account_id,
                        "result": {"status": "error", "errors": str(e)}
                    })
        
        # Déterminer le statut global
        overall_status = "success"
        for result in results:
            if result["result"].get("status") == "error":
                overall_status = "partial"
        
        if not results:
            overall_status = "warning"
        
        return {
            "status": overall_status,
            "accounts_processed": len(results),
            "details": results
        }
        
    except Exception as e:
        logger.error(f"Error during force sync for user {user_id}: {str(e)}")
        return {"status": "error", "message": str(e)}

async def check_and_sync_missing_transactions(db: Session, user_id: int) -> Dict[str, Any]:
    """Vérifie et synchronise les transactions manquantes pour un utilisateur."""
    logger.info(f"Vérification des transactions manquantes pour l'utilisateur {user_id}")
    
    try:
        # Récupérer l'état de synchronisation
        vector_storage = VectorStorageService()
        sync_status = await vector_storage.get_user_sync_status(user_id)
        logger.info(f"État actuel de la synchronisation vectorielle: {sync_status}")
        
        # Récupérer la connexion Bridge de l'utilisateur
        bridge_connection = db.query(BridgeConnection).filter(
            BridgeConnection.user_id == user_id
        ).first()
        
        if not bridge_connection:
            logger.error(f"No bridge connection found for user_id {user_id}")
            return {"status": "error", "message": "Bridge connection not found"}
        
        # Récupérer tous les comptes de l'utilisateur
        items = db.query(SyncItem).filter(SyncItem.user_id == user_id).all()
        accounts = []
        
        for item in items:
            item_accounts = db.query(SyncAccount).filter(SyncAccount.item_id == item.id).all()
            accounts.extend(item_accounts)
        
        logger.info(f"Nombre total de comptes: {len(accounts)}")
        
        # Si aucun compte, ou si la synchronisation est incomplète, forcer une synchronisation complète
        if len(accounts) == 0:
            logger.warning(f"Pas de comptes trouvés, pas d'action")
            return {"status": "warning", "message": "No accounts found"}
        
        # Lancer une synchronisation complète
        result = await force_sync_all_accounts(db, user_id)
        logger.info(f"Résultat de la synchronisation forcée: {result}")
        
        return result
        
    except Exception as e:
        error_details = traceback.format_exc()
        logger.error(f"Error checking for missing transactions: {str(e)}\n{error_details}")
        return {"status": "error", "message": str(e)}

async def get_user_vector_stats(db: Session, user_id: int) -> Dict[str, Any]:
    """Récupère les statistiques vectorielles pour un utilisateur."""
    logger.info(f"Récupération des statistiques vectorielles pour l'utilisateur {user_id}")
    
    try:
        # Initialiser le service de stockage vectoriel
        vector_storage = VectorStorageService()
        
        # Récupérer les statistiques
        stats = await vector_storage.get_user_statistics(user_id)
        logger.info(f"Statistiques récupérées: {stats}")
        
        return stats
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des statistiques vectorielles: {str(e)}")
        return {
            "status": "error",
            "message": str(e),
            "user_id": user_id
        }