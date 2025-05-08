"""
Gestionnaire des transactions bancaires.

Ce module gère la récupération et le stockage des transactions 
depuis Bridge API vers la base SQL.
"""

import logging
from datetime import datetime, timezone, timedelta
from sqlalchemy.orm import Session
from typing import Dict, List, Any, Optional, Tuple

from sync_service.models.sync import SyncItem, SyncAccount, RawTransaction
from sync_service.utils.logging import get_contextual_logger

logger = logging.getLogger(__name__)

async def sync_account_transactions(
    db: Session,
    sync_account: SyncAccount
) -> Dict[str, Any]:
    """
    Synchronise les transactions d'un compte.
    
    Args:
        db: Session de base de données
        sync_account: Compte de synchronisation
        
    Returns:
        Dict: Résultat de la synchronisation
    """
    account_id = sync_account.bridge_account_id
    item = sync_account.item
    user_id = item.user_id
    bridge_item_id = item.bridge_item_id
    
    ctx_logger = get_contextual_logger(
        "sync_service.transaction_handler", 
        user_id=user_id, 
        bridge_item_id=bridge_item_id,
        account_id=account_id
    )
    
    ctx_logger.info(f"Synchronisation des transactions pour le compte {account_id}")
    
    result = {
        "status": "pending",
        "account_id": account_id,
        "transactions_count": 0,
        "transactions_stored": 0,
        "transactions": []  # Pour retourner les transactions pour l'enrichissement
    }
    
    try:
        # Récupérer le token Bridge
        from user_service.services.bridge import get_bridge_token
        token_data = await get_bridge_token(db, user_id)
        access_token = token_data["access_token"]
        
        # Déterminer la date de début pour la synchronisation
        since_date = None
        
        # Si le compte a déjà été synchronisé, partir de sa dernière date de synchronisation
        if sync_account.last_sync_timestamp:
            # Ajouter une marge de 7 jours pour récupérer les transactions potentiellement modifiées
            since_date = sync_account.last_sync_timestamp - timedelta(days=7)
            since_date_str = since_date.isoformat()
            ctx_logger.info(f"Récupération des transactions depuis {since_date_str}")
        else:
            ctx_logger.info(f"Première synchronisation pour ce compte, récupération de l'historique complet")
        
        # Récupérer les transactions depuis Bridge API
        from user_service.services.bridge import get_bridge_transactions
        transactions = await get_bridge_transactions(
            db, 
            user_id, 
            access_token, 
            account_id=account_id,
            since=since_date_str if since_date else None
        )
        
        if not transactions:
            ctx_logger.info(f"Aucune transaction récupérée pour le compte {account_id}")
            result["status"] = "success"  # C'est un succès, même s'il n'y a pas de transactions
            
            # Mettre à jour le timestamp de synchronisation
            sync_account.last_sync_timestamp = datetime.now(timezone.utc)
            db.add(sync_account)
            db.commit()
            
            return result
            
        ctx_logger.info(f"Récupération de {len(transactions)} transactions depuis Bridge API")
        
        # Stocker les transactions dans la base SQL
        stored_count = await store_transactions_sql(db, user_id, sync_account.id, transactions)
        
        result["transactions_count"] = len(transactions)
        result["transactions_stored"] = stored_count
        result["transactions"] = transactions  # Pour l'enrichissement
        result["status"] = "success" if stored_count == len(transactions) else "partial"
        
        # Mettre à jour le timestamp de synchronisation
        sync_account.last_sync_timestamp = datetime.now(timezone.utc)
        db.add(sync_account)
        db.commit()
        
        ctx_logger.info(f"Synchronisation terminée: {stored_count}/{len(transactions)} transactions stockées")
        return result
    except Exception as e:
        ctx_logger.error(f"Erreur lors de la synchronisation des transactions: {e}", exc_info=True)
        result["status"] = "error"
        result["message"] = str(e)
        return result

async def store_transactions_sql(
    db: Session,
    user_id: int,
    account_id: int,
    transactions: List[Dict[str, Any]]
) -> int:
    """
    Stocke les transactions dans la base SQL.
    
    Args:
        db: Session de base de données
        user_id: ID de l'utilisateur
        account_id: ID du compte
        transactions: Liste des transactions à stocker
        
    Returns:
        int: Nombre de transactions stockées
    """
    ctx_logger = get_contextual_logger("sync_service.transaction_handler", user_id=user_id, account_id=account_id)
    
    # Obtenir les IDs des transactions existantes pour éviter les doublons
    existing_ids = {tx.bridge_transaction_id for tx in 
                   db.query(RawTransaction.bridge_transaction_id).filter(
                       RawTransaction.account_id == account_id,
                       RawTransaction.user_id == user_id
                   ).all()}
    
    stored_count = 0
    
    for tx_data in transactions:
        bridge_tx_id = tx_data.get("id")
        if not bridge_tx_id:
            ctx_logger.warning(f"Transaction sans ID, ignorée")
            continue
            
        try:
            # Vérifier si la transaction existe déjà
            if bridge_tx_id in existing_ids:
                # Mise à jour (seulement si les transactions sont marquées comme 'deleted')
                if tx_data.get("deleted", False):
                    tx = db.query(RawTransaction).filter(
                        RawTransaction.bridge_transaction_id == bridge_tx_id
                    ).first()
                    
                    if tx:
                        tx.deleted = True
                        db.add(tx)
                        stored_count += 1
            else:
                # Création d'une nouvelle transaction
                new_tx = RawTransaction(
                    bridge_transaction_id=bridge_tx_id,
                    account_id=account_id,
                    user_id=user_id,
                    clean_description=tx_data.get("clean_description"),
                    provider_description=tx_data.get("provider_description"),
                    amount=tx_data.get("amount", 0),
                    date=parse_date(tx_data.get("date")),
                    booking_date=parse_date(tx_data.get("booking_date")),
                    transaction_date=parse_date(tx_data.get("transaction_date")),
                    value_date=parse_date(tx_data.get("value_date")),
                    currency_code=tx_data.get("currency_code"),
                    category_id=tx_data.get("category_id"),
                    operation_type=tx_data.get("operation_type"),
                    deleted=tx_data.get("deleted", False),
                    future=tx_data.get("future", False),
                    updated_at_bridge=parse_date(tx_data.get("updated_at"))
                )
                
                db.add(new_tx)
                stored_count += 1
                existing_ids.add(bridge_tx_id)  # Ajouter à la liste des IDs existants
            
            # Commit périodique pour éviter les transactions trop longues
            if stored_count % 100 == 0:
                db.commit()
                
        except Exception as e:
            ctx_logger.error(f"Erreur lors du stockage de la transaction {bridge_tx_id}: {e}", exc_info=True)
            # Continuer avec la transaction suivante
    
    # Commit final
    try:
        db.commit()
        ctx_logger.info(f"{stored_count} transactions stockées avec succès")
    except Exception as e:
        db.rollback()
        ctx_logger.error(f"Erreur lors du commit final des transactions: {e}", exc_info=True)
        raise
        
    return stored_count

def parse_date(date_val: Any) -> Optional[datetime]:
    """
    Parse une date depuis diverses sources.
    
    Args:
        date_val: Valeur de date (chaîne, datetime, etc.)
        
    Returns:
        datetime: Date parsée ou None si erreur
    """
    if not date_val:
        return None
        
    if isinstance(date_val, datetime):
        return date_val
        
    if isinstance(date_val, str):
        try:
            return datetime.fromisoformat(date_val.replace('Z', '+00:00'))
        except (ValueError, TypeError):
            logger.warning(f"Format de date invalide: {date_val}")
            return None
            
    return None

async def force_sync_all_accounts(db: Session, user_id: int) -> Dict[str, Any]:
    """
    Force la synchronisation de tous les comptes d'un utilisateur.
    
    Args:
        db: Session de base de données
        user_id: ID de l'utilisateur
        
    Returns:
        Dict: Résultat de la synchronisation
    """
    ctx_logger = get_contextual_logger("sync_service.transaction_handler", user_id=user_id)
    ctx_logger.info(f"Force synchronisation de tous les comptes pour l'utilisateur {user_id}")
    
    result = {
        "status": "pending",
        "total_accounts": 0,
        "synced_accounts": 0,
        "failed_accounts": 0,
        "total_transactions": 0,
        "transactions": []  # Pour collecter toutes les transactions pour l'enrichissement
    }
    
    try:
        # Récupérer tous les items actifs de l'utilisateur
        items = db.query(SyncItem).filter(
            SyncItem.user_id == user_id,
            SyncItem.status == 0  # Seulement les items sans erreur
        ).all()
        
        if not items:
            ctx_logger.warning(f"Aucun item actif trouvé pour l'utilisateur {user_id}")
            result["status"] = "warning"
            result["message"] = "No active items found"
            return result
            
        ctx_logger.info(f"Récupération de {len(items)} items actifs")
        
        # Récupérer tous les comptes associés aux items
        all_accounts = []
        for item in items:
            accounts = db.query(SyncAccount).filter(SyncAccount.item_id == item.id).all()
            all_accounts.extend(accounts)
            
        result["total_accounts"] = len(all_accounts)
        
        if not all_accounts:
            ctx_logger.warning(f"Aucun compte trouvé pour l'utilisateur {user_id}")
            result["status"] = "warning"
            result["message"] = "No accounts found"
            return result
            
        ctx_logger.info(f"Synchronisation de {len(all_accounts)} comptes")
        
        # Synchroniser chaque compte
        for account in all_accounts:
            try:
                sync_result = await sync_account_transactions(db, account)
                
                if sync_result.get("status") in ["success", "partial"]:
                    result["synced_accounts"] += 1
                    result["total_transactions"] += sync_result.get("transactions_count", 0)
                    
                    # Collecter les transactions pour l'enrichissement
                    if "transactions" in sync_result and sync_result["transactions"]:
                        result["transactions"].extend(sync_result["transactions"])
                else:
                    result["failed_accounts"] += 1
                    
                ctx_logger.info(f"Synchronisation du compte {account.bridge_account_id}: {sync_result.get('status')}")
            except Exception as e:
                ctx_logger.error(f"Erreur lors de la synchronisation du compte {account.bridge_account_id}: {e}", exc_info=True)
                result["failed_accounts"] += 1
        
        # Déterminer le statut global
        if result["failed_accounts"] == 0 and result["synced_accounts"] > 0:
            result["status"] = "success"
        elif result["synced_accounts"] > 0:
            result["status"] = "partial"
        else:
            result["status"] = "error"
            result["message"] = "All account synchronizations failed"
            
        ctx_logger.info(f"Synchronisation terminée: {result['synced_accounts']}/{result['total_accounts']} comptes synchronisés, {result['total_transactions']} transactions récupérées")
        return result
    except Exception as e:
        ctx_logger.error(f"Erreur générale lors de la synchronisation des comptes: {e}", exc_info=True)
        result["status"] = "error"
        result["message"] = str(e)
        return result