"""
Utilitaires pour la gestion de la synchronisation.

Ce module fournit des fonctions utilitaires pour obtenir et formatter
l'état de synchronisation, calculer des métriques, etc.
"""

import logging
from datetime import datetime, timezone, timedelta
from sqlalchemy.orm import Session
from typing import Dict, Any, List, Optional

from db_service.models.user import User
from db_service.models.sync import SyncItem, SyncAccount
from sync_service.utils.logging import get_contextual_logger

logger = logging.getLogger(__name__)

async def get_global_sync_status(db: Session, user_id: int) -> Dict[str, Any]:
    """
    Récupère l'état de synchronisation global pour un utilisateur
    
    Args:
        db: Session de base de données
        user_id: ID de l'utilisateur
        
    Returns:
        Dict: État détaillé de la synchronisation
    """
    ctx_logger = get_contextual_logger("sync_service.sync_manager", user_id=user_id)
    ctx_logger.info(f"Récupération de l'état de synchronisation global pour l'utilisateur {user_id}")

    try:
        # Statut SQL
        sync_items = db.query(SyncItem).filter(SyncItem.user_id == user_id).all()
        # Récupérer les comptes via la relation pour éviter une requête séparée
        all_sql_accounts_count = sum(len(item.accounts) for item in sync_items)

        items_info = []
        needs_action = False
        last_sync_sql = None
        for item in sync_items:
            if item.needs_user_action:
                needs_action = True
            if item.last_successful_refresh and (last_sync_sql is None or item.last_successful_refresh > last_sync_sql):
                last_sync_sql = item.last_successful_refresh

            items_info.append({
                "bridge_item_id": item.bridge_item_id,
                "status": item.status,
                "status_code_info": item.status_code_info,
                "status_description": item.status_description,
                "needs_user_action": item.needs_user_action,
                "last_successful_refresh": item.last_successful_refresh.isoformat() if item.last_successful_refresh else None,
                "last_try_refresh": item.last_try_refresh.isoformat() if item.last_try_refresh else None,
                "sql_account_count": len(item.accounts)
            })

        items_needing_action = [item for item in items_info if item["needs_user_action"]]

        # Calculer les jours depuis la dernière synchronisation
        days_since_last_sync = None
        if last_sync_sql:
            # S'assurer que last_sync_sql est offset-aware pour la comparaison
            if last_sync_sql.tzinfo is None:
                 # Si non-aware, supposer UTC (ou le timezone approprié)
                 last_sync_sql = last_sync_sql.replace(tzinfo=timezone.utc)
            days_since_last_sync = (datetime.now(timezone.utc) - last_sync_sql).days

        status_response = {
            "user_id": user_id,
            "sql_status": {
                "total_items": len(items_info),
                "total_accounts": all_sql_accounts_count,
                "needs_user_action": needs_action,
                "last_successful_sync": last_sync_sql.isoformat() if last_sync_sql else None,
                "days_since_last_sync": days_since_last_sync,
                "items_needing_action_count": len(items_needing_action),
                "items_needing_action_ids": [item['bridge_item_id'] for item in items_needing_action],
                "items": items_info,
            }
        }
        
        # Ajouter des statistiques sur les transactions
        transactions_stats = get_transaction_stats(db, user_id)
        status_response["transactions_stats"] = transactions_stats
        
        ctx_logger.info(f"État de synchronisation récupéré avec succès pour l'utilisateur {user_id}")
        return status_response

    except Exception as e:
        ctx_logger.error(f"Erreur lors de la récupération de l'état de synchronisation global: {e}", exc_info=True)
        # Retourner une structure d'erreur cohérente
        return {
            "user_id": user_id,
            "error": f"Failed to get global sync status: {str(e)}",
            "sql_status": {"status": "error", "message": "Could not retrieve SQL status"}
        }

def get_transaction_stats(db: Session, user_id: int) -> Dict[str, Any]:
    """
    Calcule des statistiques sur les transactions de l'utilisateur.
    
    Args:
        db: Session de base de données
        user_id: ID de l'utilisateur
        
    Returns:
        Dict: Statistiques sur les transactions
    """
    from db_service.models.sync import RawTransaction
    from sqlalchemy import func, and_
    
    stats = {
        "total_transactions": 0,
        "last_30_days": 0,
        "by_account": [],
        "by_operation_type": []
    }
    
    try:
        # Total des transactions
        total = db.query(func.count(RawTransaction.id)).filter(
            RawTransaction.user_id == user_id,
            RawTransaction.deleted.is_(False)
        ).scalar()
        stats["total_transactions"] = total or 0
        
        # Transactions des 30 derniers jours
        thirty_days_ago = datetime.now(timezone.utc) - timedelta(days=30)
        last_30 = db.query(func.count(RawTransaction.id)).filter(
            RawTransaction.user_id == user_id,
            RawTransaction.deleted.is_(False),
            RawTransaction.date >= thirty_days_ago
        ).scalar()
        stats["last_30_days"] = last_30 or 0
        
        # Répartition par compte
        account_stats = db.query(
            RawTransaction.account_id,
            func.count(RawTransaction.id)
        ).filter(
            RawTransaction.user_id == user_id,
            RawTransaction.deleted.is_(False)
        ).group_by(RawTransaction.account_id).all()
        
        for account_id, count in account_stats:
            account = db.query(SyncAccount).filter(SyncAccount.id == account_id).first()
            if account:
                stats["by_account"].append({
                    "account_id": account.bridge_account_id,
                    "account_name": account.account_name,
                    "count": count
                })
        
        # Répartition par type d'opération
        operation_stats = db.query(
            RawTransaction.operation_type,
            func.count(RawTransaction.id)
        ).filter(
            RawTransaction.user_id == user_id,
            RawTransaction.deleted.is_(False)
        ).group_by(RawTransaction.operation_type).all()
        
        for operation_type, count in operation_stats:
            stats["by_operation_type"].append({
                "operation_type": operation_type or "unknown",
                "count": count
            })
    
    except Exception as e:
        logger.error(f"Erreur lors du calcul des statistiques de transactions: {e}", exc_info=True)
    
    return stats

def format_sync_status(sync_status: Dict[str, Any]) -> Dict[str, Any]:
    """
    Formatte le statut de synchronisation pour l'affichage.
    
    Args:
        sync_status: État brut de synchronisation
        
    Returns:
        Dict: État formatté pour l'affichage
    """
    # Déterminer le statut global
    user_id = sync_status.get("user_id")
    has_error = "error" in sync_status
    needs_action = sync_status.get("sql_status", {}).get("needs_user_action", False)
    
    overall_status = "ok"
    if has_error:
        overall_status = "error"
    elif needs_action:
        overall_status = "needs_action"
    
    # Formatter le message principal
    message = ""
    if has_error:
        message = sync_status.get("error", "An error occurred during sync status retrieval")
    elif needs_action:
        action_count = sync_status.get("sql_status", {}).get("items_needing_action_count", 0)
        message = f"Action required for {action_count} connection{'s' if action_count != 1 else ''}"
    else:
        days_since = sync_status.get("sql_status", {}).get("days_since_last_sync")
        if days_since is not None:
            if days_since == 0:
                message = "Data synced today"
            elif days_since == 1:
                message = "Data synced yesterday"
            else:
                message = f"Data synced {days_since} days ago"
        else:
            message = "No sync data available"
    
    # Créer le résumé formatté
    formatted_status = {
        "user_id": user_id,
        "overall_status": overall_status,
        "message": message,
        "detailed_status": sync_status
    }
    
    return formatted_status