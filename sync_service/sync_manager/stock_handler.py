"""
Gestionnaire des stocks et titres financiers.

Ce module gère la synchronisation et le stockage des stocks
et titres financiers depuis Bridge API vers la base SQL.
"""

import logging
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from typing import Dict, List, Any, Optional

from sync_service.models.sync import SyncAccount, RawStock
from sync_service.utils.logging import get_contextual_logger

logger = logging.getLogger(__name__)

async def sync_all_stocks(db: Session, user_id: int) -> Dict[str, Any]:
    """
    Synchronise tous les stocks/titres pour un utilisateur.
    
    Args:
        db: Session de base de données
        user_id: ID de l'utilisateur
        
    Returns:
        Dict: Résultat de la synchronisation
    """
    ctx_logger = get_contextual_logger("sync_service.stock_handler", user_id=user_id)
    ctx_logger.info(f"Synchronisation des stocks pour l'utilisateur {user_id}")
    
    result = {
        "status": "pending",
        "stocks_count": 0,
        "stocks_stored": 0,
        "accounts_processed": 0
    }
    
    try:
        # Récupérer le token Bridge
        from user_service.services.bridge import get_bridge_token
        token_data = await get_bridge_token(db, user_id)
        access_token = token_data["access_token"]
        
        # Récupérer les comptes de l'utilisateur
        accounts = db.query(SyncAccount).join(
            SyncAccount.item
        ).filter(
            SyncAccount.item.has(user_id=user_id)
        ).all()
        
        if not accounts:
            ctx_logger.warning(f"Aucun compte trouvé pour l'utilisateur {user_id}")
            result["status"] = "warning"
            result["message"] = "No accounts found"
            return result
        
        # Récupérer et stocker les stocks pour chaque compte
        for account in accounts:
            try:
                # Récupérer les stocks depuis Bridge API
                from user_service.services.bridge import get_bridge_stocks
                stocks = await get_bridge_stocks(
                    db, 
                    user_id, 
                    access_token, 
                    account_id=account.bridge_account_id
                )
                
                if stocks:
                    # Stocker les stocks
                    stored_count = await store_stocks_sql(db, user_id, account.id, stocks)
                    result["stocks_count"] += len(stocks)
                    result["stocks_stored"] += stored_count
                
                result["accounts_processed"] += 1
                ctx_logger.info(f"Compte {account.bridge_account_id}: {len(stocks) if stocks else 0} stocks récupérés")
                
            except Exception as e:
                ctx_logger.error(f"Erreur lors de la synchronisation des stocks du compte {account.bridge_account_id}: {e}", exc_info=True)
        
        # Déterminer le statut final
        if result["stocks_count"] > 0:
            if result["stocks_stored"] == result["stocks_count"]:
                result["status"] = "success"
            else:
                result["status"] = "partial"
        else:
            result["status"] = "success"  # Même sans stocks, c'est un succès
            result["message"] = "No stocks found"
        
        ctx_logger.info(f"Synchronisation des stocks terminée: {result['stocks_stored']}/{result['stocks_count']} stocks stockés")
        return result
    except Exception as e:
        ctx_logger.error(f"Erreur lors de la synchronisation des stocks: {e}", exc_info=True)
        result["status"] = "error"
        result["message"] = str(e)
        return result

async def store_stocks_sql(
    db: Session,
    user_id: int,
    account_id: int,
    stocks: List[Dict[str, Any]]
) -> int:
    """
    Stocke les stocks dans la base SQL.
    
    Args:
        db: Session de base de données
        user_id: ID de l'utilisateur
        account_id: ID du compte SQL
        stocks: Liste des stocks à stocker
        
    Returns:
        int: Nombre de stocks stockés
    """
    ctx_logger = get_contextual_logger("sync_service.stock_handler", user_id=user_id, account_id=account_id)
    
    # Obtenir les IDs des stocks existants pour éviter les doublons
    existing_ids = {stock.bridge_stock_id for stock in 
                   db.query(RawStock.bridge_stock_id).filter(
                       RawStock.account_id == account_id,
                       RawStock.user_id == user_id
                   ).all()}
    
    stored_count = 0
    
    for stock_data in stocks:
        bridge_stock_id = stock_data.get("id")
        if not bridge_stock_id:
            ctx_logger.warning(f"Stock sans ID, ignoré")
            continue
            
        try:
            # Vérifier si le stock existe déjà
            if bridge_stock_id in existing_ids:
                # Mise à jour
                stock = db.query(RawStock).filter(
                    RawStock.bridge_stock_id == bridge_stock_id
                ).first()
                
                if stock:
                    stock.label = stock_data.get("label", stock.label)
                    stock.ticker = stock_data.get("ticker", stock.ticker)
                    stock.isin = stock_data.get("isin", stock.isin)
                    stock.marketplace = stock_data.get("marketplace", stock.marketplace)
                    stock.stock_key = stock_data.get("stock_key", stock.stock_key)
                    stock.current_price = stock_data.get("current_price", stock.current_price)
                    stock.currency_code = stock_data.get("currency_code", stock.currency_code)
                    stock.quantity = stock_data.get("quantity", stock.quantity)
                    stock.total_value = stock_data.get("total_value", stock.total_value)
                    stock.average_purchase_price = stock_data.get("average_purchase_price", stock.average_purchase_price)
                    stock.deleted = stock_data.get("deleted", stock.deleted)
                    
                    # Gestion de la date de valeur
                    value_date = stock_data.get("value_date")
                    if value_date:
                        from sync_service.sync_manager.transaction_handler import parse_date
                        stock.value_date = parse_date(value_date)
                    
                    db.add(stock)
                    stored_count += 1
            else:
                # Création
                from sync_service.sync_manager.transaction_handler import parse_date
                new_stock = RawStock(
                    bridge_stock_id=bridge_stock_id,
                    account_id=account_id,
                    user_id=user_id,
                    label=stock_data.get("label"),
                    ticker=stock_data.get("ticker"),
                    isin=stock_data.get("isin"),
                    marketplace=stock_data.get("marketplace"),
                    stock_key=stock_data.get("stock_key"),
                    current_price=stock_data.get("current_price"),
                    currency_code=stock_data.get("currency_code"),
                    quantity=stock_data.get("quantity"),
                    total_value=stock_data.get("total_value"),
                    average_purchase_price=stock_data.get("average_purchase_price"),
                    value_date=parse_date(stock_data.get("value_date")),
                    deleted=stock_data.get("deleted", False)
                )
                
                db.add(new_stock)
                stored_count += 1
                existing_ids.add(bridge_stock_id)
            
            # Commit périodique pour éviter les transactions trop longues
            if stored_count % 50 == 0:
                db.commit()
                
        except Exception as e:
            ctx_logger.error(f"Erreur lors du stockage du stock {bridge_stock_id}: {e}", exc_info=True)
            # Continuer avec le stock suivant
    
    # Commit final
    try:
        db.commit()
        ctx_logger.info(f"{stored_count} stocks stockés avec succès")
    except Exception as e:
        db.rollback()
        ctx_logger.error(f"Erreur lors du commit final des stocks: {e}", exc_info=True)
        raise
        
    return stored_count