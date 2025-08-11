"""
Endpoints pour la gestion des stocks et titres financiers.

Ce module expose les endpoints pour accéder aux titres financiers
stockés dans la base de données.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import Dict, Any, List, Optional
from datetime import datetime

from conversation_service.api.dependencies import get_db
from user_service.api.deps import get_current_active_user
from db_service.models.user import User
from db_service.models.sync import RawStock, SyncAccount
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/")
async def get_stocks(
    account_id: Optional[int] = None,
    since: Optional[datetime] = None,
    limit: int = Query(50, gt=0, le=500),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Récupère tous les stocks de l'utilisateur avec filtres optionnels.
    
    Args:
        account_id: Filtrer par ID de compte
        since: Récupérer les stocks mis à jour après cette date
        limit: Nombre maximal de résultats
        offset: Décalage pour la pagination
        
    Returns:
        Dict: Liste des stocks et informations de pagination
    """
    # Construire la requête de base
    query = db.query(RawStock).filter(RawStock.user_id == current_user.id)
    
    # Appliquer les filtres
    if account_id:
        # Vérifier que le compte appartient à l'utilisateur
        account = db.query(SyncAccount).join(
            SyncAccount.item
        ).filter(
            SyncAccount.bridge_account_id == account_id,
            SyncAccount.item.has(user_id=current_user.id)
        ).first()
        
        if not account:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Account {account_id} not found"
            )
            
        query = query.filter(RawStock.account_id == account.id)
    
    if since:
        query = query.filter(RawStock.updated_at >= since)
    
    # Ne montrer que les stocks non supprimés par défaut
    query = query.filter(RawStock.deleted.is_(False))
    
    # Compter le total avant pagination
    total_count = query.count()
    
    # Appliquer la pagination et récupérer les résultats
    stocks = query.order_by(RawStock.updated_at.desc()).offset(offset).limit(limit).all()
    
    # Formater les résultats
    result = {
        "items": [format_stock(stock) for stock in stocks],
        "total": total_count,
        "limit": limit,
        "offset": offset,
        "has_more": (offset + limit) < total_count
    }
    
    return result

@router.get("/{stock_id}")
async def get_stock(
    stock_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Récupère un stock spécifique par son ID.
    
    Args:
        stock_id: ID du stock Bridge
        
    Returns:
        Dict: Détails du stock
    """
    stock = db.query(RawStock).filter(
        RawStock.bridge_stock_id == stock_id,
        RawStock.user_id == current_user.id
    ).first()
    
    if not stock:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Stock {stock_id} not found"
        )
    
    return format_stock(stock)

@router.post("/refresh")
async def refresh_stocks(
    account_id: Optional[int] = None,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Rafraîchit les stocks d'un compte ou de tous les comptes.
    
    Args:
        account_id: ID du compte Bridge (optionnel, si non fourni tous les comptes sont rafraîchis)
        
    Returns:
        Dict: Résultat du rafraîchissement
    """
    from sync_service.sync_manager.stock_handler import sync_all_stocks
    from fastapi import BackgroundTasks
    
    # Si un compte spécifique est demandé
    if account_id:
        # Vérifier que le compte appartient à l'utilisateur
        account = db.query(SyncAccount).join(
            SyncAccount.item
        ).filter(
            SyncAccount.bridge_account_id == account_id,
            SyncAccount.item.has(user_id=current_user.id)
        ).first()
        
        if not account:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Account {account_id} not found"
            )
            
        # Lancer la synchro en arrière-plan
        background_tasks = BackgroundTasks()
        background_tasks.add_task(
            sync_single_account_stocks,
            user_id=current_user.id,
            account_id=account.id
        )
        
        return {
            "status": "success",
            "message": f"Stock refresh initiated for account {account_id}",
            "info": "The refresh is running in the background and will continue after this response is sent."
        }
    
    # Sinon, rafraîchir tous les comptes
    background_tasks = BackgroundTasks()
    background_tasks.add_task(
        sync_all_stocks,
        db=db,
        user_id=current_user.id
    )
    
    return {
        "status": "success",
        "message": "Stock refresh initiated for all accounts",
        "info": "The refresh is running in the background and will continue after this response is sent."
    }

# Fonction auxiliaire pour le traitement en arrière-plan
async def sync_single_account_stocks(user_id: int, account_id: int):
    """
    Synchronise les stocks d'un compte spécifique en arrière-plan.
    
    Args:
        user_id: ID de l'utilisateur
        account_id: ID du compte SQL
    """
    # Créer une nouvelle session car nous sommes dans un thread/coroutine distinct
    from user_service.db.session import SessionLocal
    db = SessionLocal()
    
    try:
        from user_service.services.bridge import get_bridge_token
        from sync_service.sync_manager.stock_handler import store_stocks_sql
        
        logger.info(f"Starting background stock sync for account {account_id}")
        
        # Récupérer le compte
        account = db.query(SyncAccount).get(account_id)
        if not account:
            logger.warning(f"Account {account_id} not found during background stock sync")
            return
            
        # Récupérer le token Bridge
        token_data = await get_bridge_token(db, user_id)
        access_token = token_data["access_token"]
        
        # Récupérer les stocks
        from user_service.services.bridge import get_bridge_stocks
        stocks = await get_bridge_stocks(
            db, 
            user_id, 
            access_token, 
            account_id=account.bridge_account_id
        )
        
        if stocks:
            # Stocker les stocks
            stored_count = await store_stocks_sql(db, user_id, account_id, stocks)
            logger.info(f"Background stock sync completed: {stored_count}/{len(stocks)} stocks stored")
        else:
            logger.info(f"No stocks found for account {account.bridge_account_id}")
        
    except Exception as e:
        logger.error(f"Error during background stock sync: {str(e)}", exc_info=True)
    finally:
        db.close()

# Fonctions utilitaires pour formater les réponses
def format_stock(stock: RawStock) -> Dict[str, Any]:
    """
    Formate un stock pour la réponse API.
    
    Args:
        stock: Stock depuis la BDD
        
    Returns:
        Dict: Stock formaté
    """
    return {
        "id": stock.bridge_stock_id,
        "account_id": stock.account_id,
        "label": stock.label,
        "ticker": stock.ticker,
        "marketplace": stock.marketplace,
        "isin": stock.isin,
        "stock_key": stock.stock_key,
        "current_price": stock.current_price,
        "currency_code": stock.currency_code,
        "quantity": stock.quantity,
        "total_value": stock.total_value,
        "average_purchase_price": stock.average_purchase_price,
        "value_date": stock.value_date.strftime("%Y-%m-%d") if stock.value_date else None,
        "deleted": stock.deleted,
        "created_at": stock.created_at.isoformat(),
        "updated_at": stock.updated_at.isoformat()
    }