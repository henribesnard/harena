"""
Endpoints pour la gestion des transactions.

Ce module expose les endpoints pour accéder et rechercher
les transactions bancaires stockées.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import Dict, Any, List, Optional
from datetime import datetime, date

from user_service.db.session import get_db
from user_service.api.deps import get_current_active_user
from user_service.models.user import User
from sync_service.models.sync import RawTransaction, SyncAccount, BridgeCategory
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/")
async def get_transactions(
    account_id: Optional[int] = None,
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
    min_date: Optional[date] = None,
    max_date: Optional[date] = None,
    limit: int = Query(50, gt=0, le=500),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Récupère les transactions de l'utilisateur avec filtres optionnels.
    
    Args:
        account_id: Filtrer par ID de compte
        since: Récupérer les transactions mises à jour après cette date
        until: Récupérer les transactions mises à jour avant cette date
        min_date: Date minimale des transactions
        max_date: Date maximale des transactions
        limit: Nombre maximal de résultats
        offset: Décalage pour la pagination
        
    Returns:
        Dict: Liste des transactions et informations de pagination
    """
    query = db.query(RawTransaction).filter(RawTransaction.user_id == current_user.id)
    
    # Appliquer les filtres
    if account_id:
        query = query.filter(RawTransaction.account_id == account_id)
    
    if since:
        query = query.filter(RawTransaction.updated_at_bridge >= since)
    
    if until:
        query = query.filter(RawTransaction.updated_at_bridge <= until)
    
    if min_date:
        query = query.filter(RawTransaction.date >= min_date)
    
    if max_date:
        query = query.filter(RawTransaction.date <= max_date)
    
    # Compter le total avant pagination
    total_count = query.count()
    
    # Appliquer la pagination et récupérer les résultats
    transactions = query.order_by(RawTransaction.date.desc()).offset(offset).limit(limit).all()
    
    # Formater les résultats
    result = {
        "items": [format_transaction(tx) for tx in transactions],
        "total": total_count,
        "limit": limit,
        "offset": offset,
        "has_more": (offset + limit) < total_count
    }
    
    return result

@router.get("/{transaction_id}")
async def get_transaction(
    transaction_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Récupère une transaction spécifique par son ID.
    
    Args:
        transaction_id: ID de la transaction Bridge
        
    Returns:
        Dict: Détails de la transaction
    """
    transaction = db.query(RawTransaction).filter(
        RawTransaction.bridge_transaction_id == transaction_id,
        RawTransaction.user_id == current_user.id
    ).first()
    
    if not transaction:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Transaction {transaction_id} not found"
        )
    
    return format_transaction(transaction)

@router.get("/categories")
async def get_categories(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Récupère toutes les catégories disponibles.
    
    Returns:
        List: Liste des catégories
    """
    categories = db.query(BridgeCategory).all()
    return [format_category(cat) for cat in categories]

@router.get("/categories/{category_id}")
async def get_category(
    category_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Récupère une catégorie spécifique par son ID.
    
    Args:
        category_id: ID de la catégorie Bridge
        
    Returns:
        Dict: Détails de la catégorie
    """
    category = db.query(BridgeCategory).filter(
        BridgeCategory.bridge_category_id == category_id
    ).first()
    
    if not category:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Category {category_id} not found"
        )
    
    return format_category(category)

# Fonctions utilitaires pour formater les réponses
def format_transaction(transaction: RawTransaction) -> Dict[str, Any]:
    """
    Formate une transaction pour la réponse API.
    
    Args:
        transaction: Transaction brute depuis la BDD
        
    Returns:
        Dict: Transaction formatée
    """
    return {
        "id": transaction.bridge_transaction_id,
        "clean_description": transaction.clean_description,
        "provider_description": transaction.provider_description,
        "amount": transaction.amount,
        "date": transaction.date.strftime("%Y-%m-%d") if transaction.date else None,
        "booking_date": transaction.booking_date.strftime("%Y-%m-%d") if transaction.booking_date else None,
        "transaction_date": transaction.transaction_date.strftime("%Y-%m-%d") if transaction.transaction_date else None,
        "value_date": transaction.value_date.strftime("%Y-%m-%d") if transaction.value_date else None,
        "updated_at": transaction.updated_at_bridge.isoformat() if transaction.updated_at_bridge else None,
        "currency_code": transaction.currency_code,
        "deleted": transaction.deleted,
        "category_id": transaction.category_id,
        "operation_type": transaction.operation_type,
        "account_id": transaction.account_id,
        "future": transaction.future
    }

def format_category(category: BridgeCategory) -> Dict[str, Any]:
    """
    Formate une catégorie pour la réponse API.
    
    Args:
        category: Catégorie depuis la BDD
        
    Returns:
        Dict: Catégorie formatée
    """
    return {
        "id": category.bridge_category_id,
        "name": category.name,
        "parent_id": category.parent_id,
        "parent_name": category.parent_name
    }