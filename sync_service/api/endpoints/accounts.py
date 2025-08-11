"""
Endpoints pour la gestion des comptes bancaires.

Ce module expose les endpoints pour accéder aux comptes bancaires
et leurs détails stockés dans la base de données.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import Dict, Any, List, Optional

from conversation_service.api.dependencies import get_db
from user_service.api.deps import get_current_active_user
from db_service.models.user import User
from db_service.models.sync import SyncAccount, LoanDetail, RawStock, AccountInformation
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/")
async def get_accounts(
    item_id: Optional[int] = None,
    account_type: Optional[str] = None,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Récupère tous les comptes de l'utilisateur avec filtres optionnels.
    
    Args:
        item_id: Filtrer par ID d'item
        account_type: Filtrer par type de compte
        
    Returns:
        List: Liste des comptes
    """
    # Construire la requête de base
    query = db.query(SyncAccount).join(
        SyncAccount.item
    ).filter(
        SyncAccount.item.has(user_id=current_user.id)
    )
    
    # Appliquer les filtres supplémentaires
    if item_id:
        query = query.filter(SyncAccount.item.has(bridge_item_id=item_id))
    
    if account_type:
        query = query.filter(SyncAccount.account_type == account_type)
    
    # Exécuter la requête
    accounts = query.all()
    
    # Formater les résultats
    return [format_account(db, acc) for acc in accounts]

@router.get("/{account_id}")
async def get_account(
    account_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Récupère un compte spécifique par son ID.
    
    Args:
        account_id: ID du compte Bridge
        
    Returns:
        Dict: Détails du compte
    """
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
    
    return format_account(db, account)

@router.get("/{account_id}/transactions")
async def get_account_transactions(
    account_id: int,
    limit: int = Query(50, gt=0, le=500),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Récupère les transactions d'un compte spécifique.
    
    Args:
        account_id: ID du compte Bridge
        limit: Nombre maximal de résultats
        offset: Décalage pour la pagination
        
    Returns:
        Dict: Liste des transactions et informations de pagination
    """
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
    
    # Récupérer les transactions
    from sync_service.api.endpoints.transactions import get_transactions
    return await get_transactions(
        account_id=account.id,
        limit=limit,
        offset=offset,
        current_user=current_user,
        db=db
    )

@router.get("/{account_id}/stocks")
async def get_account_stocks(
    account_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Récupère les stocks/titres d'un compte spécifique.
    
    Args:
        account_id: ID du compte Bridge
        
    Returns:
        List: Liste des stocks
    """
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
    
    # Récupérer les stocks
    stocks = db.query(RawStock).filter(
        RawStock.account_id == account.id,
        RawStock.user_id == current_user.id,
        RawStock.deleted.is_(False)
    ).all()
    
    return [format_stock(stock) for stock in stocks]

# Fonctions utilitaires pour formater les réponses
def format_account(db: Session, account: SyncAccount) -> Dict[str, Any]:
    """
    Formate un compte pour la réponse API.
    
    Args:
        db: Session de base de données
        account: Compte depuis la BDD
        
    Returns:
        Dict: Compte formaté
    """
    result = {
        "id": account.bridge_account_id,
        "name": account.account_name,
        "balance": account.balance,
        "updated_at": account.last_sync_timestamp.isoformat() if account.last_sync_timestamp else None,
        "type": account.account_type,
        "currency_code": account.currency_code,
        "item_id": account.item.bridge_item_id,
        "provider_id": account.item.provider_id,
        "last_refresh_status": "successful" if account.item.status == 0 else "failed"
    }
    
    # Ajouter les détails de prêt si disponibles
    if account.account_type == "loan" and account.loan_details:
        loan = account.loan_details
        result["loan_details"] = {
            "next_payment_date": loan.next_payment_date.strftime("%Y-%m-%d") if loan.next_payment_date else None,
            "next_payment_amount": loan.next_payment_amount,
            "maturity_date": loan.maturity_date.strftime("%Y-%m-%d") if loan.maturity_date else None,
            "opening_date": loan.opening_date.strftime("%Y-%m-%d") if loan.opening_date else None,
            "interest_rate": loan.interest_rate,
            "borrowed_capital": loan.borrowed_capital,
            "repaid_capital": loan.repaid_capital,
            "remaining_capital": loan.remaining_capital,
            "total_estimated_interests": loan.total_estimated_interests
        }
    
    # Ajouter l'IBAN si disponible
    account_info = db.query(AccountInformation).filter(
        AccountInformation.item_id == account.item_id,
        AccountInformation.user_id == account.item.user_id
    ).first()
    
    if account_info and account_info.account_details:
        for acc_detail in account_info.account_details:
            if acc_detail.get("id") == account.bridge_account_id and "iban" in acc_detail:
                result["iban"] = acc_detail["iban"]
                result["iban_filled_manually"] = False
                break
    
    return result

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