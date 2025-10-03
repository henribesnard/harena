"""
Endpoints pour les métriques utilisateur.

Ce module expose les endpoints pour récupérer les métriques financières
de l'utilisateur : soldes par compte et évolutions mensuelles.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func, extract
from typing import Dict, Any, List
from datetime import datetime, timedelta
from decimal import Decimal
import logging

from db_service.session import get_db
from user_service.api.deps import get_current_active_user
from db_service.models.user import User
from db_service.models.sync import SyncAccount, RawTransaction

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/dashboard")
async def get_dashboard_metrics(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Récupère les métriques du tableau de bord pour l'utilisateur connecté.

    Retourne:
    - soldes par compte (pas de solde total)
    - évolution des dépenses du mois en cours vs mois précédent (%)
    - évolution des revenus du mois en cours vs mois précédent (%)
    """
    # 1. Récupérer les soldes par compte
    accounts = db.query(SyncAccount).join(
        SyncAccount.item
    ).filter(
        SyncAccount.item.has(user_id=current_user.id)
    ).all()

    account_balances = [
        {
            "account_id": acc.bridge_account_id,
            "account_name": acc.account_name,
            "balance": float(acc.balance) if acc.balance else 0.0,
            "currency_code": acc.currency_code or "EUR",
            "account_type": acc.account_type,
            "updated_at": acc.last_sync_timestamp.isoformat() if acc.last_sync_timestamp else None
        }
        for acc in accounts
    ]

    # 2. Calculer l'évolution des dépenses et revenus
    now = datetime.now()
    current_month_start = datetime(now.year, now.month, 1)

    # Début du mois précédent
    if now.month == 1:
        previous_month_start = datetime(now.year - 1, 12, 1)
        previous_month_end = datetime(now.year, 1, 1) - timedelta(days=1)
    else:
        previous_month_start = datetime(now.year, now.month - 1, 1)
        previous_month_end = current_month_start - timedelta(days=1)

    # Dépenses du mois en cours (montants négatifs)
    current_expenses = db.query(
        func.sum(RawTransaction.amount)
    ).filter(
        RawTransaction.user_id == current_user.id,
        RawTransaction.amount < 0,
        RawTransaction.transaction_date >= current_month_start,
        RawTransaction.transaction_date < now
    ).scalar() or 0

    # Dépenses du mois précédent
    previous_expenses = db.query(
        func.sum(RawTransaction.amount)
    ).filter(
        RawTransaction.user_id == current_user.id,
        RawTransaction.amount < 0,
        RawTransaction.transaction_date >= previous_month_start,
        RawTransaction.transaction_date <= previous_month_end
    ).scalar() or 0

    # Revenus du mois en cours (montants positifs)
    current_income = db.query(
        func.sum(RawTransaction.amount)
    ).filter(
        RawTransaction.user_id == current_user.id,
        RawTransaction.amount > 0,
        RawTransaction.transaction_date >= current_month_start,
        RawTransaction.transaction_date < now
    ).scalar() or 0

    # Revenus du mois précédent
    previous_income = db.query(
        func.sum(RawTransaction.amount)
    ).filter(
        RawTransaction.user_id == current_user.id,
        RawTransaction.amount > 0,
        RawTransaction.transaction_date >= previous_month_start,
        RawTransaction.transaction_date <= previous_month_end
    ).scalar() or 0

    # Calculer les évolutions en pourcentage
    expenses_evolution = calculate_evolution(
        float(current_expenses),
        float(previous_expenses)
    )

    income_evolution = calculate_evolution(
        float(current_income),
        float(previous_income)
    )

    return {
        "accounts": account_balances,
        "expenses": {
            "current_month": abs(float(current_expenses)),
            "previous_month": abs(float(previous_expenses)),
            "evolution_percent": expenses_evolution
        },
        "income": {
            "current_month": float(current_income),
            "previous_month": float(previous_income),
            "evolution_percent": income_evolution
        },
        "period": {
            "current_month_start": current_month_start.isoformat(),
            "previous_month_start": previous_month_start.isoformat(),
            "previous_month_end": previous_month_end.isoformat()
        }
    }


def calculate_evolution(current: float, previous: float) -> float:
    """
    Calcule l'évolution en pourcentage entre deux valeurs.

    Args:
        current: Valeur actuelle
        previous: Valeur précédente

    Returns:
        float: Évolution en pourcentage (positive = augmentation, négative = diminution)
    """
    if previous == 0:
        if current == 0:
            return 0.0
        return 100.0 if current > 0 else -100.0

    evolution = ((current - previous) / abs(previous)) * 100
    return round(evolution, 1)
