"""
Routes pour les métriques de DÉPENSES (YoY et MoM)
Conforme aux specs harena-metrics-5-specs.md
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Optional
from datetime import datetime
import logging

from metric_service.core.calculator import metric_calculator
from metric_service.api.dependencies import get_current_user_id

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/yoy")
async def get_yoy_expenses(
    annee_talon: Optional[int] = Query(None, description="Année de référence"),
    annee_cible: Optional[int] = Query(None, description="Année à comparer"),
    categorie: Optional[str] = Query(None),
    marchand: Optional[str] = Query(None),
    type_transaction: str = Query("expense"),
    user_id: int = Depends(get_current_user_id)
):
    """
    METRIC-001: Year over Year Dépenses

    Calcule la variation annuelle des dépenses

    **Authentification requise**: Oui (JWT Bearer token)
    """
    try:
        logger.info(f"Calcul YoY dépenses pour user {user_id}")

        # Calculer via le service (déjà implémenté avec transaction_type)
        result = await metric_calculator.calculate_yoy(
            user_id=user_id,
            year=annee_cible,
            transaction_type="expenses"
        )

        # Récupérer les comptes utilisés
        accounts_used = await metric_calculator.get_accounts_used(user_id)

        # Reformater selon les specs
        response = {
            "success": True,
            "data": {
                "metric_type": "yoy_expenses",
                "periode_talon": {
                    "annee": result.get("previous_year"),
                    "total": abs(result.get("previous_amount", 0))
                },
                "periode_cible": {
                    "annee": result.get("current_year"),
                    "total": abs(result.get("current_amount", 0))
                },
                "variation": {
                    "montant": result.get("change_amount", 0),
                    "pourcentage": result.get("change_percent", 0),
                    "direction": "down" if result.get("change_percent", 0) < 0 else "up" if result.get("change_percent", 0) > 0 else "stable"
                },
                "affichage": {
                    "couleur": "green" if result.get("change_percent", 0) < 0 else "red" if result.get("change_percent", 0) > 0 else "gray",
                    "icone": "arrow-down" if result.get("change_percent", 0) < 0 else "arrow-up" if result.get("change_percent", 0) > 0 else "minus",
                    "message": _get_expense_message(result.get("change_amount", 0), result.get("change_percent", 0))
                },
                "filtres_appliques": {
                    "categorie": categorie,
                    "marchand": marchand,
                    "type_transaction": type_transaction
                },
                "accounts_used": accounts_used
            },
            "timestamp": datetime.now().isoformat() + "Z"
        }

        return response

    except Exception as e:
        logger.error(f"❌ Error calculating YoY expenses: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/mom")
async def get_mom_expenses(
    mois_talon: Optional[int] = Query(None, ge=1, le=12),
    annee_talon: Optional[int] = Query(None),
    mois_cible: Optional[int] = Query(None, ge=1, le=12),
    annee_cible: Optional[int] = Query(None),
    categorie: Optional[str] = Query(None),
    marchand: Optional[str] = Query(None),
    type_transaction: str = Query("expense"),
    user_id: int = Depends(get_current_user_id)
):
    """
    METRIC-002: Month over Month Dépenses

    Calcule la variation mensuelle des dépenses

    **Authentification requise**: Oui (JWT Bearer token)
    """
    try:
        logger.info(f"Calcul MoM dépenses pour user {user_id}")

        # Formatter le mois pour calculate_mom
        month_str = None
        if mois_cible and annee_cible:
            month_str = f"{annee_cible}-{mois_cible:02d}"

        result = await metric_calculator.calculate_mom(
            user_id=user_id,
            month=month_str,
            transaction_type="expenses"
        )

        # Récupérer les comptes utilisés
        accounts_used = await metric_calculator.get_accounts_used(user_id)

        # Reformater selon les specs
        mois_names = [
            "Janvier", "Février", "Mars", "Avril", "Mai", "Juin",
            "Juillet", "Août", "Septembre", "Octobre", "Novembre", "Décembre"
        ]

        current_month = int(result.get("current_month", "2025-01").split("-")[1])
        previous_month = int(result.get("previous_month", "2025-01").split("-")[1])
        current_year = int(result.get("current_month", "2025-01").split("-")[0])
        previous_year = int(result.get("previous_month", "2025-01").split("-")[0])

        response = {
            "success": True,
            "data": {
                "metric_type": "mom_expenses",
                "periode_talon": {
                    "mois": previous_month,
                    "annee": previous_year,
                    "label": f"{mois_names[previous_month-1]} {previous_year}",
                    "total": abs(result.get("previous_amount", 0))
                },
                "periode_cible": {
                    "mois": current_month,
                    "annee": current_year,
                    "label": f"{mois_names[current_month-1]} {current_year}",
                    "total": abs(result.get("current_amount", 0))
                },
                "variation": {
                    "montant": result.get("change_amount", 0),
                    "pourcentage": result.get("change_percent", 0),
                    "direction": "down" if result.get("change_percent", 0) < 0 else "up" if result.get("change_percent", 0) > 0 else "stable"
                },
                "affichage": {
                    "couleur": "green" if result.get("change_percent", 0) < 0 else "red" if result.get("change_percent", 0) > 0 else "gray",
                    "icone": "arrow-down" if result.get("change_percent", 0) < 0 else "arrow-up" if result.get("change_percent", 0) > 0 else "minus",
                    "message": _get_expense_message(result.get("change_amount", 0), result.get("change_percent", 0))
                },
                "accounts_used": accounts_used
            },
            "timestamp": datetime.now().isoformat() + "Z"
        }

        return response

    except Exception as e:
        logger.error(f"❌ Error calculating MoM expenses: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _get_expense_message(change_amount: float, change_percent: float) -> str:
    """Génère le message pour les dépenses"""
    montant = abs(change_amount)

    if change_percent < -0.1:
        return f"Vous avez réduit vos dépenses de {montant:.2f}€"
    elif change_percent > 0.1:
        return f"Vos dépenses ont augmenté de {montant:.2f}€"
    else:
        return "Vos dépenses sont stables"
