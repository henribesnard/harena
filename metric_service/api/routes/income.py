"""
Routes pour les métriques de REVENUS (YoY et MoM)
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
async def get_yoy_income(
    annee_talon: Optional[int] = Query(None),
    annee_cible: Optional[int] = Query(None),
    categorie: Optional[str] = Query(None),
    marchand: Optional[str] = Query(None),
    user_id: int = Depends(get_current_user_id)
):
    """
    METRIC-003: Year over Year Revenus

    Calcule la variation annuelle des revenus

    **Authentification requise**: Oui (JWT Bearer token)
    """
    try:
        logger.info(f"Calcul YoY revenus pour user {user_id}")

        result = await metric_calculator.calculate_yoy(
            user_id=user_id,
            year=annee_cible,
            transaction_type="income"
        )

        # Reformater selon les specs
        response = {
            "success": True,
            "data": {
                "metric_type": "yoy_income",
                "periode_talon": {
                    "annee": result.get("previous_year"),
                    "total": result.get("previous_amount", 0)
                },
                "periode_cible": {
                    "annee": result.get("current_year"),
                    "total": result.get("current_amount", 0)
                },
                "variation": {
                    "montant": result.get("change_amount", 0),
                    "pourcentage": result.get("change_percent", 0),
                    "direction": "up" if result.get("change_percent", 0) > 0 else "down" if result.get("change_percent", 0) < 0 else "stable"
                },
                "affichage": {
                    "couleur": "green" if result.get("change_percent", 0) > 0 else "red" if result.get("change_percent", 0) < 0 else "gray",
                    "icone": "arrow-up" if result.get("change_percent", 0) > 0 else "arrow-down" if result.get("change_percent", 0) < 0 else "minus",
                    "message": _get_income_message(result.get("change_amount", 0), result.get("change_percent", 0))
                }
            },
            "timestamp": datetime.now().isoformat() + "Z"
        }

        return response

    except Exception as e:
        logger.error(f"❌ Error calculating YoY income: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/mom")
async def get_mom_income(
    mois_talon: Optional[int] = Query(None, ge=1, le=12),
    annee_talon: Optional[int] = Query(None),
    mois_cible: Optional[int] = Query(None, ge=1, le=12),
    annee_cible: Optional[int] = Query(None),
    categorie: Optional[str] = Query(None),
    marchand: Optional[str] = Query(None),
    user_id: int = Depends(get_current_user_id)
):
    """
    METRIC-004: Month over Month Revenus

    Calcule la variation mensuelle des revenus

    **Authentification requise**: Oui (JWT Bearer token)
    """
    try:
        logger.info(f"Calcul MoM revenus pour user {user_id}")

        # Formatter le mois
        month_str = None
        if mois_cible and annee_cible:
            month_str = f"{annee_cible}-{mois_cible:02d}"

        result = await metric_calculator.calculate_mom(
            user_id=user_id,
            month=month_str,
            transaction_type="income"
        )

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
                "metric_type": "mom_income",
                "periode_talon": {
                    "mois": previous_month,
                    "annee": previous_year,
                    "label": f"{mois_names[previous_month-1]} {previous_year}",
                    "total": result.get("previous_amount", 0)
                },
                "periode_cible": {
                    "mois": current_month,
                    "annee": current_year,
                    "label": f"{mois_names[current_month-1]} {current_year}",
                    "total": result.get("current_amount", 0)
                },
                "variation": {
                    "montant": result.get("change_amount", 0),
                    "pourcentage": result.get("change_percent", 0),
                    "direction": "up" if result.get("change_percent", 0) > 0 else "down" if result.get("change_percent", 0) < 0 else "stable"
                },
                "affichage": {
                    "couleur": "green" if result.get("change_percent", 0) > 0 else "red" if result.get("change_percent", 0) < 0 else "gray",
                    "icone": "arrow-up" if result.get("change_percent", 0) > 0 else "arrow-down" if result.get("change_percent", 0) < 0 else "minus",
                    "message": _get_income_message(result.get("change_amount", 0), result.get("change_percent", 0))
                }
            },
            "timestamp": datetime.now().isoformat() + "Z"
        }

        return response

    except Exception as e:
        logger.error(f"❌ Error calculating MoM income: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _get_income_message(change_amount: float, change_percent: float) -> str:
    """Génère le message pour les revenus"""
    montant = abs(change_amount)

    if change_percent > 0.1:
        return f"Vos revenus ont augmenté de {montant:.2f}€"
    elif change_percent < -0.1:
        return f"Vos revenus ont diminué de {montant:.2f}€"
    else:
        return "Vos revenus sont stables"
