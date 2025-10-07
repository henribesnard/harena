"""
Routes pour la métrique de Taux de Couverture
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Optional
from datetime import datetime
import logging

from metric_service.models.metrics import MetricResponse, MetricType
from metric_service.core.calculator import metric_calculator
from metric_service.core.cache import cache_manager
from metric_service.api.dependencies import get_current_user_id

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("")
async def get_coverage_rate(
    mois: Optional[int] = Query(None, ge=1, le=12, description="Mois (1-12), défaut: mois actuel"),
    annee: Optional[int] = Query(None, description="Année, défaut: année actuelle"),
    user_id: int = Depends(get_current_user_id)
):
    """
    METRIC-005: Taux de Couverture Mensuel

    Taux_Couverture (%) = ((Revenus - Dépenses) / Revenus) × 100

    Par défaut: mois en cours

    **Authentification requise**: Oui (JWT Bearer token)
    """
    try:
        logger.info(f"Calcul taux de couverture pour user {user_id}")

        coverage_data = await metric_calculator.calculate_coverage_rate(user_id, mois, annee)

        # Reformater selon les specs
        response = {
            "success": True,
            "data": {
                "metric_type": "coverage_rate",
                "periode": {
                    "mois": coverage_data["periode"]["mois"],
                    "annee": coverage_data["periode"]["annee"],
                    "label": coverage_data["periode"]["label"],
                    "is_current_month": coverage_data["periode"]["is_current_month"]
                },
                "revenus": coverage_data["revenus"],
                "depenses": coverage_data["depenses"],
                "solde": coverage_data["solde"],
                "taux_couverture": coverage_data["taux_couverture"],
                "affichage": {
                    "couleur": coverage_data["affichage"]["couleur"],
                    "niveau": coverage_data["affichage"]["niveau"],
                    "message": coverage_data["affichage"]["message"]
                },
                "mise_a_jour": coverage_data["mise_a_jour"]
            },
            "timestamp": datetime.now().isoformat() + "Z"
        }

        return response

    except Exception as e:
        logger.error(f"❌ Error calculating Coverage Rate: {e}")
        raise HTTPException(status_code=500, detail=str(e))
