"""
Routes pour les métriques de tendances (MoM, YoY)
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Optional
from datetime import datetime
import logging

from metric_service.models.metrics import MetricResponse, MoMMetric, YoYMetric, MetricType
from metric_service.core.calculator import metric_calculator
from metric_service.core.cache import cache_manager

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/mom/{user_id}", response_model=MetricResponse)
async def get_month_over_month(
    user_id: int,
    month: Optional[str] = Query(None, description="YYYY-MM format, defaults to current month"),
    category: Optional[str] = Query(None, description="Filter by category"),
    transaction_type: str = Query("expenses", description="Type: 'expenses' or 'income'"),
    use_cache: bool = Query(True, description="Use cached result if available")
):
    """
    Calcule le Month-over-Month pour un utilisateur

    Compare le mois spécifié (ou actuel) avec le mois précédent
    - expenses: Dépenses (montants négatifs)
    - income: Revenus (montants positifs)
    """
    try:
        # Générer la clé de cache
        cache_key = cache_manager.make_key(user_id, "mom", month=month, category=category, transaction_type=transaction_type)

        # Vérifier le cache
        if use_cache:
            cached = await cache_manager.get(cache_key)
            if cached:
                logger.info(f"✅ MoM {transaction_type} from cache for user {user_id}")
                return MetricResponse(
                    user_id=user_id,
                    metric_type=MetricType.MOM,
                    computed_at=datetime.fromisoformat(cached['computed_at']),
                    data=cached['data'],
                    cached=True
                )

        # Calculer la métrique
        mom_data = await metric_calculator.calculate_mom(user_id, month, category, transaction_type)

        response = MetricResponse(
            user_id=user_id,
            metric_type=MetricType.MOM,
            computed_at=datetime.now(),
            data=mom_data,
            cached=False
        )

        # Mettre en cache (5 minutes)
        await cache_manager.set(cache_key, response.model_dump(mode='json'), ttl=300)

        return response

    except Exception as e:
        logger.error(f"❌ Error calculating MoM {transaction_type} for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/yoy/{user_id}", response_model=MetricResponse)
async def get_year_over_year(
    user_id: int,
    year: Optional[int] = Query(None, description="Year, defaults to current year"),
    category: Optional[str] = Query(None, description="Filter by category"),
    transaction_type: str = Query("expenses", description="Type: 'expenses' or 'income'"),
    use_cache: bool = Query(True, description="Use cached result if available")
):
    """
    Calcule le Year-over-Year pour un utilisateur

    Compare l'année spécifiée (ou actuelle) avec l'année précédente
    - expenses: Dépenses (montants négatifs)
    - income: Revenus (montants positifs)
    """
    try:
        cache_key = cache_manager.make_key(user_id, "yoy", year=year, category=category, transaction_type=transaction_type)

        if use_cache:
            cached = await cache_manager.get(cache_key)
            if cached:
                logger.info(f"✅ YoY {transaction_type} from cache for user {user_id}")
                return MetricResponse(
                    user_id=user_id,
                    metric_type=MetricType.YOY,
                    computed_at=datetime.fromisoformat(cached['computed_at']),
                    data=cached['data'],
                    cached=True
                )

        yoy_data = await metric_calculator.calculate_yoy(user_id, year, category, transaction_type)

        response = MetricResponse(
            user_id=user_id,
            metric_type=MetricType.YOY,
            computed_at=datetime.now(),
            data=yoy_data,
            cached=False
        )

        # Cache plus long pour YoY (1 heure)
        await cache_manager.set(cache_key, response.model_dump(mode='json'), ttl=3600)

        return response

    except Exception as e:
        logger.error(f"❌ Error calculating YoY {transaction_type} for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
