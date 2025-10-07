"""
Routes pour les métriques d'optimisation
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Optional
from datetime import datetime
import logging

from metric_service.models.metrics import MetricResponse, MetricType
from metric_service.core.calculator import metric_calculator
from metric_service.core.cache import cache_manager

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/savings-potential/{user_id}", response_model=MetricResponse)
async def get_savings_potential(
    user_id: int,
    lookback_days: int = Query(90, description="Days to analyze for opportunities"),
    use_cache: bool = Query(True)
):
    """
    Calcule le potentiel d'économie

    Identifie les opportunités d'économie:
    - Abonnements inutilisés
    - Dépenses impulsives
    - Sur-dépenses vs pairs
    """
    try:
        cache_key = cache_manager.make_key(user_id, "savings_potential", lookback_days=lookback_days)

        if use_cache:
            cached = await cache_manager.get(cache_key)
            if cached:
                return MetricResponse(
                    user_id=user_id,
                    metric_type=MetricType.SAVINGS_POTENTIAL,
                    computed_at=datetime.fromisoformat(cached['computed_at']),
                    data=cached['data'],
                    cached=True
                )

        potential_data = await metric_calculator.calculate_savings_potential(user_id, lookback_days)

        response = MetricResponse(
            user_id=user_id,
            metric_type=MetricType.SAVINGS_POTENTIAL,
            computed_at=datetime.now(),
            data=potential_data,
            cached=False
        )

        # Cache 30 minutes
        await cache_manager.set(cache_key, response.model_dump(mode='json'), ttl=1800)

        return response

    except Exception as e:
        logger.error(f"❌ Error calculating savings potential for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
