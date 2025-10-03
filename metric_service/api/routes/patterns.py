"""
Routes pour les patterns financiers (récurrences, anomalies)
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

@router.get("/recurring/{user_id}", response_model=MetricResponse)
async def get_recurring_expenses(
    user_id: int,
    min_occurrences: int = Query(3, description="Minimum occurrences to be considered recurring"),
    lookback_days: int = Query(90, description="Days to look back for patterns"),
    use_cache: bool = Query(True)
):
    """
    Détecte les dépenses récurrentes

    Identifie les transactions qui se répètent régulièrement
    (abonnements, loyer, etc.)
    """
    try:
        cache_key = cache_manager.make_key(
            user_id,
            "recurring",
            min_occurrences=min_occurrences,
            lookback_days=lookback_days
        )

        if use_cache:
            cached = await cache_manager.get(cache_key)
            if cached:
                return MetricResponse(
                    user_id=user_id,
                    metric_type=MetricType.RECURRING,
                    computed_at=datetime.fromisoformat(cached['computed_at']),
                    data=cached['data'],
                    cached=True
                )

        recurring_data = await metric_calculator.calculate_recurring_expenses(
            user_id,
            min_occurrences,
            lookback_days
        )

        response = MetricResponse(
            user_id=user_id,
            metric_type=MetricType.RECURRING,
            computed_at=datetime.now(),
            data=recurring_data,
            cached=False
        )

        # Cache 30 minutes
        await cache_manager.set(cache_key, response.model_dump(mode='json'), ttl=1800)

        return response

    except Exception as e:
        logger.error(f"❌ Error detecting recurring expenses for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
