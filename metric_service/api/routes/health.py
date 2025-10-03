"""
Routes pour les métriques de santé financière
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

@router.get("/savings-rate/{user_id}", response_model=MetricResponse)
async def get_savings_rate(
    user_id: int,
    period_days: int = Query(30, description="Period in days"),
    use_cache: bool = Query(True)
):
    """
    Calcule le taux d'épargne

    Formula: (Income - Expenses) / Income * 100
    """
    try:
        cache_key = cache_manager.make_key(user_id, "savings_rate", period_days=period_days)

        if use_cache:
            cached = await cache_manager.get(cache_key)
            if cached:
                return MetricResponse(
                    user_id=user_id,
                    metric_type=MetricType.SAVINGS_RATE,
                    computed_at=datetime.fromisoformat(cached['computed_at']),
                    data=cached['data'],
                    cached=True
                )

        savings_data = await metric_calculator.calculate_savings_rate(user_id, period_days)

        response = MetricResponse(
            user_id=user_id,
            metric_type=MetricType.SAVINGS_RATE,
            computed_at=datetime.now(),
            data=savings_data,
            cached=False
        )

        await cache_manager.set(cache_key, response.model_dump(mode='json'), ttl=600)

        return response

    except Exception as e:
        logger.error(f"❌ Error calculating savings rate for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/expense-ratio/{user_id}", response_model=MetricResponse)
async def get_expense_ratio(
    user_id: int,
    period_days: int = Query(30, description="Period in days"),
    use_cache: bool = Query(True)
):
    """
    Calcule les ratios de dépenses (50/30/20 rule)

    - 50% Essentials (loyer, alimentation, transport)
    - 30% Lifestyle (restaurants, loisirs)
    - 20% Savings
    """
    try:
        cache_key = cache_manager.make_key(user_id, "expense_ratio", period_days=period_days)

        if use_cache:
            cached = await cache_manager.get(cache_key)
            if cached:
                return MetricResponse(
                    user_id=user_id,
                    metric_type=MetricType.EXPENSE_RATIO,
                    computed_at=datetime.fromisoformat(cached['computed_at']),
                    data=cached['data'],
                    cached=True
                )

        ratio_data = await metric_calculator.calculate_expense_ratio(user_id, period_days)

        response = MetricResponse(
            user_id=user_id,
            metric_type=MetricType.EXPENSE_RATIO,
            computed_at=datetime.now(),
            data=ratio_data,
            cached=False
        )

        await cache_manager.set(cache_key, response.model_dump(mode='json'), ttl=600)

        return response

    except Exception as e:
        logger.error(f"❌ Error calculating expense ratio for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/burn-rate/{user_id}", response_model=MetricResponse)
async def get_burn_rate(
    user_id: int,
    period_days: int = Query(30, description="Period to calculate burn rate"),
    use_cache: bool = Query(True)
):
    """
    Calcule le burn rate et runway

    Burn Rate: Dépenses mensuelles moyennes
    Runway: Nombre de mois avant épuisement du solde
    """
    try:
        cache_key = cache_manager.make_key(user_id, "burn_rate", period_days=period_days)

        if use_cache:
            cached = await cache_manager.get(cache_key)
            if cached:
                return MetricResponse(
                    user_id=user_id,
                    metric_type=MetricType.BURN_RATE,
                    computed_at=datetime.fromisoformat(cached['computed_at']),
                    data=cached['data'],
                    cached=True
                )

        burn_data = await metric_calculator.calculate_burn_rate(user_id, period_days)

        response = MetricResponse(
            user_id=user_id,
            metric_type=MetricType.BURN_RATE,
            computed_at=datetime.now(),
            data=burn_data,
            cached=False
        )

        await cache_manager.set(cache_key, response.model_dump(mode='json'), ttl=300)

        return response

    except Exception as e:
        logger.error(f"❌ Error calculating burn rate for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/balance-forecast/{user_id}", response_model=MetricResponse)
async def get_balance_forecast(
    user_id: int,
    periods: int = Query(90, description="Number of days to forecast"),
    use_cache: bool = Query(True)
):
    """
    Prévision de solde avec Prophet

    Utilise Prophet pour prévoir le solde sur N jours
    """
    try:
        cache_key = cache_manager.make_key(user_id, "balance_forecast", periods=periods)

        if use_cache:
            cached = await cache_manager.get(cache_key)
            if cached:
                return MetricResponse(
                    user_id=user_id,
                    metric_type=MetricType.BALANCE_FORECAST,
                    computed_at=datetime.fromisoformat(cached['computed_at']),
                    data=cached['data'],
                    cached=True
                )

        forecast_data = await metric_calculator.calculate_balance_forecast(user_id, periods)

        response = MetricResponse(
            user_id=user_id,
            metric_type=MetricType.BALANCE_FORECAST,
            computed_at=datetime.now(),
            data=forecast_data,
            cached=False
        )

        # Cache plus long pour forecasts (1 heure)
        await cache_manager.set(cache_key, response.model_dump(mode='json'), ttl=3600)

        return response

    except Exception as e:
        logger.error(f"❌ Error calculating balance forecast for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/expense-forecast/{user_id}", response_model=MetricResponse)
async def get_expense_forecast(
    user_id: int,
    periods: int = Query(30, description="Number of days to forecast"),
    use_cache: bool = Query(True)
):
    """
    Prévision des dépenses avec Prophet

    Prédit les dépenses futures basées sur l'historique
    """
    try:
        from core.forecaster import forecaster
        from datetime import timedelta

        cache_key = cache_manager.make_key(user_id, "expense_forecast", periods=periods)

        if use_cache:
            cached = await cache_manager.get(cache_key)
            if cached:
                return MetricResponse(
                    user_id=user_id,
                    metric_type=MetricType.BALANCE_FORECAST,  # Réutiliser pour expense forecast
                    computed_at=datetime.fromisoformat(cached['computed_at']),
                    data=cached['data'],
                    cached=True
                )

        # Récupérer historique de transactions
        lookback_days = max(periods * 2, 90)
        start_date = datetime.now() - timedelta(days=lookback_days)

        from core.calculator import metric_calculator
        transactions = await metric_calculator._fetch_transactions(user_id, start_date)

        # Convertir en format pour forecaster
        tx_list = [
            {
                "transaction_date": tx['transaction_date'].isoformat() if isinstance(tx['transaction_date'], datetime) else tx['transaction_date'],
                "amount": tx['amount']
            }
            for tx in transactions
        ]

        # Utiliser le forecaster
        forecast_data = forecaster.forecast_expenses(tx_list, periods)

        response = MetricResponse(
            user_id=user_id,
            metric_type=MetricType.BALANCE_FORECAST,
            computed_at=datetime.now(),
            data=forecast_data,
            cached=False
        )

        # Cache 1 heure
        await cache_manager.set(cache_key, response.model_dump(mode='json'), ttl=3600)

        return response

    except Exception as e:
        logger.error(f"❌ Error calculating expense forecast for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
