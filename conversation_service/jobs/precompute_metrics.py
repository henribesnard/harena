"""
Batch Job: Pre-Compute Metrics

Nightly job (3 AM) that pre-computes metrics for all active users:
- Monthly totals (spending, income)
- Category breakdowns
- MoM/YoY comparisons
- 6-month trends

Storage:
- Redis (24h cache) for fast access
- PostgreSQL (pre_computed_metrics) for historical data

Sprint 1.2 - T2.3
"""

from celery import shared_task
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import logging
from sqlalchemy.orm import Session

from conversation_service.jobs.celery_app import celery_app
from conversation_service.config.settings import settings
from conversation_service.models.user_profile.entities import UserProfileDB, PreComputedMetric
from conversation_service.services.metrics.metrics_service import MetricsService
from conversation_service.services.user_profile.profile_service import UserProfileService

logger = logging.getLogger(__name__)


def get_db_session():
    """Get database session for Celery tasks"""
    # Import here to avoid circular imports
    from conversation_service.api.dependencies import get_db
    return next(get_db())


@celery_app.task(name='conversation_service.jobs.precompute_metrics.precompute_all_users')
def precompute_all_users(days_active: int = 30, max_users: int = 1000):
    """
    Main job: Pre-compute metrics for all active users

    Args:
        days_active: Consider users active in last N days
        max_users: Maximum number of users to process

    Returns:
        Dict with job statistics
    """
    start_time = datetime.utcnow()
    logger.info(f"Starting pre-compute job for users active in last {days_active} days")

    db = get_db_session()
    profile_service = UserProfileService(db)

    try:
        # Get active users
        cutoff_date = datetime.utcnow() - timedelta(days=days_active)
        active_profiles = db.query(UserProfileDB).filter(
            UserProfileDB.last_active >= cutoff_date
        ).order_by(UserProfileDB.last_active.desc()).limit(max_users).all()

        logger.info(f"Found {len(active_profiles)} active users to process")

        # Process each user (spawn subtasks)
        results = []
        for profile in active_profiles:
            result = precompute_user_metrics.delay(profile.user_id)
            results.append(result)

        # Wait for all tasks to complete (with timeout)
        success_count = 0
        failure_count = 0
        for result in results:
            try:
                result.get(timeout=60)  # 1 minute timeout per user
                success_count += 1
            except Exception as e:
                logger.error(f"Task failed: {e}")
                failure_count += 1

        elapsed_time = (datetime.utcnow() - start_time).total_seconds()

        stats = {
            'total_users': len(active_profiles),
            'success_count': success_count,
            'failure_count': failure_count,
            'elapsed_seconds': round(elapsed_time, 2),
            'started_at': start_time.isoformat(),
            'completed_at': datetime.utcnow().isoformat()
        }

        logger.info(f"Pre-compute job completed: {stats}")
        return stats

    except Exception as e:
        logger.error(f"Pre-compute job failed: {e}", exc_info=True)
        raise
    finally:
        db.close()


@celery_app.task(name='conversation_service.jobs.precompute_metrics.precompute_user_metrics')
def precompute_user_metrics(user_id: int):
    """
    Pre-compute metrics for a single user

    Computes:
    - Monthly totals
    - Category breakdown
    - MoM comparison
    - 6-month trends

    Args:
        user_id: User ID

    Returns:
        Dict with computation statistics
    """
    start_time = datetime.utcnow()
    logger.info(f"Computing metrics for user_id={user_id}")

    db = get_db_session()
    metrics_service = MetricsService(db)

    try:
        current_period = datetime.utcnow().strftime('%Y-%m')
        previous_period = (datetime.utcnow() - timedelta(days=30)).strftime('%Y-%m')

        # 1. Compute monthly totals
        monthly_metrics = compute_monthly_totals(db, user_id, current_period)
        if monthly_metrics:
            metrics_service.store_metrics_sync(
                user_id=user_id,
                metric_type='monthly_total',
                period=current_period,
                metric_value=monthly_metrics,
                computation_time_ms=int((datetime.utcnow() - start_time).total_seconds() * 1000),
                data_points_count=monthly_metrics.get('transaction_count', 0)
            )

        # 2. Compute category breakdown
        category_breakdown = compute_category_breakdown(db, user_id, current_period)
        if category_breakdown:
            metrics_service.store_metrics_sync(
                user_id=user_id,
                metric_type='category_breakdown',
                period=current_period,
                metric_value=category_breakdown,
                data_points_count=len(category_breakdown.get('categories', []))
            )

        # 3. Compute MoM comparison
        mom_comparison = compute_mom_comparison(db, user_id, current_period, previous_period)
        if mom_comparison:
            metrics_service.store_metrics_sync(
                user_id=user_id,
                metric_type='mom_comparison',
                period=current_period,
                metric_value=mom_comparison
            )

        # 4. Compute 6-month trends
        trends_6m = compute_6_month_trends(db, user_id)
        if trends_6m:
            metrics_service.store_metrics_sync(
                user_id=user_id,
                metric_type='trend_6m',
                period=current_period,
                metric_value=trends_6m,
                data_points_count=len(trends_6m.get('months', []))
            )

        computation_time_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        logger.info(f"Metrics computed for user_id={user_id} in {computation_time_ms}ms")

        return {
            'user_id': user_id,
            'computation_time_ms': computation_time_ms,
            'metrics_computed': ['monthly_total', 'category_breakdown', 'mom_comparison', 'trend_6m']
        }

    except Exception as e:
        logger.error(f"Failed to compute metrics for user_id={user_id}: {e}", exc_info=True)
        raise
    finally:
        db.close()


def compute_monthly_totals(db: Session, user_id: int, period: str) -> Dict[str, Any]:
    """
    Compute monthly totals for user

    This is a placeholder - in production, this would query search_service
    for actual transaction data.

    Args:
        db: Database session
        user_id: User ID
        period: Period string (YYYY-MM)

    Returns:
        Dict with monthly totals
    """
    # TODO: Integration with search_service
    # For now, return mock data structure
    logger.debug(f"Computing monthly totals for user_id={user_id}, period={period}")

    return {
        'period': period,
        'total_spending': 0.0,
        'total_income': 0.0,
        'net_balance': 0.0,
        'transaction_count': 0,
        'computed_at': datetime.utcnow().isoformat(),
        'status': 'mock_data'  # Will be replaced with real data
    }


def compute_category_breakdown(db: Session, user_id: int, period: str) -> Dict[str, Any]:
    """
    Compute top categories by spending

    Args:
        db: Database session
        user_id: User ID
        period: Period string (YYYY-MM)

    Returns:
        Dict with category breakdown
    """
    # TODO: Integration with search_service
    logger.debug(f"Computing category breakdown for user_id={user_id}, period={period}")

    return {
        'period': period,
        'categories': [],  # Will contain [{category, amount, percentage}, ...]
        'top_category': None,
        'computed_at': datetime.utcnow().isoformat(),
        'status': 'mock_data'
    }


def compute_mom_comparison(
    db: Session,
    user_id: int,
    current_period: str,
    previous_period: str
) -> Dict[str, Any]:
    """
    Compute month-over-month comparison

    Args:
        db: Database session
        user_id: User ID
        current_period: Current period (YYYY-MM)
        previous_period: Previous period (YYYY-MM)

    Returns:
        Dict with MoM comparison
    """
    # TODO: Use Analytics Agent compare_periods()
    logger.debug(f"Computing MoM comparison for user_id={user_id}")

    return {
        'current_period': current_period,
        'previous_period': previous_period,
        'spending_change_percent': 0.0,
        'income_change_percent': 0.0,
        'computed_at': datetime.utcnow().isoformat(),
        'status': 'mock_data'
    }


def compute_6_month_trends(db: Session, user_id: int) -> Dict[str, Any]:
    """
    Compute 6-month spending trends

    Args:
        db: Database session
        user_id: User ID

    Returns:
        Dict with 6-month trends
    """
    # TODO: Use Analytics Agent calculate_trend()
    logger.debug(f"Computing 6-month trends for user_id={user_id}")

    # Generate last 6 months
    months = []
    for i in range(6):
        month = (datetime.utcnow() - timedelta(days=30 * i)).strftime('%Y-%m')
        months.append(month)

    return {
        'months': list(reversed(months)),
        'spending_trend': [],  # Will contain monthly spending values
        'trend_direction': 'stable',  # 'increasing', 'decreasing', 'stable'
        'computed_at': datetime.utcnow().isoformat(),
        'status': 'mock_data'
    }


@celery_app.task(name='conversation_service.jobs.precompute_metrics.cleanup_expired_metrics')
def cleanup_expired_metrics():
    """
    Cleanup expired metrics from PostgreSQL

    Runs daily at 2 AM (before pre-compute job)

    Returns:
        Dict with cleanup statistics
    """
    logger.info("Starting cleanup of expired metrics")

    db = get_db_session()

    try:
        # Delete metrics that expired more than 7 days ago
        cutoff_date = datetime.utcnow() - timedelta(days=7)

        deleted_count = db.query(PreComputedMetric).filter(
            PreComputedMetric.expires_at < cutoff_date
        ).delete()

        db.commit()

        logger.info(f"Cleaned up {deleted_count} expired metrics")

        return {
            'deleted_count': deleted_count,
            'cutoff_date': cutoff_date.isoformat(),
            'completed_at': datetime.utcnow().isoformat()
        }

    except Exception as e:
        db.rollback()
        logger.error(f"Cleanup job failed: {e}", exc_info=True)
        raise
    finally:
        db.close()


__all__ = [
    'precompute_all_users',
    'precompute_user_metrics',
    'cleanup_expired_metrics'
]
