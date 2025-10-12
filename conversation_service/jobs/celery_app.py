"""
Celery Application Configuration

Celery app for background jobs including:
- Pre-computed metrics calculation (nightly at 3 AM)
- User profile analytics updates
- Batch data processing

Sprint 1.2 - T2.3
"""

from celery import Celery
from celery.schedules import crontab
import logging

from conversation_service.config.settings import settings

logger = logging.getLogger(__name__)

# Get Redis configuration
redis_host = getattr(settings, 'REDIS_HOST', 'localhost')
redis_port = getattr(settings, 'REDIS_PORT', 6379)
redis_db = getattr(settings, 'REDIS_DB', 0)

# Celery broker and backend URLs
broker_url = f'redis://{redis_host}:{redis_port}/{redis_db}'
result_backend = f'redis://{redis_host}:{redis_port}/{redis_db + 1}'  # Use separate DB for results

# Create Celery app
celery_app = Celery(
    'harena_jobs',
    broker=broker_url,
    backend=result_backend,
    include=['conversation_service.jobs.precompute_metrics']
)

# Celery configuration
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='Europe/Paris',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=600,  # 10 minutes max per task
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
)

# Beat schedule (periodic tasks)
celery_app.conf.beat_schedule = {
    'precompute-metrics-nightly': {
        'task': 'conversation_service.jobs.precompute_metrics.precompute_all_users',
        'schedule': crontab(hour=3, minute=0),  # 3 AM every day
        'options': {
            'expires': 3600,  # Task expires after 1 hour if not picked up
        }
    },
    'cleanup-expired-metrics': {
        'task': 'conversation_service.jobs.precompute_metrics.cleanup_expired_metrics',
        'schedule': crontab(hour=2, minute=0),  # 2 AM every day (before precompute)
        'options': {
            'expires': 3600,
        }
    },
}

logger.info("Celery app configured successfully")

__all__ = ['celery_app']
