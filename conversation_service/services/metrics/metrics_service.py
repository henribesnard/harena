"""
Metrics Service - Pre-Computed Metrics Retrieval with Redis Caching

This service manages pre-computed metrics with a multi-layer caching strategy:
1. Redis cache (24h TTL) - for fast retrieval
2. PostgreSQL (pre_computed_metrics table) - for historical data
3. On-demand computation - fallback if no cache available

Sprint 1.2 - T2.3
"""

from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
import redis
import json
import logging
from contextlib import contextmanager

from conversation_service.models.user_profile.entities import PreComputedMetric
from conversation_service.config.settings import settings

logger = logging.getLogger(__name__)


class MetricsService:
    """Service for retrieving and caching pre-computed metrics"""

    def __init__(self, db: Session, redis_client: Optional[redis.Redis] = None):
        """
        Initialize Metrics Service

        Args:
            db: SQLAlchemy database session
            redis_client: Optional Redis client (creates default if None)
        """
        self.db = db

        # Initialize Redis client
        if redis_client is None:
            try:
                redis_host = getattr(settings, 'REDIS_HOST', 'localhost')
                redis_port = getattr(settings, 'REDIS_PORT', 6379)
                redis_db = getattr(settings, 'REDIS_DB', 0)

                self.redis = redis.Redis(
                    host=redis_host,
                    port=redis_port,
                    db=redis_db,
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5
                )
                # Test connection
                self.redis.ping()
                self.redis_available = True
                logger.info("Redis connection established")
            except (redis.ConnectionError, redis.TimeoutError, AttributeError) as e:
                logger.warning(f"Redis not available, operating without cache: {e}")
                self.redis = None
                self.redis_available = False
        else:
            self.redis = redis_client
            self.redis_available = True

        # Statistics
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'db_hits': 0,
            'on_demand_computations': 0
        }

    async def get_user_metrics(
        self,
        user_id: int,
        metric_type: Optional[str] = None,
        period: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get pre-computed metrics for user with multi-layer fallback

        Retrieval order:
        1. Redis cache (fast)
        2. PostgreSQL (historical)
        3. On-demand computation (fallback)

        Args:
            user_id: User ID
            metric_type: Optional specific metric type ('monthly_total', 'category_breakdown', etc.)
            period: Optional period ('2025-01', etc.)

        Returns:
            Dict containing metrics
        """
        cache_key = self._build_cache_key(user_id, metric_type, period)

        # 1. Try Redis cache
        if self.redis_available:
            cached_metrics = await self._get_from_redis(cache_key)
            if cached_metrics:
                self.stats['cache_hits'] += 1
                logger.debug(f"Cache HIT for user_id={user_id}")
                return cached_metrics

        self.stats['cache_misses'] += 1

        # 2. Try PostgreSQL
        db_metrics = await self._get_from_db(user_id, metric_type, period)
        if db_metrics:
            self.stats['db_hits'] += 1
            # Warm up Redis cache
            if self.redis_available:
                await self._set_in_redis(cache_key, db_metrics, ttl=86400)  # 24h
            logger.debug(f"DB HIT for user_id={user_id}")
            return db_metrics

        # 3. On-demand computation (fallback)
        logger.warning(f"Cache MISS for user_id={user_id}, computing on-demand")
        self.stats['on_demand_computations'] += 1

        computed_metrics = await self._compute_on_demand(user_id, metric_type, period)

        # Cache result (shorter TTL for on-demand)
        if self.redis_available and computed_metrics:
            await self._set_in_redis(cache_key, computed_metrics, ttl=3600)  # 1h

        return computed_metrics

    async def store_metrics(
        self,
        user_id: int,
        metric_type: str,
        period: str,
        metric_value: Dict[str, Any],
        computation_time_ms: Optional[int] = None,
        data_points_count: Optional[int] = None,
        ttl: int = 86400
    ) -> None:
        """
        Store pre-computed metrics in both Redis and PostgreSQL (async version)

        Args:
            user_id: User ID
            metric_type: Metric type
            period: Period string
            metric_value: Computed metric data
            computation_time_ms: Computation duration in ms
            data_points_count: Number of data points used
            ttl: Redis TTL in seconds (default 24h)
        """
        # Store in PostgreSQL
        await self._store_in_db(
            user_id=user_id,
            metric_type=metric_type,
            period=period,
            metric_value=metric_value,
            computation_time_ms=computation_time_ms,
            data_points_count=data_points_count
        )

        # Store in Redis
        if self.redis_available:
            cache_key = self._build_cache_key(user_id, metric_type, period)
            await self._set_in_redis(cache_key, metric_value, ttl=ttl)

        logger.info(f"Stored metrics for user_id={user_id}, type={metric_type}, period={period}")

    def store_metrics_sync(
        self,
        user_id: int,
        metric_type: str,
        period: str,
        metric_value: Dict[str, Any],
        computation_time_ms: Optional[int] = None,
        data_points_count: Optional[int] = None,
        ttl: int = 86400
    ) -> None:
        """
        Store pre-computed metrics in both Redis and PostgreSQL (sync version for Celery)

        Args:
            user_id: User ID
            metric_type: Metric type
            period: Period string
            metric_value: Computed metric data
            computation_time_ms: Computation duration in ms
            data_points_count: Number of data points used
            ttl: Redis TTL in seconds (default 24h)
        """
        # Store in PostgreSQL (sync)
        self._store_in_db_sync(
            user_id=user_id,
            metric_type=metric_type,
            period=period,
            metric_value=metric_value,
            computation_time_ms=computation_time_ms,
            data_points_count=data_points_count
        )

        # Store in Redis (sync)
        if self.redis_available:
            cache_key = self._build_cache_key(user_id, metric_type, period)
            self._set_in_redis_sync(cache_key, metric_value, ttl=ttl)

        logger.info(f"Stored metrics for user_id={user_id}, type={metric_type}, period={period}")

    async def invalidate_cache(self, user_id: int, metric_type: Optional[str] = None) -> None:
        """
        Invalidate Redis cache for user

        Args:
            user_id: User ID
            metric_type: Optional specific metric type to invalidate
        """
        if not self.redis_available:
            return

        if metric_type:
            # Invalidate specific metric type
            cache_key = self._build_cache_key(user_id, metric_type, None)
            pattern = f"{cache_key}*"
        else:
            # Invalidate all metrics for user
            pattern = f"metrics:user:{user_id}*"

        try:
            keys = self.redis.keys(pattern)
            if keys:
                self.redis.delete(*keys)
                logger.info(f"Invalidated {len(keys)} cache keys for user_id={user_id}")
        except Exception as e:
            logger.error(f"Error invalidating cache: {e}")

    async def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics

        Returns:
            Dict with cache hit/miss stats
        """
        total_requests = self.stats['cache_hits'] + self.stats['cache_misses']
        hit_rate = (self.stats['cache_hits'] / total_requests * 100) if total_requests > 0 else 0

        return {
            'cache_hits': self.stats['cache_hits'],
            'cache_misses': self.stats['cache_misses'],
            'db_hits': self.stats['db_hits'],
            'on_demand_computations': self.stats['on_demand_computations'],
            'cache_hit_rate_percent': round(hit_rate, 2),
            'redis_available': self.redis_available
        }

    def _build_cache_key(self, user_id: int, metric_type: Optional[str], period: Optional[str]) -> str:
        """Build Redis cache key"""
        key = f"metrics:user:{user_id}"
        if metric_type:
            key += f":{metric_type}"
        if period:
            key += f":{period}"
        return key

    async def _get_from_redis(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Retrieve metrics from Redis cache"""
        if not self.redis_available:
            return None

        try:
            cached = self.redis.get(cache_key)
            if cached:
                return json.loads(cached)
        except (redis.ConnectionError, json.JSONDecodeError) as e:
            logger.error(f"Error reading from Redis: {e}")

        return None

    async def _set_in_redis(self, cache_key: str, data: Dict[str, Any], ttl: int) -> None:
        """Store metrics in Redis cache (async)"""
        if not self.redis_available:
            return

        try:
            self.redis.setex(cache_key, ttl, json.dumps(data))
        except redis.ConnectionError as e:
            logger.error(f"Error writing to Redis: {e}")

    def _set_in_redis_sync(self, cache_key: str, data: Dict[str, Any], ttl: int) -> None:
        """Store metrics in Redis cache (sync)"""
        if not self.redis_available:
            return

        try:
            self.redis.setex(cache_key, ttl, json.dumps(data))
        except redis.ConnectionError as e:
            logger.error(f"Error writing to Redis: {e}")

    async def _get_from_db(
        self,
        user_id: int,
        metric_type: Optional[str],
        period: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """Retrieve metrics from PostgreSQL"""
        try:
            query = self.db.query(PreComputedMetric).filter(
                PreComputedMetric.user_id == user_id
            )

            if metric_type:
                query = query.filter(PreComputedMetric.metric_type == metric_type)

            if period:
                query = query.filter(PreComputedMetric.period == period)

            # Filter out expired metrics
            query = query.filter(
                (PreComputedMetric.expires_at.is_(None)) |
                (PreComputedMetric.expires_at > datetime.utcnow())
            )

            # Get most recent
            metric = query.order_by(PreComputedMetric.computed_at.desc()).first()

            if metric:
                return metric.metric_value

        except Exception as e:
            logger.error(f"Error reading from database: {e}")

        return None

    async def _store_in_db(
        self,
        user_id: int,
        metric_type: str,
        period: str,
        metric_value: Dict[str, Any],
        computation_time_ms: Optional[int],
        data_points_count: Optional[int]
    ) -> None:
        """Store metrics in PostgreSQL (async)"""
        try:
            # Check if metric already exists (update vs insert)
            existing = self.db.query(PreComputedMetric).filter(
                PreComputedMetric.user_id == user_id,
                PreComputedMetric.metric_type == metric_type,
                PreComputedMetric.period == period
            ).first()

            if existing:
                # Update existing
                existing.metric_value = metric_value
                existing.computed_at = datetime.utcnow()
                existing.expires_at = datetime.utcnow() + timedelta(days=30)  # 30 days expiry
                existing.computation_time_ms = computation_time_ms
                existing.data_points_count = data_points_count
            else:
                # Create new
                new_metric = PreComputedMetric(
                    user_id=user_id,
                    metric_type=metric_type,
                    period=period,
                    metric_value=metric_value,
                    computed_at=datetime.utcnow(),
                    expires_at=datetime.utcnow() + timedelta(days=30),
                    computation_time_ms=computation_time_ms,
                    data_points_count=data_points_count,
                    cache_hit=False
                )
                self.db.add(new_metric)

            self.db.commit()

        except Exception as e:
            self.db.rollback()
            logger.error(f"Error storing metrics in database: {e}")
            raise

    def _store_in_db_sync(
        self,
        user_id: int,
        metric_type: str,
        period: str,
        metric_value: Dict[str, Any],
        computation_time_ms: Optional[int],
        data_points_count: Optional[int]
    ) -> None:
        """Store metrics in PostgreSQL (sync)"""
        try:
            # Check if metric already exists (update vs insert)
            existing = self.db.query(PreComputedMetric).filter(
                PreComputedMetric.user_id == user_id,
                PreComputedMetric.metric_type == metric_type,
                PreComputedMetric.period == period
            ).first()

            if existing:
                # Update existing
                existing.metric_value = metric_value
                existing.computed_at = datetime.utcnow()
                existing.expires_at = datetime.utcnow() + timedelta(days=30)  # 30 days expiry
                existing.computation_time_ms = computation_time_ms
                existing.data_points_count = data_points_count
            else:
                # Create new
                new_metric = PreComputedMetric(
                    user_id=user_id,
                    metric_type=metric_type,
                    period=period,
                    metric_value=metric_value,
                    computed_at=datetime.utcnow(),
                    expires_at=datetime.utcnow() + timedelta(days=30),
                    computation_time_ms=computation_time_ms,
                    data_points_count=data_points_count,
                    cache_hit=False
                )
                self.db.add(new_metric)

            self.db.commit()

        except Exception as e:
            self.db.rollback()
            logger.error(f"Error storing metrics in database: {e}")
            raise

    async def _compute_on_demand(
        self,
        user_id: int,
        metric_type: Optional[str],
        period: Optional[str]
    ) -> Dict[str, Any]:
        """
        Compute metrics on-demand (fallback)

        This is a simplified version for emergency fallback.
        The batch job should normally pre-compute all metrics.

        Args:
            user_id: User ID
            metric_type: Metric type
            period: Period

        Returns:
            Basic computed metrics
        """
        # Return minimal metrics for fallback
        # In production, this would call search_service for real data
        logger.warning(f"On-demand computation for user_id={user_id} - limited data")

        return {
            'user_id': user_id,
            'metric_type': metric_type or 'fallback',
            'period': period or datetime.utcnow().strftime('%Y-%m'),
            'computed_at': datetime.utcnow().isoformat(),
            'is_fallback': True,
            'data': {
                'message': 'Metrics not yet computed. Batch job will compute full metrics.',
                'status': 'pending_computation'
            }
        }


__all__ = ['MetricsService']
