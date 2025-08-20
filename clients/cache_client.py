import logging
from typing import Any, Optional

import redis.asyncio as redis

from conversation_service.utils.metrics import get_default_metrics_collector

logger = logging.getLogger(__name__)


class CacheClient:
    """Simple Redis cache client with metrics and logging."""

    def __init__(
        self,
        url: str = "redis://localhost:6379/0",
        redis_client: Optional[redis.Redis] = None,
    ) -> None:
        self.redis: redis.Redis = redis_client or redis.from_url(url)
        self.metrics = get_default_metrics_collector()

    async def get(self, key: str) -> Optional[Any]:
        timer_id = self.metrics.performance_monitor.start_timer("redis_get")
        self.metrics.record_request("redis_get", 0)
        try:
            value = await self.redis.get(key)
            self.metrics.record_success("redis_get")
            logger.debug("Redis GET %s -> %s", key, value)
            return value
        except Exception as e:
            self.metrics.record_error("redis_get", str(e))
            logger.exception("Redis GET failed for %s", key)
            raise
        finally:
            duration_ms = self.metrics.performance_monitor.end_timer(timer_id)
            self.metrics.record_response_time("redis_get", duration_ms)

    async def set(self, key: str, value: Any, ex: Optional[int] = None) -> bool:
        timer_id = self.metrics.performance_monitor.start_timer("redis_set")
        self.metrics.record_request("redis_set", 0)
        try:
            result = await self.redis.set(key, value, ex=ex)
            self.metrics.record_success("redis_set")
            logger.debug("Redis SET %s", key)
            return bool(result)
        except Exception as e:
            self.metrics.record_error("redis_set", str(e))
            logger.exception("Redis SET failed for %s", key)
            raise
        finally:
            duration_ms = self.metrics.performance_monitor.end_timer(timer_id)
            self.metrics.record_response_time("redis_set", duration_ms)

    async def delete(self, key: str) -> None:
        timer_id = self.metrics.performance_monitor.start_timer("redis_delete")
        self.metrics.record_request("redis_delete", 0)
        try:
            await self.redis.delete(key)
            self.metrics.record_success("redis_delete")
            logger.debug("Redis DELETE %s", key)
        except Exception as e:
            self.metrics.record_error("redis_delete", str(e))
            logger.exception("Redis DELETE failed for %s", key)
            raise
        finally:
            duration_ms = self.metrics.performance_monitor.end_timer(timer_id)
            self.metrics.record_response_time("redis_delete", duration_ms)

    async def close(self) -> None:
        await self.redis.close()
