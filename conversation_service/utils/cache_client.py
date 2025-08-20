import logging
from typing import Any, Optional

from .cache import MultiLevelCache, LRUCache

logger = logging.getLogger(__name__)


class CacheClient:
    """Wrapper around cache implementations with graceful fallback."""

    def __init__(self) -> None:
        try:
            self._cache = MultiLevelCache()
            self._async = True
            logger.info("Using MultiLevelCache backend")
        except Exception as exc:  # pragma: no cover - fallback path
            logger.warning(
                "MultiLevelCache unavailable, falling back to in-memory LRUCache: %s",
                exc,
            )
            self._cache = LRUCache()
            self._async = False

    async def get(self, key: str) -> Optional[Any]:
        """Retrieve a value from cache.

        Always returns ``None`` on backend failure.
        """
        try:
            if self._async:
                return await self._cache.get(key)
            return self._cache.get(key)
        except Exception as exc:  # pragma: no cover - backend failure
            logger.error("Cache get failed: %s", exc)
            return None

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Store a value in cache.

        Any failure is logged and ignored.
        """
        try:
            if self._async:
                await self._cache.set(key, value, ttl)
            else:
                self._cache.set(key, value, ttl)
        except Exception as exc:  # pragma: no cover - backend failure
            logger.error("Cache set failed: %s", exc)


cache_client = CacheClient()
