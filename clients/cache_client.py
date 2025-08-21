import time
import logging
from typing import Any, Optional, Dict, Tuple

logger = logging.getLogger(__name__)


class CacheClient:
    """Asynchronous in-memory cache with TTL support.

    The implementation is intentionally lightweight so that it can be used in
    tests and development environments without external dependencies.  Values
    are stored in a dictionary along with an optional expiration timestamp.
    """

    def __init__(self) -> None:
        self._store: Dict[str, Tuple[Optional[float], Any]] = {}

    async def get(self, key: str) -> Any:
        """Retrieve a value from the cache or return ``None`` if missing."""
        item = self._store.get(key)
        if not item:
            return None
        expires_at, value = item
        if expires_at is not None and expires_at < time.time():
            # Expired entry
            self._store.pop(key, None)
            logger.debug("Cache miss due to expiration", key=key)
            return None
        return value

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Store ``value`` in the cache with an optional TTL in seconds."""
        expires_at = time.time() + ttl if ttl is not None else None
        self._store[key] = (expires_at, value)
        logger.debug("Cache set", key=key, ttl=ttl)

    async def clear(self) -> None:
        """Remove all items from the cache."""
        self._store.clear()
