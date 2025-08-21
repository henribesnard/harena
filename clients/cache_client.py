import time
import logging
from typing import Any, Optional, Dict, Tuple

logger = logging.getLogger(__name__)


class CacheClient:
    """Asynchronous in-memory cache with TTL support.

    The implementation is intentionally lightweight so that it can be used in
    tests and development environments without external dependencies.  Values
    are stored in a dictionary along with an optional expiration timestamp.

    All cache keys are namespaced by ``user_id`` to ensure data isolation
    between different users.
    """

    def __init__(self) -> None:
        self._store: Dict[str, Tuple[Optional[float], Any]] = {}

    def _format_key(self, user_id: int, key: str) -> str:
        return f"{user_id}:{key}"

    async def get(self, user_id: int, key: str) -> Any:
        """Retrieve a value from the cache for ``user_id`` or return ``None`` if missing."""
        namespaced_key = self._format_key(user_id, key)
        item = self._store.get(namespaced_key)
        if not item:
            return None
        expires_at, value = item
        if expires_at is not None and expires_at < time.time():
            # Expired entry
            self._store.pop(namespaced_key, None)
            logger.debug("Cache miss due to expiration", key=namespaced_key)
            return None
        return value

    async def set(self, user_id: int, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Store ``value`` in the cache for ``user_id`` with an optional TTL in seconds."""
        namespaced_key = self._format_key(user_id, key)
        expires_at = time.time() + ttl if ttl is not None else None
        self._store[namespaced_key] = (expires_at, value)
        logger.debug("Cache set", key=namespaced_key, ttl=ttl)

    async def clear(self) -> None:
        """Remove all items from the cache."""
        self._store.clear()
