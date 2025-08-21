import time
import hashlib
from typing import Any, Optional, Dict, Tuple


class MultiLevelCache:
    """Simple in-memory cache with optional TTL support.

    This minimal implementation provides the asynchronous interface expected by
    ``SearchEngine`` without relying on ``conversation_service``.  It stores
    values in a dictionary along with an optional expiration timestamp and is
    namespaced by ``user_id`` to ensure isolation between users.
    """

    def __init__(self) -> None:
        self._store: Dict[str, Tuple[Optional[float], Any]] = {}

    def _format_key(self, user_id: int, key: str) -> str:
        return f"{user_id}:{key}"

    async def get(self, user_id: int, key: str) -> Any:
        """Retrieve a value from the cache for ``user_id``.

        Returns ``None`` if the key is not present or has expired."""
        namespaced_key = self._format_key(user_id, key)
        item = self._store.get(namespaced_key)
        if not item:
            return None
        expires_at, value = item
        if expires_at is not None and expires_at < time.time():
            # Entry expired; remove it and behave as a miss
            self._store.pop(namespaced_key, None)
            return None
        return value

    async def set(self, user_id: int, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Store a value for ``user_id`` with an optional TTL in seconds."""
        namespaced_key = self._format_key(user_id, key)
        expires_at = time.time() + ttl if ttl is not None else None
        self._store[namespaced_key] = (expires_at, value)

    async def clear(self) -> None:
        """Clear all items from the cache."""
        self._store.clear()


def generate_cache_key(prefix: str, **parts: Any) -> str:
    """Generate a deterministic cache key from the provided parts.

    The parts are serialised in a stable order and hashed to avoid overly long
    keys."""
    raw = "|".join(f"{k}:{parts[k]}" for k in sorted(parts))
    digest = hashlib.sha256(raw.encode()).hexdigest()
    return f"{prefix}:{digest}"
