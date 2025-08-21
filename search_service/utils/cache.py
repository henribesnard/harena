import time
import hashlib
from typing import Any, Optional, Dict, Tuple


class MultiLevelCache:
    """Simple in-memory cache with optional TTL support.

    This minimal implementation provides the asynchronous interface expected by
    ``SearchEngine`` without relying on ``conversation_service``.  It stores
    values in a dictionary along with an optional expiration timestamp.
    """

    def __init__(self) -> None:
        self._store: Dict[str, Tuple[Optional[float], Any]] = {}

    async def get(self, key: str) -> Any:
        """Retrieve a value from the cache.

        Returns ``None`` if the key is not present or has expired."""
        item = self._store.get(key)
        if not item:
            return None
        expires_at, value = item
        if expires_at is not None and expires_at < time.time():
            # Entry expired; remove it and behave as a miss
            self._store.pop(key, None)
            return None
        return value

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Store a value in the cache with an optional time-to-live (seconds)."""
        expires_at = time.time() + ttl if ttl is not None else None
        self._store[key] = (expires_at, value)

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
