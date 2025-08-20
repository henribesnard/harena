"""Simple in-memory cache manager for core components.

The cache provides TTL based storage that can be shared between the
API layer and the team orchestrator.  It is intentionally lightweight and
thread-safe for the MVP requirements.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from threading import RLock
from typing import Any, Dict, Optional


class CacheManager:
    """In-memory cache with TTL support."""

    def __init__(self, ttl_seconds: int = 300) -> None:
        self._store: Dict[str, Any] = {}
        self._expiry: Dict[str, datetime] = {}
        self._lock = RLock()
        self.ttl = ttl_seconds

    def get(self, key: str) -> Optional[Any]:
        """Return value for *key* if not expired."""
        with self._lock:
            exp = self._expiry.get(key)
            if exp and exp > datetime.utcnow():
                return self._store.get(key)
            if exp:
                self._store.pop(key, None)
                self._expiry.pop(key, None)
        return None

    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        """Store *value* for *key* with optional TTL override."""
        ttl = ttl_seconds if ttl_seconds is not None else self.ttl
        with self._lock:
            self._store[key] = value
            self._expiry[key] = datetime.utcnow() + timedelta(seconds=ttl)

    def clear(self) -> None:
        """Clear all cached values."""
        with self._lock:
            self._store.clear()
            self._expiry.clear()

    def get_stats(self) -> Dict[str, int]:
        """Return basic cache statistics."""
        with self._lock:
            return {"items": len(self._store)}
