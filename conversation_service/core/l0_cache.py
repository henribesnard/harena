"""In-memory L0 cache for precomputed responses.

Entries stored here are intended to be static or rarely changing answers that
should bypass the regular Redis/LRU layers. Keys are automatically prefixed
with ``l0:`` to avoid collisions with the other cache levels.

Warm-up strategy
----------------
Populate this cache during application start-up with known responses by calling
:func:`warmup`. Warm-up data can come from configuration files or any other
precomputation step.

Invalidation strategy
---------------------
Use :func:`invalidate` to remove specific keys when their underlying data
changes or call it without arguments to clear the entire store.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

__all__ = ["warmup", "get", "invalidate"]

_STORE: Dict[str, Any] = {}


def _l0_key(key: str) -> str:
    """Return the L0-prefixed version of ``key``."""
    return f"l0:{key}"


def warmup(entries: Dict[str, Any]) -> None:
    """Preload ``entries`` into the L0 cache.

    Keys are provided without the ``l0:`` prefix and will be composed exactly as
    expected by :class:`CacheManager` before being prefixed.
    """

    for key, value in entries.items():
        _STORE[_l0_key(key)] = value


def get(key: str) -> Optional[Any]:
    """Retrieve ``key`` from the L0 cache if present."""
    return _STORE.get(_l0_key(key))


def invalidate(keys: Optional[Iterable[str]] = None) -> None:
    """Remove ``keys`` from the L0 cache or clear all if ``None``.

    Passing ``None`` removes every entry, providing a coarse-grained invalidation
    mechanism when data changes globally.
    """

    if keys is None:
        _STORE.clear()
        return
    for key in keys:
        _STORE.pop(_l0_key(key), None)
