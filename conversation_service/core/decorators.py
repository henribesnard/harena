"""Common decorators for metrics collection and caching."""
from __future__ import annotations

import asyncio
import time
from functools import wraps
from typing import Any, Callable, Dict, Hashable, Tuple, TypeVar, Optional

F = TypeVar("F", bound=Callable[..., Any])


def metrics(metrics_collector: Any, name: str) -> Callable[[F], F]:
    """Decorator that records call count and execution time."""

    def decorator(func: F) -> F:
        if asyncio.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                start = time.time()
                try:
                    return await func(*args, **kwargs)
                finally:
                    if metrics_collector:
                        metrics_collector.record_request(name, 1)
                        elapsed = int((time.time() - start) * 1000)
                        metrics_collector.record_response_time(name, elapsed)

            return async_wrapper  # type: ignore[return-value]

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start = time.time()
            try:
                return func(*args, **kwargs)
            finally:
                if metrics_collector:
                    metrics_collector.record_request(name, 1)
                    elapsed = int((time.time() - start) * 1000)
                    metrics_collector.record_response_time(name, elapsed)

        return sync_wrapper  # type: ignore[return-value]

    return decorator


def cache(ttl: int = 60) -> Callable[[F], F]:
    """Simple in-memory TTL cache decorator."""

    def decorator(func: F) -> F:
        store: Dict[Hashable, Tuple[float, Any]] = {}

        if asyncio.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                key = (args, frozenset(kwargs.items()))
                now = time.time()
                if key in store:
                    exp, value = store[key]
                    if now - exp < ttl:
                        return value
                value = await func(*args, **kwargs)
                store[key] = (now, value)
                return value

            return async_wrapper  # type: ignore[return-value]

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            key = (args, frozenset(kwargs.items()))
            now = time.time()
            if key in store:
                exp, value = store[key]
                if now - exp < ttl:
                    return value
            value = func(*args, **kwargs)
            store[key] = (now, value)
            return value

        return sync_wrapper  # type: ignore[return-value]

    return decorator


__all__ = ["metrics", "cache"]
