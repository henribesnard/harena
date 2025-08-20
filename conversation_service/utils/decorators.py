import time
from functools import wraps
from typing import Callable, Awaitable

from fastapi import HTTPException, status

from .cache_client import cache_client


def rate_limit(calls: int, period: int) -> Callable[[Callable[..., Awaitable]], Callable[..., Awaitable]]:
    """Simple in-memory/redis rate limiter decorator.

    Args:
        calls: Maximum number of calls allowed within the period.
        period: Time window in seconds.
    """

    def decorator(func: Callable[..., Awaitable]):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            key = f"rate_limit:{func.__name__}"
            timestamps = await cache_client.get(key) or []
            now = time.time()
            timestamps = [ts for ts in timestamps if ts > now - period]
            if len(timestamps) >= calls:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded",
                )
            timestamps.append(now)
            await cache_client.set(key, timestamps, ttl=period)
            return await func(*args, **kwargs)

        return wrapper

    return decorator
