"""Generic decorators used across the project."""

from __future__ import annotations

import functools
import logging
import time
from collections import deque
from typing import Any, Callable, Tuple, Type

Logger = logging.Logger


def retry(
    max_attempts: int = 3,
    exceptions: Tuple[Type[BaseException], ...] = (Exception,),
    delay: float = 0.0,
    backoff: float = 1.0,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Retry a function call in case of specified exceptions."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            sleep = delay
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions:
                    if attempt == max_attempts:
                        raise
                    if sleep:
                        time.sleep(sleep)
                        sleep *= backoff
        return wrapper

    return decorator


def timing(logger: Logger | None = None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Measure and log the execution time of the wrapped function."""

    logger = logger or logging.getLogger(__name__)

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                duration = time.perf_counter() - start
                logger.info("%s took %.4f seconds", func.__name__, duration)

        return wrapper

    return decorator


def rate_limit(calls: int, period: float) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Limit the number of calls to *calls* within *period* seconds."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        timestamps: deque[float] = deque()

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            now = time.time()
            while timestamps and timestamps[0] <= now - period:
                timestamps.popleft()
            if len(timestamps) >= calls:
                sleep_time = period - (now - timestamps[0])
                time.sleep(sleep_time)
            result = func(*args, **kwargs)
            timestamps.append(time.time())
            return result

        return wrapper

    return decorator
