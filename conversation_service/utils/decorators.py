"""Decorators for tracing and correlation handling."""

from __future__ import annotations

import asyncio
import functools
from typing import Any, Callable, TypeVar

from .logging import (
    clear_correlation_id,
    get_logger,
    set_correlation_id,
)

F = TypeVar("F", bound=Callable[..., Any])


def traced(func: F) -> F:
    """Decorator that logs function execution and injects correlation IDs."""
    is_coro = asyncio.iscoroutinefunction(func)

    if is_coro:
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any):
            cid = set_correlation_id()
            logger = get_logger(func.__module__)
            logger.info(
                "start", function=func.__qualname__, correlation_id=cid
            )
            try:
                result = await func(*args, **kwargs)
                logger.info(
                    "success",
                    function=func.__qualname__,
                    correlation_id=cid,
                )
                return result
            except Exception:
                logger.exception(
                    "error", function=func.__qualname__, correlation_id=cid
                )
                raise
            finally:
                clear_correlation_id()
        return async_wrapper  # type: ignore[misc]

    @functools.wraps(func)
    def sync_wrapper(*args: Any, **kwargs: Any):
        cid = set_correlation_id()
        logger = get_logger(func.__module__)
        logger.info(
            "start", function=func.__qualname__, correlation_id=cid
        )
        try:
            result = func(*args, **kwargs)
            logger.info(
                "success",
                function=func.__qualname__,
                correlation_id=cid,
            )
            return result
        except Exception:
            logger.exception(
                "error", function=func.__qualname__, correlation_id=cid
            )
            raise
        finally:
            clear_correlation_id()

    return sync_wrapper  # type: ignore[misc]