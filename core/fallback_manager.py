"""Fallback execution utilities.

This module provides a :class:`FallbackManager` that coordinates a
primary callable and a series of fallback callables.  The manager tries
each callable in order until one succeeds.  It supports both synchronous
and asynchronous callables which allows it to be used with the agents in
this project.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Sequence
import inspect
import logging

logger = logging.getLogger(__name__)

CallableType = Callable[..., Any]


async def _maybe_await(result: Any) -> Any:
    """Return the awaited result if *result* is awaitable.

    Parameters
    ----------
    result:
        The value returned by a callable which may be awaitable.
    """
    if inspect.isawaitable(result):
        return await result
    return result


class FallbackManager:
    """Execute callables with fallback strategies.

    The manager receives a sequence of callables.  When executed it tries
    each callable in order and returns the first successful result.  If
    all callables raise an exception the last exception is propagated.

    Examples
    --------
    >>> fm = FallbackManager([primary, fallback])
    >>> result = await fm.execute(args)
    """

    def __init__(self, handlers: Sequence[CallableType] | None = None) -> None:
        self.handlers: list[CallableType] = list(handlers or [])

    def add_handler(self, handler: CallableType) -> None:
        """Register an additional fallback handler."""
        self.handlers.append(handler)

    async def execute(self, args: Iterable[Any] | None = None, **kwargs: Any) -> Any:
        """Execute the registered handlers until one succeeds.

        Parameters
        ----------
        args:
            Positional arguments passed to the handlers.  If ``None`` an
            empty argument list is used.
        kwargs:
            Keyword arguments passed to the handlers.
        """
        positional = list(args or [])
        last_exc: Exception | None = None

        for handler in self.handlers:
            try:
                logger.debug("Executing handler %s", handler)
                result = handler(*positional, **kwargs)
                return await _maybe_await(result)
            except Exception as exc:  # pragma: no cover - logging path
                logger.warning("Handler %s failed: %s", handler, exc)
                last_exc = exc
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("No handlers configured")
