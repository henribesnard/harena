import functools
import inspect
from typing import Any, Callable, TypeVar, Awaitable

T = TypeVar("T")

class BusinessError(Exception):
    """Erreur métier générique."""


def wrap_business_errors(func: Callable[..., T]) -> Callable[..., T]:
    """Décorateur pour envelopper les exceptions non gérées en BusinessError."""
    if inspect.iscoroutinefunction(func):
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                return await func(*args, **kwargs)
            except BusinessError:
                raise
            except Exception as exc:  # pragma: no cover - sécurité
                raise BusinessError(str(exc)) from exc
        return async_wrapper
    else:
        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                return func(*args, **kwargs)
            except BusinessError:
                raise
            except Exception as exc:  # pragma: no cover - sécurité
                raise BusinessError(str(exc)) from exc
        return sync_wrapper
