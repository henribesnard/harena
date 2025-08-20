"""JSON logging configuration with correlation IDs."""

from __future__ import annotations

import logging
import contextvars
from typing import Optional

from pythonjsonlogger import jsonlogger

from .helpers import generate_uuid

correlation_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "correlation_id", default=None
)


class CorrelationIdFilter(logging.Filter):
    """Inject the correlation ID into log records."""

    def filter(self, record: logging.LogRecord) -> bool:  # type: ignore[override]
        record.correlation_id = correlation_id_var.get()
        return True


def set_correlation_id(correlation_id: Optional[str] = None) -> str:
    """Set the correlation ID for the current context."""
    if correlation_id is None:
        correlation_id = generate_uuid()
    correlation_id_var.set(correlation_id)
    return correlation_id


def get_correlation_id() -> Optional[str]:
    """Return the current correlation ID."""
    return correlation_id_var.get()


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logger to emit JSON logs including correlation IDs."""
    handler = logging.StreamHandler()
    formatter = jsonlogger.JsonFormatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s %(correlation_id)s"
    )
    handler.setFormatter(formatter)
    handler.addFilter(CorrelationIdFilter())

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers = [handler]
