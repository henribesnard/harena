"""Structured logging utilities for the conversation service."""

from __future__ import annotations

import logging
import os
from typing import Optional

import structlog
from structlog.contextvars import bind_contextvars, clear_contextvars

from .helpers import generate_correlation_id


def _get_log_level() -> int:
    env = os.getenv("ENV", "development").lower()
    return {
        "production": logging.INFO,
        "development": logging.DEBUG,
        "test": logging.WARNING,
    }.get(env, logging.INFO)


def configure_logging() -> None:
    """Configure structlog with JSON output and context variables."""
    log_level = _get_log_level()
    logging.basicConfig(level=log_level, format="%(message)s")
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: Optional[str] = None) -> structlog.stdlib.BoundLogger:
    """Return a structlog logger."""
    return structlog.get_logger(name)


def set_correlation_id(correlation_id: Optional[str] = None) -> str:
    """Bind a correlation ID to the logging context."""
    if correlation_id is None:
        correlation_id = generate_correlation_id()
    bind_contextvars(correlation_id=correlation_id)
    return correlation_id


def clear_correlation_id() -> None:
    """Remove correlation data from the logging context."""
    clear_contextvars()
