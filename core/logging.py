"""Structured logging helpers used across services."""
from __future__ import annotations

import json
import logging
import sys
from typing import Any, Dict


class JsonFormatter(logging.Formatter):
    """Formatter that outputs logs as JSON strings."""

    def format(self, record: logging.LogRecord) -> str:  # pragma: no cover - trivial
        log_record: Dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        extra = getattr(record, "extra", None)
        if isinstance(extra, dict):
            log_record.update(extra)
        return json.dumps(log_record, ensure_ascii=False)


def get_logger(name: str, **context: Any) -> logging.Logger:
    """Return a logger with JSON output and optional context."""

    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    if context:
        class ContextAdapter(logging.LoggerAdapter):
            def process(self, msg, kwargs):  # pragma: no cover - passthrough
                extra = kwargs.setdefault("extra", {})
                extra.update(context)
                return msg, kwargs
        return ContextAdapter(logger, {})
    return logger


def setup_logging(level: str = "INFO") -> None:
    """Configure global logging level."""

    logging.getLogger().setLevel(getattr(logging, level.upper(), logging.INFO))


__all__ = ["get_logger", "setup_logging", "JsonFormatter"]
