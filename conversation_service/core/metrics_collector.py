"""Core level wrapper for metrics collection utilities.

This module exposes :class:`MetricsCollector` in the core package so that
high level components such as API endpoints and the team orchestrator depend
on a stable import path.
"""
from __future__ import annotations

from ..utils.metrics import MetricsCollector as _BaseMetricsCollector


class MetricsCollector(_BaseMetricsCollector):
    """Thin wrapper around the utility MetricsCollector."""

    pass
