"""Metrics collection utilities for the conversation service.

This module exposes a thin abstraction over ``prometheus_client`` to allow
both in-memory aggregation and Prometheus scraping.  It records metrics for
agent calls and orchestrator operations including error rates, processing
latencies and cache efficiency.
"""

from __future__ import annotations

from collections import defaultdict
from threading import Lock
from typing import Any, Dict

from prometheus_client import Counter, Histogram

__all__ = ["MetricsCollector", "metrics_collector"]


class MetricsCollector:
    """Collect and expose metrics for agents and orchestrator."""

    def __init__(self) -> None:
        # In-memory stats for quick health summaries
        self._lock = Lock()
        self._agent_stats: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {
                "calls": 0,
                "errors": 0,
                "cache_hits": 0,
                "cache_misses": 0,
                "total_time_ms": 0.0,
            }
        )
        self._orch_stats: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"calls": 0, "errors": 0, "total_time_ms": 0.0}
        )

        # Prometheus metrics
        self._agent_calls = Counter(
            "agent_calls_total",
            "Total agent calls",
            ["agent", "status", "cached"],
        )
        self._agent_duration = Histogram(
            "agent_duration_seconds",
            "Agent processing time in seconds",
            ["agent"],
        )
        self._orch_calls = Counter(
            "orchestrator_calls_total",
            "Total orchestrator operations",
            ["operation", "status"],
        )
        self._orch_duration = Histogram(
            "orchestrator_duration_seconds",
            "Orchestrator operation duration in seconds",
            ["operation"],
        )

    async def record_agent_call(
        self,
        *,
        agent_name: str,
        success: bool,
        processing_time_ms: int,
        tokens_used: int = 0,
        error: str | None = None,
        cached: bool = False,
    ) -> None:
        """Record metrics for a single agent call."""

        status = "success" if success else "error"
        cache_status = "hit" if cached else "miss"
        self._agent_calls.labels(agent=agent_name, status=status, cached=cache_status).inc()
        self._agent_duration.labels(agent=agent_name).observe(processing_time_ms / 1000.0)

        with self._lock:
            stats = self._agent_stats[agent_name]
            stats["calls"] += 1
            stats["total_time_ms"] += processing_time_ms
            if success:
                pass
            else:
                stats["errors"] += 1
            if cached:
                stats["cache_hits"] += 1
            else:
                stats["cache_misses"] += 1

    def record_orchestrator_call(
        self,
        *,
        operation: str,
        success: bool,
        processing_time_ms: float,
    ) -> None:
        """Record metrics for orchestrator operations."""

        status = "success" if success else "error"
        self._orch_calls.labels(operation=operation, status=status).inc()
        self._orch_duration.labels(operation=operation).observe(processing_time_ms / 1000.0)

        with self._lock:
            stats = self._orch_stats[operation]
            stats["calls"] += 1
            stats["total_time_ms"] += processing_time_ms
            if not success:
                stats["errors"] += 1

    def summary(self) -> Dict[str, Any]:
        """Return a simple JSON-serialisable snapshot of metrics."""

        with self._lock:
            return {
                "agents": {name: dict(data) for name, data in self._agent_stats.items()},
                "orchestrator": {name: dict(data) for name, data in self._orch_stats.items()},
            }


# Shared instance used across the service
metrics_collector = MetricsCollector()
