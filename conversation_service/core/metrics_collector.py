"""Basic metrics collection utilities for Harena conversation service."""

from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Any, Dict, List

__all__ = ["MetricsCollector"]


class MetricsCollector:
    """Collects counters and timing information for agents."""

    def __init__(self) -> None:
        self.counters = defaultdict(int)
        self.timings: Dict[str, List[float]] = defaultdict(list)

    def increment(self, name: str, value: int = 1) -> None:
        """Increment a named counter."""
        self.counters[name] += value

    def record_timing(self, name: str, duration_ms: float) -> None:
        """Record execution time for an operation in milliseconds."""
        self.timings[name].append(duration_ms)

    def summary(self) -> Dict[str, Any]:
        """Return aggregated metrics suitable for logging or reporting."""
        return {
            "counters": dict(self.counters),
            "timings": {
                key: {
                    "count": len(values),
                    "avg_ms": mean(values) if values else 0.0,
                }
                for key, values in self.timings.items()
            },
        }
