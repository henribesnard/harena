"""Simple in-memory metrics collector for cache operations."""

from __future__ import annotations

from typing import Dict, List


class MetricsCollector:
    """Collect metrics about cache usage."""

    def __init__(self) -> None:
        self.hits: Dict[str, int] = {"l1": 0, "l2": 0}
        self.misses: Dict[str, int] = {"l1": 0, "l2": 0}
        self.latency: Dict[str, List[float]] = {"l1": [], "l2": []}
        self.size: Dict[str, int] = {"l1": 0, "l2": 0}

    # Recording helpers -------------------------------------------------
    def record_hit(self, level: str) -> None:
        self.hits[level] += 1

    def record_miss(self, level: str) -> None:
        self.misses[level] += 1

    def record_latency(self, level: str, seconds: float) -> None:
        self.latency[level].append(seconds)

    def record_size(self, level: str, size: int) -> None:
        self.size[level] = size

    # Query helpers -----------------------------------------------------
    def avg_latency(self, level: str) -> float:
        times = self.latency[level]
        return sum(times) / len(times) if times else 0.0
