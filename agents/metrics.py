from collections import defaultdict
from typing import Dict, List


class MetricsCollector:
    """Simple in-memory metrics collector."""

    def __init__(self) -> None:
        self._counters: Dict[str, int] = defaultdict(int)
        self._timings: Dict[str, List[float]] = defaultdict(list)

    def increment(self, name: str, amount: int = 1) -> None:
        """Increment a counter metric."""
        self._counters[name] += amount

    def observe(self, name: str, value: float) -> None:
        """Record a timing or numeric observation."""
        self._timings[name].append(value)

    def export(self) -> Dict[str, Dict[str, float]]:
        """Export metrics as plain dictionaries."""
        return {
            "counters": dict(self._counters),
            "timings": {k: list(v) for k, v in self._timings.items()},
        }


# Global metrics collector instance used by all agents
metrics = MetricsCollector()
