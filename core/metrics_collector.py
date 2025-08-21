from collections import defaultdict
from threading import Lock
from typing import Dict, Tuple

class MetricsCollector:
    """Collecte simple de métriques en mémoire."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._counts: Dict[Tuple[str, str, int], int] = defaultdict(int)
        self._times: Dict[Tuple[str, str, int], float] = defaultdict(float)

    def record_request(self, *, path: str, method: str, status_code: int, process_time: float) -> None:
        key = (path, method, status_code)
        with self._lock:
            self._counts[key] += 1
            self._times[key] += process_time

    def get_metrics(self) -> Dict[str, Dict[str, float]]:
        with self._lock:
            result: Dict[str, Dict[str, float]] = {}
            for key, count in self._counts.items():
                path, method, status = key
                total_time = self._times[key]
                result[f"{method} {path} {status}"] = {
                    "count": count,
                    "avg_time": total_time / count if count else 0.0,
                }
            return result

metrics_collector = MetricsCollector()
