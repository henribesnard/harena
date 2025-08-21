import pytest


class MetricsCollector:
    """Minimal metrics collector for tests."""

    def __init__(self) -> None:
        self.total_requests = 0
        self.total_response_time = 0.0

    def record_request(self, endpoint: str, count: int = 1) -> None:
        self.total_requests += count

    def record_response_time(self, endpoint: str, ms: float) -> None:
        self.total_response_time += ms

    def get_summary(self) -> dict:
        avg = self.total_response_time / self.total_requests if self.total_requests else 0.0
        return {"total_requests": self.total_requests, "avg_response_time": avg}


def test_metrics_counters_update_after_response():
    metrics = MetricsCollector()

    metrics.record_request("chat", 1)
    metrics.record_response_time("chat", 100)
    summary = metrics.get_summary()
    assert summary["total_requests"] == 1
    assert summary["avg_response_time"] == pytest.approx(100.0)

    metrics.record_request("chat", 1)
    metrics.record_response_time("chat", 200)
    summary = metrics.get_summary()
    assert summary["total_requests"] == 2
    assert summary["avg_response_time"] == pytest.approx(150.0)
