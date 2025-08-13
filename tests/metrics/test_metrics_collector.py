import os
import sys

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from conversation_service.utils.metrics import MetricsCollector


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
