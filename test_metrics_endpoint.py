import asyncio

from conversation_service.api.routes import get_metrics
from conversation_service.utils.metrics import MetricsCollector


class DummyTeamManager:
    def get_team_performance(self):
        return {}


def test_metrics_endpoint_returns_json():
    metrics = MetricsCollector()
    user = {"permissions": ["view_metrics"]}
    result = asyncio.run(get_metrics(metrics=metrics, team_manager=DummyTeamManager(), user=user))
    assert "service_metrics" in result
    assert "system_info" in result
    assert "memory_usage" in result["system_info"]
    assert "cpu_usage" in result["system_info"]


if __name__ == "__main__":
    import pytest, sys

    sys.exit(pytest.main([__file__]))
