from fastapi import FastAPI
from fastapi.testclient import TestClient

from conversation_service.api.routes import router
from conversation_service.api.dependencies import (
    get_team_manager,
    get_conversation_manager,
    get_metrics_collector,
)


class StubTeamManager:
    async def health_check(self):
        return {
            "healthy": True,
            "details": {
                "agent_statuses": {},
                "last_health_check": "2025-01-01T00:00:00"
            }
        }


class StubConversationManager:
    async def get_stats(self):
        return {"active_conversations": 0, "total_turns": 0}


class StubMetricsCollector:
    def get_summary(self):
        return {"total_requests": 0, "avg_response_time": 0}


def create_test_app():
    app = FastAPI()
    app.include_router(router, prefix="/conversation")

    async def override_team_manager():
        return StubTeamManager()

    async def override_conversation_manager():
        return StubConversationManager()

    async def override_metrics_collector():
        return StubMetricsCollector()

    app.dependency_overrides[get_team_manager] = override_team_manager
    app.dependency_overrides[get_conversation_manager] = override_conversation_manager
    app.dependency_overrides[get_metrics_collector] = override_metrics_collector
    return app


def test_health_route_serializes_last_activity():
    app = create_test_app()
    client = TestClient(app)
    response = client.get("/conversation/health")
    assert response.status_code == 200
    data = response.json()
    last_activity = data["components"]["team_manager"]["last_activity"]
    assert last_activity == "2025-01-01T00:00:00"

