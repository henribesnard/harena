import time
from typing import Any, Dict, Optional

import pytest
from fastapi import FastAPI, HTTPException, status
from fastapi.testclient import TestClient

from conversation_service.api.routes import router
from conversation_service.api import dependencies
from conversation_service.models.conversation_models import ConversationRequest


class DummyTeamManager:
    async def process_user_message_with_metadata(self, user_message: str, user_id: int, conversation_id: str) -> Dict[str, Any]:
        return {
            "success": True,
            "content": "Bonjour, comment puis-je vous aider ?",
            "metadata": {},
            "confidence_score": 0.9,
        }

    async def health_check(self) -> Dict[str, Any]:
        return {
            "healthy": True,
            "details": {"agent_statuses": {}, "last_health_check": "2025-01-01"},
        }

    async def get_team_performance(self) -> Dict[str, Any]:
        return {}


class DummyConversationManager:
    async def get_context(self, conversation_id: str, user_id: int):
        class DummyStore:
            async def save_context(self, context):
                pass

        class Context:
            def __init__(self, uid: int):
                self.user_id = uid
                self.turns = []
                self.store = DummyStore()

        return Context(user_id)

    async def add_turn(self, *args, **kwargs):
        pass

    async def get_stats(self) -> Dict[str, int]:
        return {"active_conversations": 0, "total_turns": 0}


class DummyMetricsCollector:
    def __init__(self) -> None:
        self.start_time = time.time()

    def record_request(self, *args, **kwargs):
        pass

    def record_response_time(self, *args, **kwargs):
        pass

    def record_success(self, *args, **kwargs):
        pass

    def record_error(self, *args, **kwargs):
        pass

    def get_summary(self) -> Dict[str, int]:
        return {"total_requests": 1, "avg_response_time": 0}

    def get_memory_usage(self) -> int:
        return 0

    def get_cpu_usage(self) -> int:
        return 0


class DummyConversationService:
    def get_or_create_conversation(self, user_id: int, conversation_id: Optional[str]):
        class Conv:
            def __init__(self, cid: str, uid: int):
                self.conversation_id = cid or "conv-1"
                self.user_id = uid

        return Conv(conversation_id, user_id)

    def add_turn(self, **kwargs):
        pass


def create_app(extra_overrides: Optional[Dict[Any, Any]] = None) -> FastAPI:
    app = FastAPI()
    app.include_router(router)

    async def override_team_manager():
        return DummyTeamManager()

    async def override_conversation_manager():
        return DummyConversationManager()

    def override_metrics_collector():
        return DummyMetricsCollector()

    def override_conversation_service(db=None):
        return DummyConversationService()

    def override_validate_request(req: ConversationRequest) -> ConversationRequest:
        return req

    async def override_rate_limit(*args, **kwargs):
        pass

    async def override_current_user():
        return {"user_id": 1, "permissions": ["chat:write"], "rate_limit_tier": "standard"}

    def override_get_db():
        yield None

    overrides = {
        dependencies.get_team_manager: override_team_manager,
        dependencies.get_conversation_manager: override_conversation_manager,
        dependencies.get_metrics_collector: override_metrics_collector,
        dependencies.get_conversation_service: override_conversation_service,
        dependencies.validate_conversation_request: override_validate_request,
        dependencies.validate_request_rate_limit: override_rate_limit,
        dependencies.get_current_user: override_current_user,
        dependencies.get_db: override_get_db,
    }

    if extra_overrides:
        overrides.update(extra_overrides)

    for dep, override in overrides.items():
        app.dependency_overrides[dep] = override

    return app


def create_client(extra_overrides: Optional[Dict[Any, Any]] = None) -> TestClient:
    app = create_app(extra_overrides)
    return TestClient(app)


def test_health_endpoint():
    client = create_client()
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_status_endpoint():
    client = create_client()
    response = client.get("/status")
    assert response.status_code == 200
    assert response.json()["status"] == "running"


def test_metrics_endpoint():
    client = create_client()
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "service_metrics" in data


def test_chat_success():
    client = create_client()
    payload = {"message": "Bonjour", "conversation_id": "conv-1"}
    response = client.post("/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Bonjour, comment puis-je vous aider ?"
    assert data["conversation_id"] == "conv-1"


def test_chat_unauthorized():
    async def unauthorized_user():
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")

    client = create_client({dependencies.get_current_user: unauthorized_user})
    payload = {"message": "Bonjour", "conversation_id": "conv-1"}
    response = client.post("/chat", json=payload)
    assert response.status_code == status.HTTP_401_UNAUTHORIZED


def test_chat_service_unavailable():
    async def unavailable_team_manager():
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Service unavailable")

    client = create_client({dependencies.get_team_manager: unavailable_team_manager})
    payload = {"message": "Bonjour", "conversation_id": "conv-1"}
    response = client.post("/chat", json=payload)
    assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
