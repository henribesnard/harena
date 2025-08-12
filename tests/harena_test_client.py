import time
from typing import Any, Dict, Optional

from fastapi import FastAPI
from fastapi.testclient import TestClient

from conversation_service.api.routes import router
from conversation_service.api import dependencies
from conversation_service.models.conversation_models import ConversationRequest


class DummyTeamManager:
    async def process_user_message_with_metadata(
        self, user_message: str, user_id: int, conversation_id: str
    ) -> Dict[str, Any]:
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

    def get_conversations(self, user_id: int, limit: int = 10, offset: int = 0):
        class Conv:
            def __init__(self, cid: str):
                self.conversation_id = cid
                self.title = f"Conversation {cid}"
                self.status = "active"
                self.total_turns = 0
                self.last_activity_at = "2025-01-01T00:00:00"

        return [Conv("conv-1")]


def create_app(extra_overrides: Optional[Dict[Any, Any]] = None) -> FastAPI:
    app = FastAPI()
    app.include_router(router, prefix="/conversation")

    async def override_team_manager():
        return DummyTeamManager()

    async def override_conversation_manager():
        return DummyConversationManager()

    def override_metrics_collector():
        return DummyMetricsCollector()

    def override_conversation_service(db=None):
        return DummyConversationService()

    async def override_conversation_read_service(db=None):
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
        dependencies.get_conversation_read_service: override_conversation_read_service,
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


class HarenaTestClient:
    def __init__(self):
        self.client = TestClient(create_app())
        self.headers = {"Authorization": "Bearer test-token"}

    def test_conversation_chat(self):
        payload = {"message": "Bonjour", "conversation_id": "conv-1"}
        response = self.client.post("/conversation/chat", json=payload, headers=self.headers)
        assert response.status_code == 200

    def test_conversation_list(self):
        response = self.client.get("/conversation/conversations", headers=self.headers)
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)


def run_full_test():
    client = HarenaTestClient()
    client.test_conversation_chat()
    client.test_conversation_list()


def test_run_full():
    run_full_test()
