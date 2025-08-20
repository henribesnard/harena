import types

from fastapi.testclient import TestClient

from conversation_service.main import create_app
from conversation_service.api.dependencies import (
    get_team_manager,
    get_conversation_manager,
    get_current_user,
    get_metrics_collector,
    get_conversation_service,
    get_db,
    validate_request_rate_limit,
)


class DummyTeamManager:
    async def process_user_message_with_metadata(self, user_message, user_id, conversation_id):
        return {
            "content": f"echo:{user_message}",
            "success": True,
            "confidence_score": 0.9,
            "metadata": {},
        }


class DummyConversationManager:
    def __init__(self):
        self.store = types.SimpleNamespace(save_context=lambda ctx: None)

    async def get_context(self, conversation_id, user_id):
        return types.SimpleNamespace(user_id=user_id, turns=[])

    async def add_turn(self, **kwargs):
        pass


class DummyConversationDBService:
    def get_or_create_conversation(self, user_id, conversation_id):
        return types.SimpleNamespace(conversation_id=conversation_id, user_id=user_id)

    def add_turn(self, **kwargs):
        pass


class DummyMetrics:
    def record_request(self, *args, **kwargs):
        pass

    def record_response_time(self, *args, **kwargs):
        pass

    def record_success(self, *args, **kwargs):
        pass

    def record_error(self, *args, **kwargs):
        pass


async def override_get_team_manager():
    return DummyTeamManager()


async def override_get_conversation_manager():
    return DummyConversationManager()


async def override_get_current_user():
    return {"user_id": 1, "permissions": ["chat:write"]}


def override_get_metrics_collector():
    return DummyMetrics()


def override_get_conversation_service():
    return DummyConversationDBService()


def override_get_db():
    return None


async def override_rate_limit():
    return None


app = create_app()
app.dependency_overrides[get_team_manager] = override_get_team_manager
app.dependency_overrides[get_conversation_manager] = override_get_conversation_manager
app.dependency_overrides[get_current_user] = override_get_current_user
app.dependency_overrides[get_metrics_collector] = override_get_metrics_collector
app.dependency_overrides[get_conversation_service] = override_get_conversation_service
app.dependency_overrides[get_db] = override_get_db
app.dependency_overrides[validate_request_rate_limit] = override_rate_limit

client = TestClient(app)


def test_chat_rest_flow():
    response = client.post(
        "/api/v1/chat", json={"message": "bonjour", "conversation_id": "c1"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "echo:bonjour"


def test_chat_websocket_flow():
    with client.websocket_connect("/api/v1/chat/ws") as websocket:
        websocket.send_text("bonjour")
        message = websocket.receive_text()
        assert message == "echo:bonjour"
