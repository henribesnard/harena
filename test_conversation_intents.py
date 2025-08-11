import pytest
from fastapi.testclient import TestClient

from conversation_service.main import create_app
from conversation_service.api.dependencies import get_current_user
from conversation_service.agents.search_query_agent import SearchQueryAgent
from conversation_service.models.service_contracts import (
    SearchServiceResponse,
    ResponseMetadata,
    TransactionResult,
)

# ---------------------------------------------------------------------
# Infrastructure commune
# ---------------------------------------------------------------------
app = create_app()
client = TestClient(app)


@pytest.fixture(autouse=True)
def mock_user():
    """Simule un utilisateur authentifié."""
    async def _mock_user():
        return {"user_id": 42, "username": "test", "email": "test@example.com"}

    app.dependency_overrides[get_current_user] = _mock_user
    yield
    app.dependency_overrides.pop(get_current_user, None)


# ---------------------------------------------------------------------
# 1. Intention sans recherche
# ---------------------------------------------------------------------
def test_greeting_intent():
    payload = {"conversation_id": "conv-1", "message": "Bonjour"}
    resp = client.post("/api/v1/conversation/chat", json=payload)

    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert "bonjour" in data["message"].lower()


# ---------------------------------------------------------------------
# 2. Intention avec recherche et cloisonnement par utilisateur
# ---------------------------------------------------------------------
def test_search_intent_scoped_to_user(monkeypatch):
    captured_user_id = {}

    async def fake_execute(self, search_query):
        """Capture le user_id transmis et renvoie une réponse factice."""
        captured_user_id["user_id"] = search_query.query_metadata.user_id
        return SearchServiceResponse(
            response_metadata=ResponseMetadata(
                query_id="q1",
                processing_time_ms=1.0,
                total_results=1,
                returned_results=1,
                has_more_results=False,
                search_strategy_used="mock",
            ),
            results=[
                TransactionResult(
                    transaction_id="tx1",
                    date="2024-01-01T00:00:00Z",
                    amount=-10.0,
                    currency="EUR",
                    description="NETFLIX",
                    merchant="NETFLIX",
                    category="subscription",
                    account_id="acc-42",
                    transaction_type="debit",
                )
            ],
        )

    monkeypatch.setattr(SearchQueryAgent, "_execute_search_query", fake_execute)

    payload = {"conversation_id": "conv-2", "message": "mes achats netflix"}
    resp = client.post("/api/v1/conversation/chat", json=payload)

    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    # Vérifie que l’ID utilisateur transmis au Search Service correspond à l’utilisateur authentifié.
    assert captured_user_id["user_id"] == 42
