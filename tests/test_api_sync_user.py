import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, AsyncMock
from types import SimpleNamespace
from datetime import datetime

from enrichment_service.api.routes import router, get_account_enrichment_service
from user_service.api.deps import get_current_active_user
from db_service.session import get_db
from db_service.models.user import User
from db_service.models.sync import RawTransaction, SyncAccount
from enrichment_service.core.processor import ElasticsearchTransactionProcessor
from enrichment_service.models import UserSyncResult
from db_service.models.sync import RawTransaction, SyncAccount

def create_test_app(processor_mock, db_mock, user):
    app = FastAPI()
    app.include_router(router, prefix="/api/v1/enrichment")

    app.dependency_overrides[get_current_active_user] = lambda: user
    app.dependency_overrides[get_db] = lambda: db_mock
    app.dependency_overrides[get_account_enrichment_service] = lambda: None

    import enrichment_service.api.routes as routes
    routes.elasticsearch_processor = processor_mock

    return app

def test_sync_user_endpoint_invokes_processor():
    dummy_user = User(id=1, email="test@example.com", password_hash="x")
    dummy_user.is_active = True
    dummy_user.is_superuser = True

    account = SimpleNamespace(
        id=123,
        account_id=123,
        account_name="Main",
        account_type="checking",
        balance=1000.0,
        currency_code="EUR",
        last_sync_timestamp=datetime(2024, 1, 2),
    )

    raw_tx = SimpleNamespace(
        bridge_transaction_id=1,
        user_id=1,
        account_id=123,
        clean_description="desc",
        provider_description=None,
        amount=10.0,
        date=datetime(2024, 1, 1),
        booking_date=None,
        transaction_date=None,
        value_date=None,
        currency_code="EUR",
        category_id=None,
        operation_type=None,
        deleted=False,
        future=False,
    )

    class DummyQuery:
        def __init__(self, results):
            self._results = results

        def filter(self, *args, **kwargs):
            return self

        def all(self):
            return self._results

    class DummyDB:
        def __init__(self, tx_results, account_results):
            self.tx_results = tx_results
            self.account_results = account_results

        def query(self, model):
            if model is RawTransaction:
                return DummyQuery(self.tx_results)
            if model is SyncAccount:
                return DummyQuery(self.account_results)
            return DummyQuery([])

    dummy_db = DummyDB([raw_tx], [account])

    processor_mock = MagicMock(spec=ElasticsearchTransactionProcessor)
    processor_mock.sync_user_transactions = AsyncMock(
        return_value=UserSyncResult(
            user_id=1,
            total_transactions=1,
            transactions_indexed=1,
            accounts_indexed=1,
            updated=0,
            errors=0,
            with_account_metadata=1,
            processing_time=0.0,
        )
    )

    app = create_test_app(processor_mock, dummy_db, dummy_user)
    client = TestClient(app)

    response = client.post("/api/v1/enrichment/elasticsearch/sync-user/1")
    assert response.status_code == 200
    processor_mock.sync_user_transactions.assert_awaited_once()
    kwargs = processor_mock.sync_user_transactions.await_args.kwargs
    assert kwargs["user_id"] == 1
    assert len(kwargs["transactions"]) == 1
    assert kwargs["accounts_map"][123].account_name == "Main"
    assert len(kwargs["accounts"]) == 1
    assert response.json()["with_account_metadata"] == 1


def test_sync_user_with_account_without_id_returns_200():
    dummy_user = User(id=1, email="test@example.com", password_hash="x")
    dummy_user.is_active = True
    dummy_user.is_superuser = True

    raw_tx = SimpleNamespace(
        bridge_transaction_id=1,
        user_id=1,
        account_id=123,
        clean_description="desc",
        provider_description=None,
        amount=10.0,
        date=datetime(2024, 1, 1),
        booking_date=None,
        transaction_date=None,
        value_date=None,
        currency_code="EUR",
        category_id=None,
        operation_type=None,
        deleted=False,
        future=False,
    )

    account = SimpleNamespace(
        account_id=123,
        account_name="test",
        account_type="checking",
        balance=0.0,
        currency_code="EUR",
        last_sync_timestamp=datetime(2024, 1, 1),
    )

    class DummyQuery:
        def __init__(self, results):
            self._results = results

        def filter(self, *args, **kwargs):
            return self

        def all(self):
            return self._results

    class DummyDB:
        def __init__(self, tx_results, account_results):
            self.tx_results = tx_results
            self.account_results = account_results

        def query(self, model):
            if model is RawTransaction:
                return DummyQuery(self.tx_results)
            if model is SyncAccount:
                return DummyQuery(self.account_results)
            return DummyQuery([])

    dummy_db = DummyDB([raw_tx], [account])

    processor_mock = MagicMock(spec=ElasticsearchTransactionProcessor)
    processor_mock.sync_user_transactions = AsyncMock(
        return_value=UserSyncResult(
            user_id=1,
            total_transactions=1,
            transactions_indexed=1,
            accounts_indexed=0,
            updated=0,
            errors=0,
            with_account_metadata=1,
            processing_time=0.0,
        )
    )

    app = create_test_app(processor_mock, dummy_db, dummy_user)
    client = TestClient(app)

    response = client.post("/api/v1/enrichment/elasticsearch/sync-user/1")
    assert response.status_code == 200
