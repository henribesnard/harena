from datetime import datetime

from fastapi import FastAPI
import sys
import types

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from db_service.base import Base
from db_service.models.user import User
from db_service.models.sync import SyncItem, SyncAccount, RawTransaction

# Stub the user_service dependency to avoid heavy imports
current_user = None
deps_stub = types.ModuleType("user_service.api.deps")

async def get_current_active_user():  # type: ignore
    return current_user

deps_stub.get_current_active_user = get_current_active_user
sys.modules["user_service.api.deps"] = deps_stub

from enrichment_service.api.routes import (
    router,
    get_db,
    get_elasticsearch_processor,
    get_current_active_user,
)
from enrichment_service.core.account_enrichment_service import AccountEnrichmentService
from enrichment_service.core.processor import ElasticsearchTransactionProcessor


class DummyElasticsearchClient:
    def __init__(self):
        self.documents = []
        self.account_documents = []
        self.default_batch_size = 500

    async def delete_user_transactions(self, user_id: int) -> int:
        self.documents = []
        return 0

    async def bulk_index_documents(self, docs, force_update: bool = False):
        self.documents.extend(docs)
        return {
            "indexed": len(docs),
            "errors": 0,
            "responses": [{"success": True} for _ in docs],
        }

    async def index_accounts(self, accounts, user_id: int):
        count = 0
        for acc in accounts:
            acc_id = getattr(acc, "id", getattr(acc, "account_id", None))
            if acc_id is None:
                continue
            doc = {
                "account_id": acc_id,
                "user_id": user_id,
                "account_name": getattr(acc, "account_name", None),
            }
            self.account_documents.append(doc)
            count += 1
        return count


def create_app_and_db():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    db = Session()

    user = User(id=1, email="test@example.com", password_hash="x")
    item = SyncItem(id=1, user_id=1, bridge_item_id=1)
    account = SyncAccount(
        id=1,
        item_id=1,
        bridge_account_id=123,
        account_name="Main Account",
        account_type="checking",
        balance=1000.0,
        currency_code="EUR",
        last_sync_timestamp=datetime(2024, 1, 2),
    )
    tx = RawTransaction(
        id=1,
        bridge_transaction_id=10,
        account_id=1,
        user_id=1,
        clean_description="Coffee",
        provider_description="Coffee shop",
        amount=-3.5,
        date=datetime(2024, 1, 3),
        currency_code="EUR",
    )
    db.add_all([user, item, account, tx])
    db.commit()

    app = FastAPI()
    app.include_router(router, prefix="/api/v1/enrichment")

    def override_get_db():
        try:
            yield db
        finally:
            pass

    es_client = DummyElasticsearchClient()
    account_service = AccountEnrichmentService(db)
    processor = ElasticsearchTransactionProcessor(es_client, account_service)

    def override_processor():
        return processor

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_elasticsearch_processor] = override_processor
    app.dependency_overrides[get_current_active_user] = lambda: current_user

    global current_user
    current_user = types.SimpleNamespace(id=1, is_superuser=True, is_active=True)

    return app, es_client, db


def test_sync_user_produces_account_metadata(sample_es_account_response):
    app, es_client, db = create_app_and_db()
    client = TestClient(app)

    response = client.post("/api/v1/enrichment/elasticsearch/sync-user/1")
    assert response.status_code == 200
    assert es_client.documents, "No documents indexed"

    doc = es_client.documents[0]["document"]
    for field, value in sample_es_account_response.items():
        assert doc[field] == value

    # Accounts should also be indexed separately
    assert len(es_client.account_documents) == 1
    account_doc = es_client.account_documents[0]
    assert account_doc["account_id"] == 1

    payload = response.json()
    assert payload["transactions_indexed"] == 1
    assert payload["accounts_indexed"] == 1

    db.close()

