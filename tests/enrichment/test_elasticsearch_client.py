import asyncio
from datetime import datetime

from enrichment_service.storage.elasticsearch_client import ElasticsearchClient
from enrichment_service.models import StructuredTransaction


class DummyResponse:
    status = 200

    async def json(self):
        return {"items": [{"index": {"status": 201, "_id": "1"}}]}

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class DummySession:
    def post(self, *args, **kwargs):
        return DummyResponse()


def test_index_transactions_batch_smoke():
    client = ElasticsearchClient()
    client.session = DummySession()
    client._initialized = True

    tx = StructuredTransaction(
        transaction_id=1,
        user_id=1,
        account_id=1,
        searchable_text="test",
        primary_description="desc",
        amount=1.0,
        amount_abs=1.0,
        transaction_type="credit",
        currency_code="EUR",
        date=datetime.utcnow(),
        date_str="2024-01-01",
        month_year="2024-01",
        weekday="Monday",
        category_id=None,
        operation_type=None,
        is_future=False,
        is_deleted=False,
    )

    result = asyncio.run(client.index_transactions_batch([tx]))
    assert result["indexed"] == 1
    assert result["errors"] == 0
    assert result["total"] == 1


def test_index_transactions_batch_includes_account_metadata():
    client = ElasticsearchClient()
    client._initialized = True

    captured = {}

    async def fake_bulk_index_documents(docs, force_update: bool = False):
        captured["docs"] = docs
        return {"indexed": len(docs), "errors": 0, "total": len(docs), "responses": []}

    client.bulk_index_documents = fake_bulk_index_documents

    sync = datetime(2024, 1, 2)
    tx = StructuredTransaction(
        transaction_id=1,
        user_id=1,
        account_id=1,
        searchable_text="test",
        primary_description="desc",
        amount=1.0,
        amount_abs=1.0,
        transaction_type="credit",
        currency_code="EUR",
        date=datetime.utcnow(),
        date_str="2024-01-01",
        month_year="2024-01",
        weekday="Monday",
        category_id=None,
        operation_type=None,
        is_future=False,
        is_deleted=False,
        account_last_sync=sync,
        category_name="Food",
    )

    result = asyncio.run(client.index_transactions_batch([tx]))

    doc = captured["docs"][0]["document"]
    assert doc["account_last_sync"] == sync.isoformat()
    assert doc["category_name"] == "Food"
    assert result["indexed"] == 1
