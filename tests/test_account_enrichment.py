import pytest

from enrichment_service.models import TransactionInput, BatchTransactionInput
from enrichment_service.core.processor import ElasticsearchTransactionProcessor


class DummyESClient:
    index_name = "test-index"

    async def document_exists(self, document_id):
        return False

    async def index_document(self, document_id, document):
        return True

    async def bulk_index_documents(self, documents, force_update=False):
        return {
            "indexed": len(documents),
            "errors": 0,
            "responses": [{"success": True} for _ in documents],
        }


class DummyAccountService:
    async def enrich_with_account_data(self, transaction):
        return {
            "account_name": "Main Account",
            "category_name": "Food",
            "merchant_name": "Starbucks",
        }


@pytest.mark.asyncio
async def test_process_transaction_enriches_account_info():
    processor = ElasticsearchTransactionProcessor(DummyESClient(), DummyAccountService())
    tx = TransactionInput(
        bridge_transaction_id=1,
        user_id=1,
        account_id=123,
        clean_description="Payment to Coffee",
        amount=-10.5,
        date="2024-01-01T00:00:00",
        currency_code="EUR",
    )
    result = await processor.process_single_transaction(tx, False)
    assert "Main Account" in result.searchable_text
    assert "Food" in result.searchable_text
    assert "Starbucks" in result.searchable_text


@pytest.mark.asyncio
async def test_process_batch_enriches_account_info():
    processor = ElasticsearchTransactionProcessor(DummyESClient(), DummyAccountService())
    batch = BatchTransactionInput(
        user_id=1,
        transactions=[
            TransactionInput(
                bridge_transaction_id=1,
                user_id=1,
                account_id=123,
                clean_description="Payment to Coffee",
                amount=-10.5,
                date="2024-01-01T00:00:00",
                currency_code="EUR",
            ),
            TransactionInput(
                bridge_transaction_id=2,
                user_id=1,
                account_id=123,
                clean_description="Groceries",
                amount=-20.0,
                date="2024-01-02T00:00:00",
                currency_code="EUR",
            ),
        ],
    )
    result = await processor.process_transactions_batch(batch, False)
    assert result.results
    searchable = result.results[0].searchable_text
    assert "Main Account" in searchable
    assert "Food" in searchable
    assert "Starbucks" in searchable
