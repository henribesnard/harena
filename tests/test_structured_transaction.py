from datetime import datetime
from enrichment_service.models import TransactionInput, StructuredTransaction


def test_structured_transaction_with_account_data():
    tx_input = TransactionInput(
        bridge_transaction_id=1,
        user_id=1,
        account_id=123,
        clean_description="Coffee",
        amount=-4.5,
        date=datetime(2024, 1, 1),
        currency_code="EUR",
        category_id=10,
        category_name="Food",
        account_name="Main Account",
        account_type="checking",
        account_balance=1000.0,
        account_currency="EUR",
        account_last_sync=datetime(2024, 1, 2),
    )

    structured = StructuredTransaction.from_transaction_input(tx_input)

    assert structured.account_name == "Main Account"
    assert structured.account_type == "checking"
    assert structured.account_balance == 1000.0
    assert structured.account_currency == "EUR"
    assert structured.account_last_sync == datetime(2024, 1, 2)
    assert structured.category_name == "Food"

    doc = structured.to_elasticsearch_document()
    assert doc["account_name"] == "Main Account"
    assert doc["account_type"] == "checking"
    assert doc["account_balance"] == 1000.0
    assert doc["account_currency"] == "EUR"
    assert doc["account_last_sync"] == datetime(2024, 1, 2).isoformat()
    assert doc["category_name"] == "Food"
