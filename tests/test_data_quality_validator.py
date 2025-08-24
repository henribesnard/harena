import logging
from datetime import datetime

from enrichment_service.data_quality import DataQualityValidator
from enrichment_service.models import TransactionInput, StructuredTransaction


def test_validate_account_balance_consistency_pass():
    validator = DataQualityValidator(threshold=1.0)
    result = validator.validate_account_balance_consistency(100.0, [50.0, 30.0, 20.0])
    assert result["balance_check_passed"] is True
    assert result["quality_score"] == 1.0


def test_validate_account_balance_consistency_fail(caplog):
    validator = DataQualityValidator(threshold=1.0)
    with caplog.at_level(logging.WARNING):
        result = validator.validate_account_balance_consistency(90.0, [50.0, 30.0, 20.0])
        assert result["balance_check_passed"] is False
        assert result["quality_score"] < 1.0
        assert "Balance inconsistency detected" in caplog.text


def test_structured_transaction_includes_quality_fields():
    tx_input = TransactionInput(
        bridge_transaction_id=1,
        user_id=1,
        account_id=1,
        amount=50.0,
        date=datetime(2024, 1, 1),
        account_balance=100.0,
        recent_transactions=[50.0, 30.0, 20.0],
    )
    structured = StructuredTransaction.from_transaction_input(tx_input)
    doc = structured.to_elasticsearch_document()
    assert doc["balance_check_passed"] is True
    assert doc["quality_score"] == 1.0
