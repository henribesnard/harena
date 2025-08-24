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
from datetime import datetime
from enrichment_service.core.data_quality_validator import (
    DataQualityValidator as CoreDataQualityValidator,
)
from enrichment_service.models import TransactionInput


def make_tx(**kwargs):
    data = {
        "bridge_transaction_id": 1,
        "user_id": 1,
        "account_id": 1,
        "amount": 100.0,
        "date": datetime.now(),
        "currency_code": "EUR",
    }
    data.update(kwargs)
    return TransactionInput(**data)


def test_detect_amount_anomaly():
    validator = CoreDataQualityValidator(amount_threshold=1000)
    tx = make_tx(amount=5000)
    assert validator.detect_amount_anomalies(tx)
    is_valid, score, flags = validator.evaluate(tx)
    assert not is_valid
    assert "amount_anomaly" in flags
    assert score < 1


def test_missing_currency_code_inconsistency():
    validator = CoreDataQualityValidator()
    tx = make_tx(currency_code=None)
    assert not validator.validate_transaction_consistency(tx)
    is_valid, score, flags = validator.evaluate(tx)
    assert not is_valid
    assert "inconsistent_transaction" in flags


def test_account_balance_inconsistency():
    validator = CoreDataQualityValidator()
    tx = make_tx(amount=-150)
    assert not validator.validate_account_balance_consistency(tx, account_balance=100)
    is_valid, score, flags = validator.evaluate(tx, account_balance=100)
    assert not is_valid
    assert "account_balance_mismatch" in flags
