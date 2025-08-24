from datetime import datetime
from enrichment_service.core.data_quality_validator import DataQualityValidator
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
    validator = DataQualityValidator(amount_threshold=1000)
    tx = make_tx(amount=5000)
    assert validator.detect_amount_anomalies(tx)
    is_valid, score, flags = validator.evaluate(tx)
    assert not is_valid
    assert "amount_anomaly" in flags
    assert score < 1


def test_missing_currency_code_inconsistency():
    validator = DataQualityValidator()
    tx = make_tx(currency_code=None)
    assert not validator.validate_transaction_consistency(tx)
    is_valid, score, flags = validator.evaluate(tx)
    assert not is_valid
    assert "inconsistent_transaction" in flags


def test_account_balance_inconsistency():
    validator = DataQualityValidator()
    tx = make_tx(amount=-150)
    assert not validator.validate_account_balance_consistency(tx, account_balance=100)
    is_valid, score, flags = validator.evaluate(tx, account_balance=100)
    assert not is_valid
    assert "account_balance_mismatch" in flags
