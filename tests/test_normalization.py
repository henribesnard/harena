import pytest

from conversation_service.utils.normalization import (
    normalize_amount,
    normalize_date,
    normalize_merchant,
    normalize_transaction,
)


def test_normalize_date():
    assert normalize_date("01/02/2024") == "2024-02-01"
    assert normalize_date("2024-02-01") == "2024-02-01"
    assert normalize_date("date invalide") is None


def test_normalize_amount():
    assert normalize_amount("1 234,56 €") == pytest.approx(1234.56)
    assert normalize_amount("$1,234.56") == pytest.approx(1234.56)
    assert normalize_amount("montant") is None


def test_normalize_merchant():
    assert normalize_merchant("  café-Paris  ") == "CAFE PARIS"
    assert normalize_merchant("") is None


def test_normalize_transaction():
    data = {"date": "01/02/2024", "amount": "1 234,56 €", "merchant": "café-Paris"}
    normalized = normalize_transaction(data)
    assert normalized["date"] == "2024-02-01"
    assert normalized["amount"] == pytest.approx(1234.56)
    assert normalized["merchant"] == "CAFE PARIS"
