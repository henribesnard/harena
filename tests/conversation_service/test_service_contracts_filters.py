import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "service_contracts", Path(__file__).resolve().parents[2] / "conversation_service/models/service_contracts.py"
)
service_contracts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(service_contracts)
SearchFilters = service_contracts.SearchFilters


def test_date_filter_valid():
    filters = SearchFilters(date={"gte": "2024-01-01", "lte": "2024-01-31"})
    assert filters.date == {"gte": "2024-01-01", "lte": "2024-01-31"}


def test_date_filter_missing_keys():
    with pytest.raises(ValueError):
        SearchFilters.validate_date({"gte": "2024-01-01"})
    with pytest.raises(ValueError):
        SearchFilters.validate_date({"lte": "2024-01-31"})


def test_date_filter_inverted_range():
    with pytest.raises(ValueError):
        SearchFilters.validate_date({"gte": "2024-02-01", "lte": "2024-01-01"})


def test_amount_filter_valid_range():
    filters = SearchFilters(amount={"gte": 10.0, "lte": 100.0})
    assert filters.amount == {"gte": 10.0, "lte": 100.0}


def test_amount_filter_valid_gte_only():
    filters = SearchFilters(amount={"gte": 10.0})
    assert filters.amount == {"gte": 10.0}


def test_amount_filter_valid_lte_only():
    filters = SearchFilters(amount={"lte": 100.0})
    assert filters.amount == {"lte": 100.0}


def test_amount_filter_missing_keys():
    with pytest.raises(ValueError):
        SearchFilters.validate_amount({})
    with pytest.raises(ValueError):
        SearchFilters.validate_amount({"foo": 1.0})


def test_amount_filter_inverted_range():
    with pytest.raises(ValueError):
        SearchFilters.validate_amount({"gte": 100.0, "lte": 10.0})
