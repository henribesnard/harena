from datetime import datetime

import pytest


@pytest.fixture
def sample_es_account_response():
    """Sample Elasticsearch document containing account metadata."""
    return {
        "transaction_id": 10,
        "account_id": 1,
        "account_name": "Main Account",
        "account_type": "checking",
        "account_balance": 1000.0,
        "account_currency": "EUR",
        "account_last_sync": datetime(2024, 1, 2).isoformat(),
    }

