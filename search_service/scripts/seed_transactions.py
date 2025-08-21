#!/usr/bin/env python
"""Seed sample transactions into the `harena_transactions` index.

This script populates the Elasticsearch index with a handful of debit and
credit transactions spread across several months of 2025. It is intended for
local development and testing so that search queries return meaningful results
out of the box.
"""

import asyncio
from datetime import datetime
import os

from config_service.config import settings

# Ensure an Elasticsearch endpoint is configured for local runs
os.environ.setdefault("BONSAI_URL", settings.BONSAI_URL or "http://localhost:9200")

from enrichment_service.storage.elasticsearch_client import ElasticsearchClient
from enrichment_service.models import TransactionInput, StructuredTransaction


async def seed_transactions() -> None:
    """Create the index (if needed) and insert sample transactions."""
    client = ElasticsearchClient()
    await client.initialize()

    sample_inputs = [
        TransactionInput(
            bridge_transaction_id=1,
            user_id=1,
            account_id=1,
            amount=-50.0,
            date=datetime(2025, 1, 15),
            currency_code="EUR",
            clean_description="Courses supermarché",
            operation_type="card",
        ),
        TransactionInput(
            bridge_transaction_id=2,
            user_id=1,
            account_id=1,
            amount=2000.0,
            date=datetime(2025, 1, 31),
            currency_code="EUR",
            clean_description="Salaire",
            operation_type="transfer",
        ),
        TransactionInput(
            bridge_transaction_id=3,
            user_id=1,
            account_id=1,
            amount=-70.5,
            date=datetime(2025, 2, 10),
            currency_code="EUR",
            clean_description="Restaurant",
            operation_type="card",
        ),
        TransactionInput(
            bridge_transaction_id=4,
            user_id=1,
            account_id=1,
            amount=-120.0,
            date=datetime(2025, 3, 5),
            currency_code="EUR",
            clean_description="Électricité",
            operation_type="transfer",
        ),
        TransactionInput(
            bridge_transaction_id=5,
            user_id=1,
            account_id=1,
            amount=150.0,
            date=datetime(2025, 4, 20),
            currency_code="EUR",
            clean_description="Remboursement",
            operation_type="transfer",
        ),
        TransactionInput(
            bridge_transaction_id=6,
            user_id=1,
            account_id=1,
            amount=-35.0,
            date=datetime(2025, 5, 18),
            currency_code="EUR",
            clean_description="Cinéma",
            operation_type="card",
        ),
    ]

    structured = [StructuredTransaction.from_transaction_input(tx) for tx in sample_inputs]
    summary = await client.index_transactions_batch(structured)
    print(f"Indexed {summary['indexed']}/{summary['total']} transactions")

    await client.close()


if __name__ == "__main__":
    asyncio.run(seed_transactions())
