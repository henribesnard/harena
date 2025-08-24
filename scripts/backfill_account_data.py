"""Backfill account data for existing transactions in Elasticsearch.

This script iterates over transactions stored in the database, fetches
associated account information, and reindexes the documents into the
``harena_transactions`` Elasticsearch index. It provides a dry-run mode to
preview the operations without modifying the index and verifies a sample of
updated documents once completed.

Usage::

    python scripts/backfill_account_data.py [--limit 100] [--dry-run]

The script can be executed as a one-off management command (e.g. via
``heroku run`` or the process type defined in the ``Procfile``).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from typing import Iterable, List

from sqlalchemy.orm import Session

from db_service.session import SessionLocal
from db_service.models.sync import RawTransaction, SyncAccount
from enrichment_service.models import TransactionInput, StructuredTransaction
from enrichment_service.storage.elasticsearch_client import ElasticsearchClient

logger = logging.getLogger("backfill_account_data")
BATCH_SIZE = 100


def _build_transaction_input(tx: RawTransaction, account: SyncAccount) -> TransactionInput:
    """Create a :class:`TransactionInput` from ORM objects."""
    currency = tx.currency_code or account.currency_code
    return TransactionInput(
        bridge_transaction_id=tx.bridge_transaction_id,
        user_id=tx.user_id,
        account_id=tx.account_id,
        clean_description=tx.clean_description,
        provider_description=tx.provider_description,
        amount=tx.amount,
        date=tx.date,
        booking_date=tx.booking_date,
        transaction_date=tx.transaction_date,
        value_date=tx.value_date,
        currency_code=currency,
        category_id=tx.category_id,
        operation_type=tx.operation_type,
        deleted=tx.deleted,
        future=tx.future,
    )


def _fetch_transactions(db: Session, limit: int | None) -> Iterable[tuple[RawTransaction, SyncAccount]]:
    """Yield transactions joined with account information."""
    query = db.query(RawTransaction, SyncAccount).join(SyncAccount, RawTransaction.account_id == SyncAccount.id)
    if limit:
        query = query.limit(limit)
    return query.yield_per(BATCH_SIZE)


async def _verify_documents(client: ElasticsearchClient, doc_ids: List[str]) -> None:
    """Fetch a sample of documents to ensure indexing succeeded."""
    if not doc_ids:
        return

    for doc_id in doc_ids:
        url = f"{client.base_url}/{client.index_name}/_doc/{doc_id}"
        try:
            async with client.session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    account_id = data.get("_source", {}).get("account_id")
                    logger.info("Verified %s with account_id=%s", doc_id, account_id)
                else:
                    logger.warning("Document %s not found (status %s)", doc_id, resp.status)
        except Exception as exc:  # pragma: no cover - network issues
            logger.error("Error verifying %s: %s", doc_id, exc)


async def backfill_account_data(limit: int | None = None, dry_run: bool = False, verify_sample: int = 5) -> None:
    """Backfill account information for existing transactions."""
    db = SessionLocal()
    processed: List[str] = []

    client = ElasticsearchClient()
    if not dry_run:
        await client.initialize()

    try:
        for tx, account in _fetch_transactions(db, limit):
            tx_input = _build_transaction_input(tx, account)
            structured = StructuredTransaction.from_transaction_input(tx_input)
            doc_id = structured.get_document_id()

            if dry_run:
                logger.info(
                    "[DRY-RUN] Would reindex transaction %s for user %s (account %s - %s)",
                    tx.bridge_transaction_id,
                    tx.user_id,
                    account.bridge_account_id,
                    account.account_name,
                )
                continue

            ok = await client.index_transaction(structured)
            if ok:
                processed.append(doc_id)
                logger.info(
                    "Reindexed transaction %s for user %s (account %s - %s)",
                    tx.bridge_transaction_id,
                    tx.user_id,
                    account.bridge_account_id,
                    account.account_name,
                )
            else:
                logger.error("Failed to reindex transaction %s", tx.bridge_transaction_id)

        if not dry_run and verify_sample:
            sample = processed[:verify_sample]
            await _verify_documents(client, sample)

    finally:
        db.close()
        if not dry_run and client.session:
            await client.session.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill account data for existing transactions")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of transactions to process")
    parser.add_argument("--dry-run", action="store_true", help="Log actions without reindexing")
    parser.add_argument("--sample-size", type=int, default=5, help="Number of documents to verify after reindexing")
    return parser.parse_args()


async def _async_main() -> None:
    args = parse_args()
    await backfill_account_data(limit=args.limit, dry_run=args.dry_run, verify_sample=args.sample_size)


def main() -> None:  # pragma: no cover - entry point
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    asyncio.run(_async_main())


if __name__ == "__main__":  # pragma: no cover - script execution
    main()
