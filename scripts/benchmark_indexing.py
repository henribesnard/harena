import asyncio
import json
import time
from datetime import datetime

from enrichment_service.storage.elasticsearch_client import ElasticsearchClient
from enrichment_service.models import TransactionInput, BatchTransactionInput
from enrichment_service.core.processor import ElasticsearchTransactionProcessor


async def run_benchmark():
    with open("samples/sample_transactions.json", "r") as f:
        raw = json.load(f)
    transactions = [TransactionInput(**{**tx, "date": datetime.fromisoformat(tx["date"])} ) for tx in raw]

    batch = BatchTransactionInput(user_id=1, transactions=transactions)

    client = ElasticsearchClient()
    await client.initialize()
    processor = ElasticsearchTransactionProcessor(client)

    start = time.perf_counter()
    result = await processor.process_transactions_batch(batch)
    elapsed = time.perf_counter() - start
    print(f"Indexed {result.successful}/{result.total_transactions} in {elapsed:.2f}s")

    await client.session.close()


if __name__ == "__main__":
    asyncio.run(run_benchmark())
