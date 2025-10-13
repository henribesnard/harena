"""
Script pour verifier les comptes finaux du user 100 dans Elasticsearch
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from enrichment_service.storage.elasticsearch_client import ElasticsearchClient
from config_service.config import settings

async def verify_counts():
    """Verifie les comptes finaux du user 100"""

    client = ElasticsearchClient()

    try:
        await client.initialize()

        print("=== Verification des comptes pour user 100 ===\n")

        # Compter les transactions
        count_transactions_query = {
            'query': {
                'bool': {
                    'must': [
                        {'term': {'user_id': 100}},
                        {'term': {'document_type': 'transaction'}}
                    ]
                }
            }
        }

        async with client.session.post(
            f"{client.base_url}/{client.transactions_index}/_count",
            json=count_transactions_query
        ) as response:
            if response.status == 200:
                result = await response.json()
                tx_count = result.get('count', 0)
                print(f"Transactions (document_type='transaction'): {tx_count}")
            else:
                print(f"[ERROR] Erreur comptage transactions")

        # Compter les comptes
        count_accounts_query = {
            'query': {
                'bool': {
                    'must': [
                        {'term': {'user_id': 100}},
                        {'term': {'document_type': 'account'}}
                    ]
                }
            }
        }

        async with client.session.post(
            f"{client.base_url}/{client.transactions_index}/_count",
            json=count_accounts_query
        ) as response:
            if response.status == 200:
                result = await response.json()
                acc_count = result.get('count', 0)
                print(f"Comptes (document_type='account'): {acc_count}")
            else:
                print(f"[ERROR] Erreur comptage comptes")

        # Compter tous les documents
        count_all_query = {
            'query': {
                'term': {'user_id': 100}
            }
        }

        async with client.session.post(
            f"{client.base_url}/{client.transactions_index}/_count",
            json=count_all_query
        ) as response:
            if response.status == 200:
                result = await response.json()
                total_count = result.get('count', 0)
                print(f"Total documents user 100: {total_count}")
            else:
                print(f"[ERROR] Erreur comptage total")

        print("\n=== Verification terminee ===")

    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(verify_counts())
