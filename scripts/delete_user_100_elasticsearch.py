"""
Script pour supprimer toutes les transactions et comptes du user 100 dans Elasticsearch
"""
import asyncio
import sys
from pathlib import Path

# Ajouter le répertoire parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from enrichment_service.storage.elasticsearch_client import ElasticsearchClient
from config_service.config import settings

async def delete_user_data():
    """Supprime toutes les données du user 100 dans Elasticsearch"""

    client = ElasticsearchClient()

    try:
        # Initialiser la connexion
        await client.initialize()

        print("=== Suppression des donnees du user 100 ===\n")

        # Compter tous les documents du user 100
        print("-> Comptage initial...")
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
                print(f"  [INFO] Total documents user 100 avant suppression: {total_count}")
            else:
                print(f"  [ERROR] Erreur comptage initial")

        # Supprimer TOUS les documents du user 100 (sans filtre document_type)
        print("\n-> Suppression de tous les documents du user 100...")
        delete_all_query = {
            'query': {
                'term': {'user_id': 100}
            }
        }

        async with client.session.post(
            f"{client.base_url}/{client.transactions_index}/_delete_by_query?refresh=true",
            json=delete_all_query
        ) as response:
            if response.status == 200:
                result = await response.json()
                print(f"  [OK] {result.get('deleted', 0)} documents supprimes")
            else:
                error_text = await response.text()
                print(f"  [ERROR] Erreur suppression: {response.status} - {error_text}")

        # Verifier les documents restants
        print("\n-> Verification finale...")
        async with client.session.post(
            f"{client.base_url}/{client.transactions_index}/_count",
            json=count_all_query
        ) as response:
            if response.status == 200:
                result = await response.json()
                remaining_count = result.get('count', 0)
                print(f"  [OK] Documents restants pour user 100: {remaining_count}")
            else:
                print(f"  [ERROR] Erreur comptage final")

        print("\n=== Suppression terminee ===")

    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(delete_user_data())
