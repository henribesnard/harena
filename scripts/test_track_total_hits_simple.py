"""
Test simple de la correction track_total_hits
Vérifie que QueryBuilder ajoute track_total_hits:True dans les requêtes générées
"""
import sys
import os
import json

# Fix encodage Windows
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from search_service.models.request import SearchRequest
from search_service.core.query_builder import QueryBuilder

def test_track_total_hits_in_query():
    """Test que track_total_hits est présent dans les requêtes générées"""

    print("=" * 80)
    print("TEST track_total_hits DANS QueryBuilder")
    print("=" * 80)

    try:
        query_builder = QueryBuilder()
        user_id = 100

        # Test 1: Requête simple sans filtre
        print("\n" + "-" * 80)
        print("Test 1: Requête simple sans filtre")
        print("-" * 80)

        search_request = SearchRequest(
            user_id=user_id,
            query="",
            page=1,
            page_size=10
        )

        result = query_builder.build_query(search_request)

        # Le résultat contient {"query": {...}, "target_index": "...", "search_type": "..."}
        # track_total_hits est dans result["query"]
        es_query = result.get("query", {})

        print(f"\nQuery générée (extrait):")
        print(f"  - 'track_total_hits' présent: {'track_total_hits' in es_query}")
        if 'track_total_hits' in es_query:
            print(f"  - Valeur: {es_query['track_total_hits']}")

        if 'track_total_hits' not in es_query:
            print("\n❌ ERREUR: track_total_hits manquant dans la query ES")
            print(f"\nClés présentes dans result: {list(result.keys())}")
            print(f"Clés présentes dans query: {list(es_query.keys())}")
            return False

        if es_query['track_total_hits'] != True:
            print(f"\n❌ ERREUR: track_total_hits = {es_query['track_total_hits']} (attendu: True)")
            return False

        print("✓ track_total_hits = True correctement défini")

        # Test 2: Requête avec filtres
        print("\n" + "-" * 80)
        print("Test 2: Requête avec filtre amount <= 75€")
        print("-" * 80)

        search_request = SearchRequest(
            user_id=user_id,
            query="",
            filters={"amount": {"lte": 75.0}},
            page=1,
            page_size=10
        )

        result = query_builder.build_query(search_request)
        es_query = result.get("query", {})

        if 'track_total_hits' not in es_query or es_query['track_total_hits'] != True:
            print("❌ ERREUR: track_total_hits manquant ou incorrect avec filtres")
            return False

        print("✓ track_total_hits = True correctement défini avec filtres")

        # Test 3: Requête avec query textuelle
        print("\n" + "-" * 80)
        print("Test 3: Requête avec recherche textuelle")
        print("-" * 80)

        search_request = SearchRequest(
            user_id=user_id,
            query="restaurant",
            page=1,
            page_size=10
        )

        result = query_builder.build_query(search_request)
        es_query = result.get("query", {})

        if 'track_total_hits' not in es_query or es_query['track_total_hits'] != True:
            print("❌ ERREUR: track_total_hits manquant ou incorrect avec query textuelle")
            return False

        print("✓ track_total_hits = True correctement défini avec query textuelle")

        # Test 4: Requête avec agrégations
        print("\n" + "-" * 80)
        print("Test 4: Requête avec agrégations")
        print("-" * 80)

        search_request = SearchRequest(
            user_id=user_id,
            query="",
            aggregations={"by_category": {"terms": {"field": "category_name"}}},
            page=1,
            page_size=10
        )

        result = query_builder.build_query(search_request)
        es_query = result.get("query", {})

        if 'track_total_hits' not in es_query or es_query['track_total_hits'] != True:
            print("❌ ERREUR: track_total_hits manquant ou incorrect avec agrégations")
            return False

        print("✓ track_total_hits = True correctement défini avec agrégations")

        # Résumé
        print("\n" + "=" * 80)
        print("✓ TOUS LES TESTS PASSENT")
        print("=" * 80)
        print("\nConclusion:")
        print("  - track_total_hits:True est bien ajouté dans toutes les requêtes")
        print("  - Elasticsearch comptera désormais tous les résultats sans limite 10K")
        print("  - Les logs afficheront des comptages précis (ex: 5234/5234 au lieu de 10000/10000)")

        return True

    except Exception as e:
        print(f"\n❌ ERREUR lors du test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_track_total_hits_in_query()
    sys.exit(0 if success else 1)
