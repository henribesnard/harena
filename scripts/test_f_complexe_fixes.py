"""
Script de test rapide pour valider les corrections F_complexe
"""
import asyncio
import sys
from pathlib import Path

# Ajouter le répertoire parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from conversation_service.core.conversation_orchestrator import ConversationOrchestrator

async def test_fixes():
    """Test des cas problématiques F003, F004, F011, F015, F016, F017"""

    # Initialiser l'orchestrateur
    orchestrator = ConversationOrchestrator()
    await orchestrator.initialize()

    # Cas de test
    test_cases = [
        {
            "id": "F003",
            "question": "Mes revenus Netflix supérieurs à 50 euros ce mois",
            "expected": {
                "merchant": "Netflix",
                "amount": 50,
                "operator": "gt",
                "date_range": "this_month",
                "transaction_type": "credit"
            }
        },
        {
            "id": "F004",
            "question": "Toutes mes transactions Carrefour de janvier",
            "expected": {
                "merchant": "Carrefour",
                "date_range": "2025-01",
                "transaction_type": "all"
            }
        },
        {
            "id": "F011",
            "question": "Mes achats en ligne du week-end dernier",
            "expected": {
                "date_range": "last_weekend",
                "transaction_type": "debit",
                "categories": ["achats en ligne"]  # PAS 7 catégories!
            }
        },
        {
            "id": "F015",
            "question": "Mes virements de 1000 euros et plus en juin",
            "expected": {
                "amount": 1000,
                "operator": "gte",
                "date_range": "2025-06",
                "operation_type": "transfer"  # PAS transaction_type!
            }
        },
        {
            "id": "F016",
            "question": "Mes paiements par carte de 15 euros ou moins",
            "expected": {
                "amount": 15,
                "operator": "lte",
                "operation_type": "card",
                "transaction_type": "debit"
            }
        },
        {
            "id": "F017",
            "question": "Mes retraits d'espèces d'au minimum 50 euros cette semaine",
            "expected": {
                "amount": 50,
                "operator": "gte",
                "date_range": "this_week",
                "operation_type": "withdrawal",
                "transaction_type": "debit"
            }
        }
    ]

    results = []

    for test in test_cases:
        print(f"\n{'='*80}")
        print(f"Test {test['id']}: {test['question']}")
        print(f"{'='*80}")

        try:
            # Extraction d'entités
            result = await orchestrator.process_message(
                user_id=100,
                message=test['question'],
                conversation_id=f"test_{test['id']}"
            )

            entities = result.get("entities_structured", {})
            query_data = result.get("query_data", {})

            print(f"\n📋 Entités extraites:")
            for key, value in entities.items():
                print(f"  {key}: {value}")

            print(f"\n🔍 Filtres de query:")
            filters = query_data.get("filters", {})
            for key, value in filters.items():
                print(f"  {key}: {value}")

            # Vérification
            issues = []
            for expected_key, expected_value in test['expected'].items():
                if expected_key not in entities:
                    issues.append(f"❌ Manque: {expected_key}")
                elif expected_key == "categories" and isinstance(expected_value, list):
                    # Vérification spéciale pour les catégories
                    actual_categories = entities.get(expected_key, [])
                    if len(actual_categories) != len(expected_value):
                        issues.append(f"❌ {expected_key}: attendu {expected_value}, obtenu {actual_categories}")
                    elif actual_categories != expected_value:
                        issues.append(f"⚠️  {expected_key}: contenu différent - attendu {expected_value}, obtenu {actual_categories}")

            if not issues:
                print(f"\n✅ {test['id']}: PASS")
                results.append({"id": test['id'], "status": "PASS", "issues": []})
            else:
                print(f"\n❌ {test['id']}: FAIL")
                for issue in issues:
                    print(f"  {issue}")
                results.append({"id": test['id'], "status": "FAIL", "issues": issues})

        except Exception as e:
            print(f"\n❌ {test['id']}: ERROR - {str(e)}")
            results.append({"id": test['id'], "status": "ERROR", "error": str(e)})

    # Résumé
    print(f"\n{'='*80}")
    print("RÉSUMÉ")
    print(f"{'='*80}")

    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    errors = sum(1 for r in results if r["status"] == "ERROR")

    print(f"Tests passés: {passed}/{len(test_cases)}")
    print(f"Tests échoués: {failed}/{len(test_cases)}")
    print(f"Erreurs: {errors}/{len(test_cases)}")

    if failed > 0 or errors > 0:
        print("\nDétails des échecs:")
        for r in results:
            if r["status"] != "PASS":
                print(f"  {r['id']}: {r.get('issues', r.get('error', 'Unknown'))}")

if __name__ == "__main__":
    asyncio.run(test_fixes())
