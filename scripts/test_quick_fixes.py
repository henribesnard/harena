"""
Test rapide des corrections apportées
"""
import asyncio
import sys
from pathlib import Path

# Ajouter le répertoire racine au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from conversation_service.core.orchestrator import ConversationOrchestrator
from conversation_service.config.settings import ConfigManager

async def test_fixes():
    """Test les questions problématiques après corrections"""

    # Initialiser l'orchestrateur
    config_manager = ConfigManager()
    orchestrator = ConversationOrchestrator(config_manager)
    await orchestrator.initialize()

    test_cases = [
        # C009 - Query textuel devrait être présent
        {
            "id": "C009",
            "question": "Mes dépenses électronique",
            "expected_query_field": True,
            "expected_entities": ["query", "transaction_type"]
        },
        # F001 - Merchant + amount + operator
        {
            "id": "F001",
            "question": "Mes achats Amazon de plus de 100 euros",
            "expected_merchant": "Amazon",
            "expected_amount": 100,
            "expected_entities": ["merchant", "amount", "operator", "transaction_type"]
        },
        # F006 - Categories + amount + operator
        {
            "id": "F006",
            "question": "Mes achats en ligne inférieurs à 20 euros",
            "expected_categories": ["achats en ligne"],
            "expected_amount": 20,
            "expected_entities": ["categories", "amount", "operator", "transaction_type"]
        },
        # B015 - Merchants (liste)
        {
            "id": "B015",
            "question": "Mes achats Auchan Lidl Netflix",
            "expected_merchants": ["Auchan", "Lidl", "Netflix"],
            "expected_entities": ["merchants", "transaction_type"]
        }
    ]

    print("\n" + "="*80)
    print("TEST DES CORRECTIONS - Extraction et traduction des entités")
    print("="*80 + "\n")

    results = []

    for test_case in test_cases:
        print(f"\n{'='*80}")
        print(f"Test {test_case['id']}: {test_case['question']}")
        print(f"{'='*80}")

        try:
            # Traiter la question
            response = await orchestrator.process_message(
                user_id=100,
                message=test_case['question'],
                conversation_id=None
            )

            # Extraire les données
            entities = response.get('entities_structured', {})
            query_data = response.get('query_data', {})
            filters = query_data.get('filters', {})

            print(f"\n✓ Entités extraites:")
            for key, value in entities.items():
                print(f"  - {key}: {value}")

            print(f"\n✓ Query data - filters:")
            for key, value in filters.items():
                print(f"  - {key}: {value}")

            # Vérifications
            issues = []

            # Vérifier que toutes les entités attendues sont présentes
            for expected_entity in test_case.get('expected_entities', []):
                if expected_entity not in entities:
                    issues.append(f"❌ Entité manquante: {expected_entity}")

            # Vérifier merchant
            if 'expected_merchant' in test_case:
                if 'merchant_name' not in filters:
                    issues.append(f"❌ merchant_name absent de filters")
                elif filters['merchant_name'] != test_case['expected_merchant']:
                    issues.append(f"❌ merchant_name incorrect: {filters['merchant_name']} != {test_case['expected_merchant']}")

            # Vérifier merchants (liste)
            if 'expected_merchants' in test_case:
                if 'merchant_name' not in filters:
                    issues.append(f"❌ merchant_name absent de filters")
                elif not isinstance(filters['merchant_name'], list):
                    issues.append(f"❌ merchant_name devrait être une liste: {type(filters['merchant_name'])}")

            # Vérifier amount
            if 'expected_amount' in test_case:
                if 'amount_abs' not in filters:
                    issues.append(f"❌ amount_abs absent de filters")

            # Vérifier query field
            if test_case.get('expected_query_field'):
                if 'query' not in query_data:
                    issues.append(f"❌ Champ 'query' absent de query_data")

            # Afficher résultat
            if issues:
                print(f"\n❌ ÉCHOUÉ - Problèmes détectés:")
                for issue in issues:
                    print(f"  {issue}")
                results.append({
                    "id": test_case['id'],
                    "success": False,
                    "issues": issues
                })
            else:
                print(f"\n✅ RÉUSSI - Toutes les vérifications passées")
                results.append({
                    "id": test_case['id'],
                    "success": True
                })

        except Exception as e:
            print(f"\n❌ ERREUR: {str(e)}")
            import traceback
            traceback.print_exc()
            results.append({
                "id": test_case['id'],
                "success": False,
                "error": str(e)
            })

    # Résumé final
    print("\n\n" + "="*80)
    print("RÉSUMÉ DES TESTS")
    print("="*80)

    success_count = sum(1 for r in results if r['success'])
    total_count = len(results)

    print(f"\nRésultats: {success_count}/{total_count} tests réussis")

    for result in results:
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        print(f"  {status} - {result['id']}")
        if 'issues' in result:
            for issue in result['issues']:
                print(f"    {issue}")

    return success_count == total_count

if __name__ == "__main__":
    success = asyncio.run(test_fixes())
    sys.exit(0 if success else 1)
