"""
Test script pour vérifier la correction operation_type
Ce script teste que le LLM génère uniquement les valeurs correctes: Carte, Prélèvement, Virement, Chèque
"""
import asyncio
import sys
import os

# Ajouter le répertoire courant au path pour les imports
sys.path.insert(0, os.path.dirname(__file__))

from conversation_service.agents.llm.intent_classifier import IntentClassifier, ClassificationRequest
from conversation_service.config.settings import ConfigManager

# Couleurs pour l'affichage
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

# Requêtes de test avec les operation_type attendus
TEST_QUERIES = [
    {
        "query": "Mes paiements par carte du mois dernier",
        "expected_operation_type": "Carte",
        "description": "Paiements par carte"
    },
    {
        "query": "Mes retraits espèces de la semaine",
        "expected_operation_type": "Carte",
        "description": "Retraits espèces (doivent être 'Carte', PAS 'retrait')"
    },
    {
        "query": "Tous mes prélèvements automatiques",
        "expected_operation_type": "Prélèvement",
        "description": "Prélèvements automatiques"
    },
    {
        "query": "Mes virements SEPA du dernier trimestre",
        "expected_operation_type": "Virement",
        "description": "Virements SEPA"
    },
    {
        "query": "Les chèques que j'ai émis",
        "expected_operation_type": "Chèque",
        "description": "Chèques"
    },
    {
        "query": "Mes dépenses alimentaires",
        "expected_operation_type": None,
        "description": "Pas de mention de moyen de paiement (ne doit PAS extraire operation_type)"
    },
]

# Valeurs autorisées (extraites de la base de données)
VALID_OPERATION_TYPES = ["Carte", "Prélèvement", "Virement", "Chèque"]

async def test_operation_type_extraction():
    """Test l'extraction d'operation_type par le LLM"""

    print(f"\n{BLUE}{'='*80}{RESET}")
    print(f"{BLUE}TEST CORRECTION OPERATION_TYPE{RESET}")
    print(f"{BLUE}{'='*80}{RESET}\n")

    # Initialiser le classifier
    config_manager = ConfigManager()
    classifier = IntentClassifier(config_manager)

    print(f"{YELLOW}Valeurs autorisées:{RESET} {', '.join(VALID_OPERATION_TYPES)}\n")

    results = []

    for i, test_case in enumerate(TEST_QUERIES, 1):
        query = test_case["query"]
        expected = test_case["expected_operation_type"]
        description = test_case["description"]

        print(f"{BLUE}Test {i}/{len(TEST_QUERIES)}:{RESET} {description}")
        print(f"  Requête: '{query}'")
        print(f"  Attendu: {expected if expected else 'Aucun'}")

        try:
            # Créer la requête de classification
            classification_request = ClassificationRequest(
                user_message=query,
                user_id=100,
                conversation_context=[]
            )

            # Classifier la requête
            result = await classifier.classify_intent(classification_request)

            # Extraire operation_type des entités
            extracted_operation_type = None
            if result.entities:
                print(f"  Entités extraites: {[f'{e.name}={e.value}' for e in result.entities]}")
                for entity in result.entities:
                    if entity.name == "operation_type":
                        extracted_operation_type = entity.value
                        break
            else:
                print(f"  Entités extraites: Aucune")

            print(f"  operation_type extrait: {extracted_operation_type if extracted_operation_type else 'Aucun'}")

            if not result.success:
                print(f"  Erreur classification: {result.error_message}")

            # Vérifier la correction
            is_valid = True
            error_messages = []

            # 1. Vérifier que la valeur extraite est dans la liste autorisée
            if extracted_operation_type is not None:
                if extracted_operation_type not in VALID_OPERATION_TYPES:
                    is_valid = False
                    error_messages.append(f"Valeur invalide '{extracted_operation_type}' (pas dans {VALID_OPERATION_TYPES})")

            # 2. Vérifier que c'est la valeur attendue
            if expected is None:
                if extracted_operation_type is not None:
                    is_valid = False
                    error_messages.append(f"Ne devrait pas extraire operation_type, mais a extrait '{extracted_operation_type}'")
            else:
                if extracted_operation_type != expected:
                    is_valid = False
                    error_messages.append(f"Attendu '{expected}' mais extrait '{extracted_operation_type}'")

            # Afficher le résultat
            if is_valid:
                print(f"  {GREEN}OK SUCCES{RESET}")
                results.append({"test": description, "success": True})
            else:
                print(f"  {RED}X ECHEC:{RESET}")
                for err in error_messages:
                    print(f"    - {err}")
                results.append({"test": description, "success": False, "errors": error_messages})

            print()

        except Exception as e:
            print(f"  {RED}X ERREUR:{RESET} {str(e)}\n")
            results.append({"test": description, "success": False, "errors": [str(e)]})

    # Résumé
    print(f"{BLUE}{'='*80}{RESET}")
    print(f"{BLUE}RÉSUMÉ{RESET}")
    print(f"{BLUE}{'='*80}{RESET}\n")

    total = len(results)
    successes = sum(1 for r in results if r["success"])
    failures = total - successes

    print(f"Total: {total} tests")
    print(f"{GREEN}Succès: {successes}{RESET}")
    if failures > 0:
        print(f"{RED}Échecs: {failures}{RESET}")
        print(f"\n{YELLOW}Tests échoués:{RESET}")
        for r in results:
            if not r["success"]:
                print(f"  - {r['test']}")
                for err in r.get("errors", []):
                    print(f"    {err}")

    print(f"\n{BLUE}{'='*80}{RESET}")

    return successes == total

if __name__ == "__main__":
    success = asyncio.run(test_operation_type_extraction())
    sys.exit(0 if success else 1)
