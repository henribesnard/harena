#!/usr/bin/env python3
"""
Test pour vérifier que les opérateurs de montant fonctionnent correctement
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'conversation_service'))

from agents.llm.intent_classifier import IntentClassifier, ClassificationRequest
from agents.llm.llm_provider import LLMProviderManager

def test_amount_operators():
    """Test les différents opérateurs de montant"""
    
    # Test cases avec opérateurs
    test_cases = [
        {
            "query": "mes dépenses de moins de 500 euros",
            "expected_operator": "lt",
            "expected_amount": 500,
            "description": "moins de = less than"
        },
        {
            "query": "mes dépenses de plus de 200 euros", 
            "expected_operator": "gte",
            "expected_amount": 200,
            "description": "plus de = greater than or equal"
        },
        {
            "query": "transactions supérieures à 100 euros",
            "expected_operator": "gt", 
            "expected_amount": 100,
            "description": "supérieures à = greater than"
        },
        {
            "query": "transactions de 50 euros exactement",
            "expected_operator": "eq",
            "expected_amount": 50,
            "description": "exactement = equal"
        },
        {
            "query": "achats entre 20 et 100 euros",
            "expected_operator": "range",
            "expected_min": 20,
            "expected_max": 100,
            "description": "entre X et Y = range"
        }
    ]
    
    print("Test des opérateurs de montant avec few-shot examples")
    print("=" * 60)
    
    success_count = 0
    total_tests = len(test_cases)
    
    # Tester la détection des opérateurs avec les examples
    classifier = IntentClassifier(llm_manager=None)
    
    # Simuler les few-shot examples chargés
    classifier._examples_loaded = True
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case['description']}")
        print(f"Query: '{test_case['query']}'")
        
        # Tester extraction basique (sans LLM)
        entities = classifier._extract_basic_entities(test_case['query'])
        
        montant_entities = [e for e in entities if e.name == "montant"]
        
        if not montant_entities:
            print(f"FAILED: Aucune entité 'montant' détectée")
            continue
            
        entity = montant_entities[0]
        
        if not isinstance(entity.value, dict):
            print(f"FAILED: Valeur d'entité pas un dict: {entity.value}")
            continue
        
        if "operator" not in entity.value:
            print(f"FAILED: Pas d'opérateur dans l'entité: {entity.value}")
            continue
            
        actual_operator = entity.value["operator"]
        
        if test_case["expected_operator"] == "range":
            # Test spécial pour range
            if actual_operator != "range":
                print(f"FAILED: Opérateur attendu 'range', reçu '{actual_operator}'")
                continue
                
            if entity.value.get("min") != test_case["expected_min"]:
                print(f"FAILED: Min attendu {test_case['expected_min']}, reçu {entity.value.get('min')}")
                continue
                
            if entity.value.get("max") != test_case["expected_max"]:
                print(f"FAILED: Max attendu {test_case['expected_max']}, reçu {entity.value.get('max')}")
                continue
        else:
            # Test pour opérateurs simples
            if actual_operator != test_case["expected_operator"]:
                print(f"FAILED: Opérateur attendu '{test_case['expected_operator']}', reçu '{actual_operator}'")
                continue
                
            if entity.value.get("amount") != test_case["expected_amount"]:
                print(f"FAILED: Montant attendu {test_case['expected_amount']}, reçu {entity.value.get('amount')}")
                continue
        
        print(f"SUCCESS: Opérateur '{actual_operator}' correctement détecté")
        success_count += 1
    
    print("\n" + "=" * 60)
    print(f"Résumé: {success_count}/{total_tests} tests réussis")
    
    if success_count == total_tests:
        print("SUCCESS: Tous les opérateurs fonctionnent correctement!")
        return True
    else:
        print("FAILURE: Certains opérateurs ne fonctionnent pas correctement.")
        return False

def test_few_shot_examples_structure():
    """Vérifie que les few-shot examples sont bien structurés"""
    
    print("\nVérification de la structure des few-shot examples")
    print("=" * 50)
    
    classifier = IntentClassifier(llm_manager=None)
    classifier._examples_loaded = True
    
    examples = classifier._few_shot_examples
    
    # Rechercher les exemples avec montant
    amount_examples = []
    for example in examples:
        if "montant" in example["assistant"] or "moins de" in example["user"] or "plus de" in example["user"]:
            amount_examples.append(example)
    
    print(f"Trouvé {len(amount_examples)} exemples avec montant")
    
    operators_found = set()
    for example in amount_examples:
        user_query = example["user"]
        assistant_response = example["assistant"]
        
        print(f"\nExample: {user_query}")
        
        # Extraire l'opérateur de la réponse
        if '"operator": "lt"' in assistant_response:
            operators_found.add("lt")
            print("  → Opérateur: lt (moins de)")
        elif '"operator": "gte"' in assistant_response:
            operators_found.add("gte") 
            print("  → Opérateur: gte (plus de)")
        elif '"operator": "gt"' in assistant_response:
            operators_found.add("gt")
            print("  → Opérateur: gt (supérieur à)")
        elif '"operator": "eq"' in assistant_response:
            operators_found.add("eq")
            print("  → Opérateur: eq (égal à)")
        elif '"operator": "range"' in assistant_response:
            operators_found.add("range")
            print("  → Opérateur: range (entre X et Y)")
    
    expected_operators = {"lt", "gte", "gt", "eq", "range"}
    missing_operators = expected_operators - operators_found
    
    # Charger les exemples manuellement pour le test
    await classifier._load_few_shot_examples()
    examples = classifier._few_shot_examples
    
    if missing_operators:
        print(f"\nMISSING: Opérateurs manquants dans les examples: {missing_operators}")
        return False
    else:
        print(f"\nSUCCESS: Tous les opérateurs sont présents dans les examples: {operators_found}")
        return True

if __name__ == "__main__":
    print("Test de correction des opérateurs de montant")
    print("=" * 60)
    
    success = True
    success &= test_few_shot_examples_structure()
    success &= test_amount_operators()
    
    print("\n" + "=" * 60)
    if success:
        print("SUCCESS: Tous les tests passent! Les opérateurs sont maintenant corrects.")
    else:
        print("FAILURE: Certains tests échouent.")
    
    sys.exit(0 if success else 1)