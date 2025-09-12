#!/usr/bin/env python3
"""
Test simple pour vérifier que les opérateurs de montant fonctionnent correctement
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'conversation_service'))

from agents.llm.intent_classifier import IntentClassifier

def test_basic_entity_extraction():
    """Test la détection basique des opérateurs sans LLM"""
    
    test_cases = [
        {
            "query": "mes dépenses de moins de 500 euros",
            "expected_operator": "lt",
            "expected_amount": 500.0,
            "description": "moins de = less than"
        },
        {
            "query": "mes dépenses de plus de 200 euros", 
            "expected_operator": "gte",
            "expected_amount": 200.0,
            "description": "plus de = greater than or equal"
        },
        {
            "query": "transactions supérieures à 100 euros",
            "expected_operator": "gte",  # Note: "supérieures à" maps to gte pattern
            "expected_amount": 100.0,
            "description": "supérieures à = greater than or equal"
        },
        {
            "query": "achats entre 20 et 100 euros",
            "expected_operator": "range",
            "expected_min": 20.0,
            "expected_max": 100.0,
            "description": "entre X et Y = range"
        }
    ]
    
    print("Test de l'extraction basique des opérateurs")
    print("=" * 50)
    
    classifier = IntentClassifier(llm_manager=None)
    
    success_count = 0
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case['description']}")
        print(f"Query: '{test_case['query']}'")
        
        # Extraire entités avec _extract_basic_entities
        entities = classifier._extract_basic_entities(test_case['query'])
        
        # Trouver entités montant
        montant_entities = [e for e in entities if e.name == "montant"]
        
        if not montant_entities:
            print("FAILED: Aucune entité 'montant' détectée")
            continue
            
        entity = montant_entities[0]
        print(f"Entity détectée: {entity.value}")
        
        if not isinstance(entity.value, dict):
            print(f"FAILED: Valeur d'entité n'est pas un dict: {entity.value}")
            continue
        
        if "operator" not in entity.value:
            print(f"FAILED: Pas d'opérateur dans l'entité: {entity.value}")
            continue
            
        actual_operator = entity.value["operator"]
        
        # Test selon le type d'opérateur
        if test_case["expected_operator"] == "range":
            if actual_operator != "range":
                print(f"FAILED: Opérateur attendu 'range', reçu '{actual_operator}'")
                continue
                
            if entity.value.get("min") != test_case["expected_min"]:
                print(f"FAILED: Min attendu {test_case['expected_min']}, reçu {entity.value.get('min')}")
                continue
                
            if entity.value.get("max") != test_case["expected_max"]:
                print(f"FAILED: Max attendu {test_case['expected_max']}, reçu {entity.value.get('max')}")
                continue
                
            print(f"SUCCESS: Opérateur 'range' détecté avec min={entity.value['min']}, max={entity.value['max']}")
        else:
            if actual_operator != test_case["expected_operator"]:
                print(f"FAILED: Opérateur attendu '{test_case['expected_operator']}', reçu '{actual_operator}'")
                continue
                
            if entity.value.get("amount") != test_case["expected_amount"]:
                print(f"FAILED: Montant attendu {test_case['expected_amount']}, reçu {entity.value.get('amount')}")
                continue
                
            print(f"SUCCESS: Opérateur '{actual_operator}' détecté avec montant={entity.value['amount']}")
        
        success_count += 1
    
    print("\n" + "=" * 50)
    print(f"Résumé: {success_count}/{len(test_cases)} tests réussis")
    
    return success_count == len(test_cases)

def check_few_shot_examples():
    """Vérifier manuellement les few-shot examples"""
    
    print("\nVérification manuelle des few-shot examples")
    print("=" * 50)
    
    classifier = IntentClassifier(llm_manager=None)
    
    # Charger les examples (synchrone)
    classifier._few_shot_examples = []  # Reset first
    
    # Les exemples sont définis dans le code - on peut les vérifier directement
    examples_to_check = [
        ("Mes dépenses de plus de 500 euros", "gte"),
        ("Transactions supérieures à 100€", "gt"),
        ("Mes dépenses de moins de 500 euros", "lt"),
        ("Transactions de 50 euros exactement", "eq"), 
        ("Achats entre 20 et 100 euros", "range")
    ]
    
    # Simuler le chargement des exemples
    try:
        # Force sync load by calling the internal method structure
        from agents.llm.intent_classifier import IntentClassifier
        import inspect
        
        # Get _load_few_shot_examples method
        source = inspect.getsource(classifier._load_few_shot_examples)
        
        # Check if examples contain the operators we expect
        operators_found = set()
        if '"operator": "lt"' in source:
            operators_found.add("lt")
        if '"operator": "gte"' in source:
            operators_found.add("gte")
        if '"operator": "gt"' in source:
            operators_found.add("gt")
        if '"operator": "eq"' in source:
            operators_found.add("eq")
        if '"operator": "range"' in source:
            operators_found.add("range")
        
        expected_operators = {"lt", "gte", "gt", "eq", "range"}
        missing = expected_operators - operators_found
        
        if missing:
            print(f"MISSING operators in few-shot examples: {missing}")
            return False
        else:
            print(f"SUCCESS: All operators found in few-shot examples: {operators_found}")
            return True
            
    except Exception as e:
        print(f"Erreur inspection: {e}")
        return False

if __name__ == "__main__":
    print("Test de correction des opérateurs")
    print("=" * 50)
    
    success = True
    success &= check_few_shot_examples()
    success &= test_basic_entity_extraction()
    
    print("\n" + "=" * 50)
    if success:
        print("SUCCESS: Les opérateurs semblent correctement implémentés!")
    else:
        print("FAILURE: Il y a encore des problèmes avec les opérateurs.")
    
    sys.exit(0 if success else 1)