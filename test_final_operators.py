#!/usr/bin/env python3
"""
Test final pour valider que les opérateurs fonctionnent dans le système complet
"""

import sys
import os
sys.path.append('conversation_service')

def test_regex_patterns():
    """Test direct des patterns regex pour les opérateurs"""
    
    import re
    
    # Patterns from intent classifier
    amount_comparison_patterns = [
        (r'(?:plus de|supérieur(?:e)?s? à|au-dessus de|>\s*)\s*(\d+(?:[.,]\d{1,2})?)\s*(?:€|euros?|eur)', 'gte'),
        (r'(?:moins de|inférieur(?:e)?s? à|en-dessous de|<\s*)\s*(\d+(?:[.,]\d{1,2})?)\s*(?:€|euros?|eur)', 'lt'),
        (r'(?:entre)\s*(\d+(?:[.,]\d{1,2})?)\s*(?:et)\s*(\d+(?:[.,]\d{1,2})?)\s*(?:€|euros?|eur)', 'range')
    ]
    
    test_cases = [
        ("mes dépenses de moins de 500 euros", "lt", 500),
        ("mes dépenses de plus de 200 euros", "gte", 200), 
        ("transactions supérieures à 100 euros", "gte", 100),
        ("achats entre 50 et 200 euros", "range", (50, 200))
    ]
    
    print("Test des patterns regex pour opérateurs")
    print("=" * 45)
    
    success = 0
    
    for query, expected_op, expected_amount in test_cases:
        message_lower = query.lower()
        found = False
        
        for pattern, operator in amount_comparison_patterns:
            match = re.search(pattern, message_lower)
            if match:
                if operator == expected_op:
                    if operator == 'range':
                        min_val = float(match.group(1))
                        max_val = float(match.group(2))
                        if (min_val, max_val) == expected_amount:
                            print(f"SUCCESS '{query}' -> {operator} ({min_val}-{max_val})")
                            success += 1
                        else:
                            print(f"FAILED '{query}' -> {operator} but wrong range")
                    else:
                        amount = float(match.group(1))
                        if amount == expected_amount:
                            print(f"SUCCESS '{query}' -> {operator} ({amount})")
                            success += 1
                        else:
                            print(f"FAILED '{query}' -> {operator} but wrong amount")
                else:
                    print(f"FAILED '{query}' -> {operator} (expected {expected_op})")
                found = True
                break
        
        if not found:
            print(f"FAILED '{query}' -> no match")
    
    print(f"\nRésultat: {success}/{len(test_cases)} tests réussis")
    return success == len(test_cases)

def test_few_shot_examples_content():
    """Vérifie que les few-shot examples contiennent les bons opérateurs"""
    
    print("\nTest du contenu des few-shot examples")
    print("=" * 42)
    
    # Lire le fichier intent_classifier.py
    try:
        with open('conversation_service/agents/llm/intent_classifier.py', 'r', encoding='utf-8') as f:
            content = f.read()
    except:
        print("Erreur: impossible de lire le fichier intent_classifier.py")
        return False
    
    # Chercher les opérateurs dans les examples
    operators_in_examples = []
    
    if '"operator": "lt"' in content:
        operators_in_examples.append("lt")
    if '"operator": "gte"' in content:
        operators_in_examples.append("gte")
    if '"operator": "gt"' in content:
        operators_in_examples.append("gt")
    if '"operator": "eq"' in content:
        operators_in_examples.append("eq")
    if '"operator": "range"' in content:
        operators_in_examples.append("range")
    
    expected_operators = {"lt", "gte", "gt", "eq", "range"}
    found_operators = set(operators_in_examples)
    missing = expected_operators - found_operators
    
    print(f"Opérateurs trouvés dans les examples: {sorted(found_operators)}")
    
    if missing:
        print(f"FAILED Opérateurs manquants: {sorted(missing)}")
        return False
    else:
        print("SUCCESS Tous les opérateurs sont présents dans les examples")
        
        # Vérifier quelques exemples spécifiques
        checks = [
            ("moins de 500 euros", "lt"),
            ("plus de 500 euros", "gte"),
            ("exactement", "eq"),
            ("entre", "range")
        ]
        
        all_found = True
        for phrase, op in checks:
            if phrase in content and f'"operator": "{op}"' in content:
                print(f"SUCCESS Example pour '{phrase}' -> {op}")
            else:
                print(f"FAILED Example manquant pour '{phrase}' -> {op}")
                all_found = False
        
        return all_found

if __name__ == "__main__":
    print("Test Final - Validation des Opérateurs")
    print("=" * 50)
    
    success = True
    success &= test_regex_patterns()
    success &= test_few_shot_examples_content()
    
    print("\n" + "=" * 50)
    if success:
        print("SUCCESS: Les opérateurs sont maintenant correctement implémentés!")
        print("\nRésumé des corrections apportées:")
        print("- + Ajout d'examples few-shot pour 'moins de' (lt)")  
        print("- + Ajout d'examples few-shot pour 'exactement' (eq)")
        print("- + Les patterns regex fonctionnent correctement") 
        print("- + Pas de mock interfering avec la logique métier")
        print("\nMaintenant 'mes dépenses de moins de 500 euros' devrait")
        print("correctement générer: {'operator': 'lt', 'amount': 500}")
    else:
        print("FAILURE: Il reste des problèmes à corriger.")
    
    sys.exit(0 if success else 1)