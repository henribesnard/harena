#!/usr/bin/env python3
"""
Test de conversion des objets montant dans le template engine
"""

import sys
import os
sys.path.append('conversation_service')

def test_conversion():
    """Test la conversion d'objets montant en filtres Elasticsearch"""
    
    from core.template_engine import TemplateEngine
    
    engine = TemplateEngine()
    
    # Test de la fonction _to_elasticsearch_amount_filter
    print("Test de _to_elasticsearch_amount_filter:")
    
    test_cases = [
        {"operator": "lt", "amount": 500, "currency": "EUR"},
        {"operator": "gte", "amount": 200, "currency": "EUR"},
        {"operator": "eq", "amount": 50, "currency": "EUR"},
        {"operator": "range", "min": 20, "max": 100, "currency": "EUR"}
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        result = engine._to_elasticsearch_amount_filter(test_case)
        print(f"  {i}. {test_case} -> {result}")
    
    # Test de la fonction _remove_null_values avec string parsing
    print("\nTest de _remove_null_values avec parsing de string:")
    
    test_obj = {
        "user_id": 34,
        "filters": {
            "transaction_type": "debit",
            "amount_abs": "{'operator': 'lt', 'amount': 500, 'currency': 'EUR'}"
        }
    }
    
    print(f"Avant: {test_obj}")
    result = engine._remove_null_values(test_obj)
    print(f"Après: {result}")
    
    # Test avec vraie string des logs
    print("\nTest avec la vraie string problématique:")
    
    real_test = {
        "amount_abs": "{'operator': 'lt', 'amount': 500, 'currency': 'EUR'}"
    }
    
    print(f"Avant: {real_test}")
    result2 = engine._remove_null_values(real_test)
    print(f"Après: {result2}")
    
    # Test du pattern matching
    test_string = "{'operator': 'lt', 'amount': 500, 'currency': 'EUR'}"
    print(f"\nTest pattern matching:")
    print(f"String: {test_string}")
    print(f"startswith('{{''): {test_string.startswith('{\'')}")
    print(f"endswith('\'}}'): {test_string.endswith('\'}')}")
    
    # Test ast.literal_eval
    print(f"\nTest ast.literal_eval:")
    try:
        import ast
        parsed = ast.literal_eval(test_string)
        print(f"Parsed: {parsed}")
        print(f"Type: {type(parsed)}")
        print(f"Has operator: {'operator' in parsed}")
        
        # Test conversion
        es_filter = engine._to_elasticsearch_amount_filter(parsed)
        print(f"ES Filter: {es_filter}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_conversion()