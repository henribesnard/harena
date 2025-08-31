#!/usr/bin/env python3
"""
Test complet de la correction du bug query=null
Simule le pipeline complet sans le service web
"""
import sys
import json
from pathlib import Path

# Add the project to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from conversation_service.utils.query_validator import QueryValidator
from conversation_service.models.contracts.search_service import (
    SearchQuery, SearchFilters, QueryGenerationRequest
)

def test_complete_fix():
    """Test complet du fix pour éviter les erreurs de validation"""
    
    print("Testing complete fix for query validation...")
    print("=" * 50)
    
    # Simuler une requête générée par le QueryBuilder
    # Avec le problème typique: query=None et user_id potentiellement dans filters
    test_query_data = {
        "user_id": 34,
        "query": None,  # <-- Problème 1: null au lieu de ""
        "filters": {
            "merchant_name": {"match": "Carrefour"},
            "date": {"gte": "2024-01-01", "lte": "2024-12-31"}
        },
        "page_size": 10,
        "aggregation_only": False
    }
    
    print("1. Original query data (before processing):")
    print(f"   user_id: {test_query_data.get('user_id')}")
    print(f"   query: {test_query_data.get('query')!r}")
    print(f"   filters: {test_query_data.get('filters')}")
    print()
    
    # Créer l'objet SearchQuery (avec fix de sérialisation)
    search_query = SearchQuery(**test_query_data)
    
    print("2. After SearchQuery object creation:")
    serialized = search_query.dict()
    print(f"   query: {serialized.get('query')!r}")
    print(f"   query is empty string: {serialized.get('query') == ''}")
    print()
    
    # Validation avec QueryValidator (avec fix user_id)
    validator = QueryValidator()
    validation_result = validator.validate_query(test_query_data)
    
    print("3. Validation result:")
    print(f"   schema_valid: {validation_result.schema_valid}")
    print(f"   contract_compliant: {validation_result.contract_compliant}")
    print(f"   errors: {validation_result.errors}")
    print(f"   warnings: {validation_result.warnings}")
    print()
    
    # Vérifier que user_id n'est pas dans les filtres après validation
    print("4. Query after validation (optimized):")
    print(f"   user_id at root: {test_query_data.get('user_id')}")
    print(f"   user_id in filters: {'user_id' in test_query_data.get('filters', {})}")
    print(f"   filters: {test_query_data.get('filters')}")
    print()
    
    # Test JSON final qui serait envoyé à search_service
    final_query = SearchQuery(**test_query_data)
    json_payload = final_query.dict()
    json_str = json.dumps(json_payload)
    
    print("5. Final JSON payload for search_service:")
    print(f"   Contains 'query': null: {'\"query\": null' in json_str}")
    print(f"   Contains 'query': '': {'\"query\": \"\"' in json_str}")
    print(f"   Contains user_id in filters: {'\"user_id\"' in json.dumps(json_payload.get('filters', {}))}")
    print()
    
    # Vérifications finales
    success_checks = [
        ("Schema validation passes", validation_result.schema_valid),
        ("Contract validation passes", validation_result.contract_compliant),
        ("Query is not null", '\"query\": null' not in json_str),
        ("Query is empty string", '\"query\": \"\"' in json_str),
        ("user_id not in filters", 'user_id' not in json_payload.get('filters', {})),
        ("user_id at root level", json_payload.get('user_id') == 34),
        ("No validation errors", len(validation_result.errors) == 0)
    ]
    
    print("6. Success checks:")
    all_passed = True
    for check_name, result in success_checks:
        status = "PASS" if result else "FAIL"
        print(f"   {check_name}: {status}")
        if not result:
            all_passed = False
    print()
    
    if all_passed:
        print("✅ [OVERALL SUCCESS] Complete fix working!")
        print("   - query=null → query='' (no 422 errors)")
        print("   - user_id not in filters (no validation errors)")
        print("   - Schema and contract validation pass")
        return True
    else:
        print("❌ [OVERALL FAIL] Fix needs more work")
        return False

if __name__ == "__main__":
    success = test_complete_fix()
    print()
    if success:
        print("🎉 Ready for production testing with search_service!")
    else:
        print("🔧 Needs additional fixes before testing")