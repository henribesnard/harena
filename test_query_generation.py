#!/usr/bin/env python3
"""
Test direct de génération de requête pour vérifier la correction query=null
"""
import sys
import os
import json
from pathlib import Path

# Add the project to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from conversation_service.models.contracts.search_service import SearchQuery, SearchFilters

def test_query_serialization():
    """Test que SearchQuery sérialise query=null en query=''"""
    
    print("Testing SearchQuery serialization fix...")
    
    # Test 1: Création avec query=None explicite
    query1 = SearchQuery(
        user_id=34,
        query=None,
        filters=SearchFilters(
            merchant_name={"match": "Carrefour"}
        )
    )
    
    print("Query 1 (query=None explicit):")
    serialized1 = query1.dict()
    print(f"  Serialized query field: {serialized1.get('query')!r}")
    print(f"  Is empty string: {serialized1.get('query') == ''}")
    print()
    
    # Test 2: Création sans spécifier query (défaut=None)
    query2 = SearchQuery(
        user_id=34,
        filters=SearchFilters(
            merchant_name={"match": "Carrefour"}
        )
    )
    
    print("Query 2 (query not specified, defaults to None):")
    serialized2 = query2.dict()
    print(f"  Serialized query field: {serialized2.get('query')!r}")
    print(f"  Is empty string: {serialized2.get('query') == ''}")
    print()
    
    # Test 3: JSON serialization (ce qui sera envoyé à l'API)
    print("JSON serialization test:")
    json_str = json.dumps(serialized2)
    print(f"  JSON: {json_str}")
    print(f"  Contains 'query': '': {'\"query\": \"\"' in json_str}")
    print(f"  Contains 'query': null: {'\"query\": null' in json_str}")
    print()
    
    # Test 4: Query avec vraie valeur
    query3 = SearchQuery(
        user_id=34,
        query="Carrefour",
        filters=SearchFilters()
    )
    
    print("Query 3 (query='Carrefour'):")
    serialized3 = query3.dict()
    print(f"  Serialized query field: {serialized3.get('query')!r}")
    print(f"  Query value preserved: {serialized3.get('query') == 'Carrefour'}")
    print()
    
    # Verification finale
    success = (
        serialized1.get('query') == '' and
        serialized2.get('query') == '' and
        serialized3.get('query') == 'Carrefour' and
        '\"query\": null' not in json.dumps(serialized1) and
        '\"query\": null' not in json.dumps(serialized2)
    )
    
    return success

if __name__ == "__main__":
    success = test_query_serialization()
    if success:
        print("[OVERALL SUCCESS] SearchQuery serialization fix working!")
    else:
        print("[OVERALL FAIL] SearchQuery serialization needs more work")