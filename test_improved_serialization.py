#!/usr/bin/env python3
"""
Test de la sérialisation améliorée pour éliminer les valeurs null
"""
import sys
import json
from pathlib import Path

# Add the project to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from conversation_service.models.contracts.search_service import (
    SearchQuery, SearchFilters, DateRange, TextFilter, Aggregation, AggregationConfig, NestedAggregation
)

def test_improved_serialization():
    """Test que la sérialisation exclut les valeurs null"""
    
    print("Testing improved serialization...")
    
    # Créer une requête complexe similaire à celle du problème
    query = SearchQuery(
        user_id=34,
        query=None,  # Should become ""
        filters=SearchFilters(
            date=DateRange(
                gte="2025-05-01",
                lte="2025-05-31",
                gt=None,  # Should be excluded
                lt=None   # Should be excluded
            ),
            merchant_name=None,  # Should be excluded
            transaction_type="credit",
            account_id=None  # Should be excluded
        ),
        aggregations={
            "operation_stats": Aggregation(
                terms=AggregationConfig(
                    field="operation_type.keyword",
                    size=10
                ),
                sum=None,  # Should be excluded
                avg=None,  # Should be excluded
                aggs={
                    "operation_total": NestedAggregation(
                        sum={"field": "amount_abs"},
                        avg=None,  # Should be excluded
                        max=None   # Should be excluded
                    )
                }
            )
        },
        page_size=50
    )
    
    print("1. Testing dict() serialization:")
    dict_result = query.dict()
    print(f"   query field: {dict_result.get('query')!r}")
    print(f"   Contains null values: {'null' in json.dumps(dict_result)}")
    
    print("\n2. Testing dict(exclude_none=True):")
    dict_clean = query.dict(exclude_none=True)
    json_clean = json.dumps(dict_clean, indent=2)
    print(f"   query field: {dict_clean.get('query')!r}")
    print(f"   Contains null values: {'null' in json_clean}")
    print(f"   JSON size: {len(json_clean)} characters")
    
    print("\n3. Testing json() method:")
    json_result = query.json()
    print(f"   Contains null values: {'null' in json_result}")
    print(f"   Contains query empty string: '\"query\": \"\"' in json_result")
    
    print("\n4. Sample of clean JSON (first 500 chars):")
    print(json_clean[:500] + "..." if len(json_clean) > 500 else json_clean)
    
    print("\n5. Verification checks:")
    checks = [
        ("query is empty string", dict_clean.get('query') == ''),
        ("No null in JSON", 'null' not in json_clean),
        ("date filter has only gte/lte", set(dict_clean['filters']['date'].keys()) == {'gte', 'lte'}),
        ("No null aggregations", 'null' not in json.dumps(dict_clean.get('aggregations', {}))),
        ("merchant_name excluded", 'merchant_name' not in dict_clean['filters']),
        ("transaction_type included", dict_clean['filters']['transaction_type'] == 'credit')
    ]
    
    all_passed = True
    for check_name, result in checks:
        status = "PASS" if result else "FAIL"
        print(f"   {check_name}: {status}")
        if not result:
            all_passed = False
    
    return all_passed, json_clean

if __name__ == "__main__":
    success, clean_json = test_improved_serialization()
    print(f"\n{'='*50}")
    if success:
        print("SUCCESS: Improved serialization working!")
        print("- query: null -> query: ''")
        print("- Optional null fields excluded")
        print("- Clean JSON ready for search_service")
    else:
        print("FAIL: Serialization needs more work")
        
    print(f"\nFinal JSON ready for POST to search_service:")
    print(clean_json)