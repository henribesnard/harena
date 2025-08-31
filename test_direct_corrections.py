#!/usr/bin/env python3
"""
Test direct des corrections sans passer par le service web
"""
import sys
import json
from pathlib import Path

# Add the project to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from conversation_service.models.contracts.search_service import (
    SearchQuery, SearchFilters, DateRange, TextFilter, Aggregation, 
    AggregationConfig, NestedAggregation
)

def test_serialization_corrections():
    """Test direct de nos corrections de sérialisation"""
    
    print("Testing serialization corrections directly...")
    print("=" * 60)
    
    # Créer une requête similaire à celle qui échouait
    query = SearchQuery(
        user_id=34,
        query=None,  # Should become "" with our fix
        filters=SearchFilters(
            date=DateRange(
                gte="2025-05-01",
                lte="2025-05-31",
                gt=None,  # Should be excluded
                lt=None   # Should be excluded
            ),
            amount=None,  # Should be excluded entirely
            amount_abs=None,  # Should be excluded entirely
            merchant_name=None,  # Should be excluded entirely
            primary_description=None,  # Should be excluded entirely
            category_name=None,  # Should be excluded entirely
            operation_type=None,  # Should be excluded entirely
            transaction_type="credit",  # Should be included
            account_id=None,  # Should be excluded entirely
            account_type=None   # Should be excluded entirely
        ),
        aggregations={
            "operation_stats": Aggregation(
                terms=AggregationConfig(
                    field="operation_type.keyword",
                    size=10,
                    calendar_interval=None  # Should be excluded
                ),
                sum=None,  # Should be excluded
                avg=None,  # Should be excluded
                max=None,  # Should be excluded
                min=None,  # Should be excluded
                value_count=None,  # Should be excluded
                date_histogram=None,  # Should be excluded
                cardinality=None,  # Should be excluded
                aggs={
                    "operation_total": NestedAggregation(
                        sum={"field": "amount_abs"},
                        avg=None,  # Should be excluded
                        max=None,  # Should be excluded
                        min=None,  # Should be excluded
                        value_count=None,  # Should be excluded
                        cardinality=None   # Should be excluded
                    ),
                    "operation_count": NestedAggregation(
                        sum=None,  # Should be excluded
                        avg=None,  # Should be excluded
                        max=None,  # Should be excluded
                        min=None,  # Should be excluded
                        value_count={"field": "transaction_id"},
                        cardinality=None   # Should be excluded
                    ),
                    "avg_amount": NestedAggregation(
                        sum=None,  # Should be excluded
                        avg={"field": "amount_abs"},
                        max=None,  # Should be excluded
                        min=None,  # Should be excluded
                        value_count=None,  # Should be excluded
                        cardinality=None   # Should be excluded
                    )
                }
            ),
            "monthly_operations": Aggregation(
                terms=None,  # Should be excluded
                sum=None,  # Should be excluded
                avg=None,  # Should be excluded
                max=None,  # Should be excluded
                min=None,  # Should be excluded
                value_count=None,  # Should be excluded
                date_histogram=AggregationConfig(
                    field="date",
                    size=10,
                    calendar_interval="month"
                ),
                cardinality=None,  # Should be excluded
                aggs={
                    "monthly_count": NestedAggregation(
                        sum=None,  # Should be excluded
                        avg=None,  # Should be excluded
                        max=None,  # Should be excluded
                        min=None,  # Should be excluded
                        value_count={"field": "transaction_id"},
                        cardinality=None   # Should be excluded
                    ),
                    "monthly_total": NestedAggregation(
                        sum={"field": "amount_abs"},
                        avg=None,  # Should be excluded
                        max=None,  # Should be excluded
                        min=None,  # Should be excluded
                        value_count=None,  # Should be excluded
                        cardinality=None   # Should be excluded
                    )
                }
            )
        },
        sort=[{"date": {"order": "desc"}}],
        page_size=50,
        offset=0,
        include_fields=[
            "transaction_id",
            "amount", 
            "merchant_name",
            "date",
            "operation_type",
            "primary_description"
        ],
        exclude_fields=None,  # Should be excluded
        aggregation_only=False,
        explain=False
    )
    
    print("1. Testing default dict() serialization:")
    default_dict = query.dict()
    default_json = json.dumps(default_dict)
    print(f"   Contains null values: {'null' in default_json}")
    print(f"   Query field: {default_dict.get('query')!r}")
    print(f"   JSON size: {len(default_json)} characters")
    
    print("\n2. Testing exclude_none=True serialization:")
    clean_dict = query.dict(exclude_none=True)
    clean_json = json.dumps(clean_dict, indent=2)
    
    null_count = clean_json.count('null')
    print(f"   Contains null values: {'null' in clean_json}")
    print(f"   Null count: {null_count}")
    print(f"   Query field: {clean_dict.get('query')!r}")
    print(f"   JSON size: {len(clean_json)} characters")
    
    print("\n3. Testing json() method:")
    json_method_result = query.json()
    print(f"   Contains null values: {'null' in json_method_result}")
    print(f"   JSON size: {len(json_method_result)} characters")
    
    print("\n4. Sample of clean JSON (first 800 chars):")
    print(clean_json[:800] + "..." if len(clean_json) > 800 else clean_json)
    
    print("\n5. Key verification checks:")
    checks = [
        ("Query is empty string", clean_dict.get('query') == ''),
        ("No null values in clean JSON", 'null' not in clean_json),
        ("Filters exclude null fields", 'null' not in json.dumps(clean_dict.get('filters', {}))),
        ("transaction_type preserved", clean_dict['filters']['transaction_type'] == 'credit'),
        ("date gte/lte preserved", 'gte' in clean_dict['filters']['date'] and 'lte' in clean_dict['filters']['date']),
        ("date gt/lt excluded", 'gt' not in clean_dict['filters']['date'] and 'lt' not in clean_dict['filters']['date']),
        ("aggregations cleaned", 'null' not in json.dumps(clean_dict.get('aggregations', {}))),
        ("JSON much smaller", len(clean_json) < len(default_json) * 0.7)
    ]
    
    all_passed = True
    for check_name, passed in checks:
        status = "PASS" if passed else "FAIL"
        print(f"   {check_name}: {status}")
        if not passed:
            all_passed = False
    
    print(f"\n6. Summary:")
    print(f"   Original JSON size: {len(default_json)} chars")
    print(f"   Clean JSON size: {len(clean_json)} chars")
    print(f"   Size reduction: {(1 - len(clean_json)/len(default_json))*100:.1f}%")
    
    return all_passed, clean_dict

if __name__ == "__main__":
    success, clean_query = test_serialization_corrections()
    print(f"\n{'='*60}")
    if success:
        print("SUCCESS: All serialization corrections working!")
        print("- query: null -> query: \"\"")
        print("- Null optional fields excluded")  
        print("- JSON size optimized")
        print("- Ready for search_service API")
    else:
        print("FAIL: Some corrections not working properly")
        
    print(f"\nThis clean query should work with search_service:")
    print(f"Size: {len(json.dumps(clean_query))} characters")
    print(f"Query field: {clean_query.get('query')!r}")
    print(f"Contains null: {'null' in json.dumps(clean_query)}")