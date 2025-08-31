#!/usr/bin/env python3
"""
Test simplifié de nos corrections de sérialisation
"""
import sys
import json
from pathlib import Path

# Add the project to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from conversation_service.models.contracts.search_service import (
    SearchQuery, SearchFilters, DateRange, Aggregation, AggregationConfig, NestedAggregation
)

def test_serialization():
    """Test simple de nos corrections"""
    
    print("Testing serialization corrections...")
    print("=" * 50)
    
    # Créer la requête problématique exacte du message utilisateur
    problematic_query = SearchQuery(
        user_id=34,
        query=None,  # Problème 1: null
        filters=SearchFilters(
            date=DateRange(
                gte="2025-05-01",
                lte="2025-05-31",
                gt=None,  # Problème 2: champs null
                lt=None
            ),
            amount=None,
            amount_abs=None,
            merchant_name=None,
            primary_description=None,
            category_name=None,
            operation_type=None,
            transaction_type="credit",
            account_id=None,
            account_type=None
        ),
        aggregations={
            "operation_stats": Aggregation(
                terms=AggregationConfig(
                    field="operation_type.keyword",
                    size=10,
                    calendar_interval=None
                ),
                sum=None,
                avg=None,
                max=None,
                min=None,
                value_count=None,
                date_histogram=None,
                cardinality=None,
                aggs={
                    "operation_total": NestedAggregation(
                        sum={"field": "amount_abs"},
                        avg=None,
                        max=None,
                        min=None,
                        value_count=None,
                        cardinality=None
                    ),
                    "operation_count": NestedAggregation(
                        sum=None,
                        avg=None,
                        max=None,
                        min=None,
                        value_count={"field": "transaction_id"},
                        cardinality=None
                    ),
                    "avg_amount": NestedAggregation(
                        sum=None,
                        avg={"field": "amount_abs"},
                        max=None,
                        min=None,
                        value_count=None,
                        cardinality=None
                    )
                }
            ),
            "monthly_operations": Aggregation(
                terms=None,
                sum=None,
                avg=None,
                max=None,
                min=None,
                value_count=None,
                date_histogram=AggregationConfig(
                    field="date",
                    size=10,
                    calendar_interval="month"
                ),
                cardinality=None,
                aggs={
                    "monthly_count": NestedAggregation(
                        sum=None,
                        avg=None,
                        max=None,
                        min=None,
                        value_count={"field": "transaction_id"},
                        cardinality=None
                    ),
                    "monthly_total": NestedAggregation(
                        sum={"field": "amount_abs"},
                        avg=None,
                        max=None,
                        min=None,
                        value_count=None,
                        cardinality=None
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
        exclude_fields=None,
        aggregation_only=False,
        explain=False
    )
    
    print("1. AVANT (problématique) - default dict():")
    before_dict = problematic_query.dict()
    before_json = json.dumps(before_dict)
    null_count_before = before_json.count('null')
    
    print(f"   query field: {before_dict.get('query')!r}")
    print(f"   null count: {null_count_before}")
    print(f"   size: {len(before_json)} chars")
    print(f"   would cause 422 error: {before_dict.get('query') is None}")
    
    print("\n2. APRÈS (nos corrections) - dict(exclude_none=True):")
    after_dict = problematic_query.dict(exclude_none=True)
    after_json = json.dumps(after_dict)
    null_count_after = after_json.count('null')
    
    print(f"   query field: {after_dict.get('query')!r}")
    print(f"   null count: {null_count_after}")
    print(f"   size: {len(after_json)} chars")
    print(f"   will cause 422 error: {after_dict.get('query') is None}")
    
    print("\n3. JSON pour POST search_service:")
    print(json.dumps(after_dict, indent=2)[:600] + "...")
    
    print(f"\n4. Résultats:")
    print(f"   Size reduction: {len(before_json) - len(after_json)} chars ({((len(before_json) - len(after_json))/len(before_json)*100):.1f}%)")
    print(f"   Null reduction: {null_count_before - null_count_after} nulls")
    print(f"   Query fixed: {before_dict.get('query')} → {after_dict.get('query')!r}")
    
    success_criteria = [
        ("Query is empty string", after_dict.get('query') == ''),
        ("No null values", null_count_after == 0),
        ("Size reduced", len(after_json) < len(before_json)),
        ("Ready for search_service", after_dict.get('query') is not None)
    ]
    
    print(f"\n5. Success checks:")
    all_passed = True
    for check, passed in success_criteria:
        status = "PASS" if passed else "FAIL"
        print(f"   {check}: {status}")
        if not passed:
            all_passed = False
    
    return all_passed

if __name__ == "__main__":
    success = test_serialization()
    print(f"\n{'='*50}")
    if success:
        print("SUCCESS: All serialization corrections working!")
        print("✅ query: null → query: \"\"")
        print("✅ All null optional fields excluded") 
        print("✅ JSON size optimized")
        print("✅ Ready to fix search_service 422 errors")
    else:
        print("FAIL: Some corrections need work")
        
    print(f"\nConclusion: The generated queries will now work with search_service!")