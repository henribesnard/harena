#!/usr/bin/env python3
"""
Test de la correction des agrégations date_histogram
"""
import sys
import json
from pathlib import Path

# Add the project to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from conversation_service.models.contracts.search_service import (
    SearchQuery, SearchFilters, DateRange, Aggregation, 
    TermsAggregationConfig, DateHistogramAggregationConfig, NestedAggregation
)

def test_aggregation_fix():
    """Test que les agrégations ne contiennent plus size dans date_histogram"""
    
    print("Testing aggregation fix for date_histogram...")
    print("=" * 50)
    
    # Créer une requête avec les nouvelles configurations
    query = SearchQuery(
        user_id=34,
        query="",
        filters=SearchFilters(
            date=DateRange(gte="2025-05-01", lte="2025-05-31"),
            transaction_type="credit"
        ),
        aggregations={
            "operation_stats": Aggregation(
                terms=TermsAggregationConfig(
                    field="operation_type.keyword",
                    size=10
                ),
                aggs={
                    "operation_total": NestedAggregation(
                        sum={"field": "amount_abs"}
                    )
                }
            ),
            "monthly_operations": Aggregation(
                date_histogram=DateHistogramAggregationConfig(
                    field="date",
                    calendar_interval="month"
                ),
                aggs={
                    "monthly_count": NestedAggregation(
                        value_count={"field": "transaction_id"}
                    )
                }
            )
        }
    )
    
    # Sérialiser avec notre fix
    clean_dict = query.dict(exclude_none=True)
    clean_json = json.dumps(clean_dict, indent=2)
    
    print("Generated query:")
    print(clean_json)
    
    # Vérifications
    operation_stats = clean_dict["aggregations"]["operation_stats"]
    monthly_operations = clean_dict["aggregations"]["monthly_operations"]
    
    checks = [
        ("Terms has size field", "size" in operation_stats.get("terms", {})),
        ("Date histogram has calendar_interval", "calendar_interval" in monthly_operations.get("date_histogram", {})),
        ("Date histogram DOES NOT have size", "size" not in monthly_operations.get("date_histogram", {})),
        ("Query field is empty string", clean_dict.get("query") == ""),
        ("No null values", "null" not in clean_json)
    ]
    
    print("\nValidation checks:")
    all_passed = True
    for check_name, passed in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  {check_name}: {status}")
        if not passed:
            all_passed = False
    
    return all_passed, clean_dict

if __name__ == "__main__":
    success, query_dict = test_aggregation_fix()
    print(f"\n{'='*50}")
    if success:
        print("SUCCESS: Aggregation fix working!")
        print("✅ date_histogram no longer has 'size' field")
        print("✅ terms aggregation still has 'size' field")
        print("✅ Should fix Elasticsearch 400 error")
    else:
        print("FAIL: Aggregation fix needs work")
        
    print(f"\nThis query should now work with Elasticsearch!")