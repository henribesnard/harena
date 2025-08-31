#!/usr/bin/env python3
"""
Direct test of QueryValidator to verify the fix
"""
import sys
import os
from pathlib import Path

# Add the project to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from conversation_service.utils.query_validator import QueryValidator

def test_query_validator_fix():
    """Test that the QueryValidator no longer adds user_id to filters"""
    
    print("Testing QueryValidator fix...")
    
    # Create a test query that would previously fail
    test_query = {
        "user_id": 34,
        "filters": {
            "merchant_name": {"match": "Carrefour"},
            "date": {"gte": "2024-01-01", "lte": "2024-12-31"}
        },
        "aggregation_only": False,
        "page_size": 10
    }
    
    print("Original query:")
    print(f"  user_id: {test_query.get('user_id')}")
    print(f"  filters: {test_query.get('filters')}")
    print()
    
    # Initialize validator
    validator = QueryValidator()
    
    # Validate the query
    try:
        validation_result = validator.validate_query(test_query)
        
        print("Validation Result:")
        print(f"  schema_valid: {validation_result.schema_valid}")
        print(f"  contract_compliant: {validation_result.contract_compliant}")
        print(f"  errors: {validation_result.errors}")
        print(f"  warnings: {validation_result.warnings}")
        print(f"  optimizations: {validation_result.optimization_applied}")
        print()
        
        # Check the optimized query (query_dict is modified in-place)
        print("Query after validation (optimized):")
        print(f"  user_id: {test_query.get('user_id')}")
        print(f"  filters: {test_query.get('filters')}")
        print()
        
        # Check if user_id is in filters (it shouldn't be)
        filters = test_query.get('filters', {})
        if 'user_id' in filters:
            print("[FAIL] user_id is still in filters - bug not fixed!")
            return False
        else:
            print("[SUCCESS] user_id is NOT in filters - fix working!")
            
        # Check if user_id is at root level (it should be)
        if test_query.get('user_id') == 34:
            print("[SUCCESS] user_id is correctly at root level")
            
            # Final success if schema and contract validation passed
            if validation_result.schema_valid and validation_result.contract_compliant:
                print("[SUCCESS] Query passes all validations!")
                return True
            else:
                print("[FAIL] Query validation failed")
                return False
        else:
            print("[FAIL] user_id missing at root level")
            return False
            
    except Exception as e:
        print(f"[ERROR] Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_query_validator_fix()
    if success:
        print("\n[OVERALL SUCCESS] Query validation fix is working!")
    else:
        print("\n[OVERALL FAIL] Query validation fix needs more work")