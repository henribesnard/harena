#!/usr/bin/env python3
"""
Test script to validate the SearchResponse aggregations fix
"""
import sys
import os
from datetime import datetime, timezone

# Add the conversation_service to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'conversation_service'))
sys.path.insert(0, os.path.dirname(__file__))

def test_search_response_with_null_aggregations():
    """Test that SearchResponse accepts None for aggregations field"""
    print("Testing SearchResponse with null aggregations...")
    
    try:
        from conversation_service.models.contracts.search_service import SearchResponse
        
        # Test creating SearchResponse with null aggregations (as returned by search_service)
        response = SearchResponse(
            hits=[],
            total_hits=13,
            aggregations=None,  # This should now work
            took_ms=192,
            query_id="test_query",
            timestamp=datetime.now(timezone.utc)
        )
        
        print("  OK - SearchResponse created with aggregations=None")
        print(f"  - Total hits: {response.total_hits}")
        print(f"  - Aggregations: {response.aggregations}")
        print(f"  - Has aggregations: {bool(response.aggregations)}")
        
        # Test creating SearchResponse with dict aggregations
        response_with_aggs = SearchResponse(
            hits=[],
            total_hits=5,
            aggregations={"test_agg": {"value": 100}},
            took_ms=150,
            query_id="test_query_2"
        )
        
        print("  OK - SearchResponse created with aggregations=dict")
        print(f"  - Has aggregations: {bool(response_with_aggs.aggregations)}")
        print(f"  - Aggregations count: {len(response_with_aggs.aggregations) if response_with_aggs.aggregations else 0}")
        
        return True
        
    except Exception as e:
        print(f"  ERROR - {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_phase4_metrics_with_null_aggregations():
    """Test that Phase4 response metrics handle None aggregations"""
    print("\nTesting Phase4 metrics with null aggregations...")
    
    try:
        from conversation_service.models.contracts.search_service import SearchResponse
        from conversation_service.agents.search.search_executor import SearchExecutorResponse
        
        # Create a SearchResponse with null aggregations
        search_response = SearchResponse(
            hits=[],
            total_hits=13,
            aggregations=None,
            took_ms=192
        )
        
        # Create a SearchExecutorResponse 
        executor_response = SearchExecutorResponse(
            success=True,
            search_results=search_response,
            execution_time_ms=500,
            request_id="test_request"
        )
        
        # Test the metrics creation that was failing
        has_aggs = bool(executor_response.search_results.aggregations)
        aggs_count = len(executor_response.search_results.aggregations) if executor_response.search_results.aggregations else 0
        
        print(f"  OK - Has aggregations: {has_aggs}")
        print(f"  OK - Aggregations count: {aggs_count}")
        
        return True
        
    except Exception as e:
        print(f"  ERROR - {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Test SearchResponse aggregations fix"""
    print("Testing SearchResponse aggregations fix...\n")
    
    results = []
    results.append(test_search_response_with_null_aggregations())
    results.append(test_phase4_metrics_with_null_aggregations())
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"\nAll tests passed! ({passed}/{total})")
        print("SearchResponse should now accept null aggregations from search_service.")
        return True
    else:
        print(f"\nSome tests failed: {total - passed} failures out of {total}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)