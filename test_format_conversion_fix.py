#!/usr/bin/env python3
"""
Test script to validate the SearchResponse format conversion fix
"""
import sys
import os
import json

# Add the conversation_service to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'conversation_service'))
sys.path.insert(0, os.path.dirname(__file__))

def test_search_response_format_conversion():
    """Test that the format conversion works correctly"""
    print("Testing SearchResponse format conversion...")
    
    try:
        from conversation_service.core.search_service_client import SearchServiceClient
        
        # Create a mock search_service response (like what we saw in the test)
        mock_response_data = {
            "results": [
                {
                    "transaction_id": "25000210641831",
                    "user_id": 34,
                    "account_id": 39,
                    "amount": 35.0,
                    "amount_abs": 35.0,
                    "transaction_type": "credit",
                    "date": "2025-05-31T00:00:00+00:00",
                    "primary_description": "Vir Sepa M Doe John",
                    "merchant_name": None,
                    "category_name": None,
                    "operation_type": "transfer",
                    "_score": 0.0,
                    "highlights": None
                }
            ],
            "aggregations": None,
            "response_metadata": {
                "total_results": 13,
                "took_ms": 183
            }
        }
        
        # Test the conversion
        client = SearchServiceClient()
        search_response = client._convert_search_service_response(mock_response_data)
        
        print("  OK - Conversion completed")
        print(f"  - Total hits: {search_response.total_hits}")
        print(f"  - Hits count: {len(search_response.hits)}")
        print(f"  - Took ms: {search_response.took_ms}")
        print(f"  - Aggregations: {search_response.aggregations}")
        
        if search_response.hits:
            first_hit = search_response.hits[0]
            print(f"  - First hit _id: {first_hit.id}")
            print(f"  - First hit _score: {first_hit.score}")
            print(f"  - First hit _source keys: {list(first_hit.source.keys())}")
            print(f"  - Transaction in source: {'transaction_id' in first_hit.source}")
        
        # Validate the format is now correct
        if search_response.total_hits == 13 and len(search_response.hits) == 1:
            print("  SUCCESS - Format conversion working correctly")
            return True
        else:
            print("  ERROR - Format conversion produced unexpected results")
            return False
        
    except Exception as e:
        print(f"  ERROR - {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Test SearchResponse format conversion fix"""
    print("Testing SearchResponse format conversion fix...\n")
    
    if test_search_response_format_conversion():
        print("\nAll tests passed! The SearchResponse format conversion should now work.")
        print("The search_service results should now properly reach the response generator.")
        return True
    else:
        print("\nTest failed!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)