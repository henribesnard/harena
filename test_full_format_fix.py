#!/usr/bin/env python3
"""
Test script to validate the complete SearchResponse format fix with real data structure
"""
import sys
import os
import asyncio
import json

# Add the conversation_service to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'conversation_service'))
sys.path.insert(0, os.path.dirname(__file__))

def test_full_format_conversion():
    """Test the complete format conversion with realistic data"""
    print("Testing complete format conversion with realistic data...")
    
    try:
        from conversation_service.core.search_service_client import SearchServiceClient
        from conversation_service.models.contracts.search_service import SearchQuery, SearchFilters, DateRange
        
        # Create realistic search_service response (based on logs)
        search_service_response = {
            "results": [
                {
                    "transaction_id": "25000210641831",
                    "user_id": 34,
                    "account_id": 39,
                    "account_name": "Compte Courant 1",
                    "account_type": "checking",
                    "account_balance": 3588.76,
                    "account_currency": "EUR",
                    "amount": 2302.2,
                    "amount_abs": 2302.2,
                    "currency_code": "EUR",
                    "transaction_type": "credit",
                    "date": "2025-05-15T00:00:00+00:00",
                    "month_year": "2025-05",
                    "weekday": "Thursday",
                    "primary_description": "Salaire Acme Corp",
                    "merchant_name": "Acme Corp",
                    "category_name": "Salary",
                    "operation_type": "transfer",
                    "_score": 0.85,
                    "highlights": None
                },
                {
                    "transaction_id": "25000210641832",
                    "user_id": 34,
                    "account_id": 39,
                    "account_name": "Compte Courant 1",
                    "account_type": "checking",
                    "account_balance": 3588.76,
                    "account_currency": "EUR",
                    "amount": 994.14,
                    "amount_abs": 994.14,
                    "currency_code": "EUR",
                    "transaction_type": "credit",
                    "date": "2025-05-10T00:00:00+00:00",
                    "month_year": "2025-05",
                    "weekday": "Saturday",
                    "primary_description": "Allocation Pole Emploi",
                    "merchant_name": "Pôle Emploi",
                    "category_name": "Government_Benefits",
                    "operation_type": "transfer",
                    "_score": 0.75,
                    "highlights": None
                }
            ],
            "aggregations": None,
            "success": True,
            "error_message": None,
            "response_metadata": {
                "total_results": 13,
                "took_ms": 183,
                "query_id": "phase5_1756659394222_34"
            }
        }
        
        # Test the conversion
        client = SearchServiceClient()
        search_response = client._convert_search_service_response(search_service_response)
        
        print(f"  OK - Conversion completed")
        print(f"  - Total hits: {search_response.total_hits}")
        print(f"  - Hits count: {len(search_response.hits)}")
        print(f"  - Took ms: {search_response.took_ms}")
        print(f"  - Has aggregations: {search_response.aggregations is not None}")
        
        # Validate first transaction
        if search_response.hits:
            first_hit = search_response.hits[0]
            print(f"  - First hit _id: {first_hit.id}")
            print(f"  - First hit _score: {first_hit.score}")
            print(f"  - Amount in source: {first_hit.source.get('amount')}")
            print(f"  - Merchant in source: {first_hit.source.get('merchant_name')}")
            print(f"  - Description: {first_hit.source.get('primary_description')}")
            
            # Validate second transaction
            if len(search_response.hits) > 1:
                second_hit = search_response.hits[1]
                print(f"  - Second hit amount: {second_hit.source.get('amount')}")
                print(f"  - Second hit merchant: {second_hit.source.get('merchant_name')}")
        
        # Verify critical data is preserved
        if (search_response.total_hits == 13 and 
            len(search_response.hits) == 2 and
            search_response.hits[0].source.get('amount') == 2302.2 and
            search_response.hits[1].source.get('amount') == 994.14):
            
            print("  SUCCESS - Format conversion preserving all transaction data correctly")
            return True
        else:
            print("  ERROR - Format conversion not preserving data correctly")
            return False
        
    except Exception as e:
        print(f"  ERROR - {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_response_generator_data_access():
    """Test that response generator can access the converted data"""
    print("\nTesting response generator data access...")
    
    try:
        from conversation_service.core.search_service_client import SearchServiceClient
        
        # Sample converted response
        search_service_response = {
            "results": [
                {
                    "transaction_id": "25000210641831",
                    "user_id": 34,
                    "amount": 2302.2,
                    "transaction_type": "credit",
                    "date": "2025-05-15T00:00:00+00:00",
                    "primary_description": "Salaire Acme Corp",
                    "merchant_name": "Acme Corp",
                    "_score": 0.85
                }
            ],
            "aggregations": None,
            "response_metadata": {
                "total_results": 1,
                "took_ms": 183
            }
        }
        
        client = SearchServiceClient()
        search_response = client._convert_search_service_response(search_service_response)
        
        # Simulate what response generator would do
        if search_response.hits:
            # Access transaction data like response generator would
            transactions = []
            for hit in search_response.hits:
                transaction = {
                    "id": hit.id,
                    "amount": hit.source.get("amount"),
                    "type": hit.source.get("transaction_type"), 
                    "description": hit.source.get("primary_description"),
                    "merchant": hit.source.get("merchant_name"),
                    "date": hit.source.get("date")
                }
                transactions.append(transaction)
            
            print(f"  OK - Response generator can access {len(transactions)} transactions")
            print(f"  - Transaction 1: {transactions[0]['amount']}€ from {transactions[0]['merchant']}")
            
            # Simulate generating contextual response
            total_amount = sum(t["amount"] for t in transactions if t["amount"])
            print(f"  - Total amount available for analysis: {total_amount}€")
            
            if total_amount > 0:
                print("  SUCCESS - Response generator can now analyze real transaction data")
                return True
            else:
                print("  ERROR - No transaction amounts available")
                return False
        else:
            print("  ERROR - No hits available for response generator")
            return False
            
    except Exception as e:
        print(f"  ERROR - {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Test complete SearchResponse format fix"""
    print("Testing complete SearchResponse format fix...\n")
    
    results = []
    results.append(test_full_format_conversion())
    results.append(test_response_generator_data_access())
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"\nAll tests passed! ({passed}/{total})")
        print("The SearchResponse format fix should now allow the response generator")
        print("to access real transaction data and generate contextual responses.")
        print("\nExpected behavior:")
        print("- Search service finds 13 results (salaire Acme 2302.2€, Pôle Emploi 994.14€, etc.)")
        print("- SearchExecutor now receives 13 results instead of 0")
        print("- Response generator can analyze actual financial data")
        print("- Instead of generic response, user gets: 'En mai, vos rentrées incluent...'")
        return True
    else:
        print(f"\nSome tests failed: {total - passed} failures out of {total}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)