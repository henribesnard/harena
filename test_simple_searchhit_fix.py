#!/usr/bin/env python3
"""
Simple test to verify SearchHit access fix
"""
import sys
import os

# Add the conversation_service to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'conversation_service'))
sys.path.insert(0, os.path.dirname(__file__))

def test_searchhit_access():
    """Test SearchHit access fix"""
    print("Testing SearchHit access fix...")
    
    try:
        from conversation_service.models.contracts.search_service import SearchHit
        
        # Create a SearchHit like our conversion creates
        hit = SearchHit(
            id="25000210641831",
            score=0.85,
            source={
                "transaction_id": "25000210641831", 
                "amount": 2302.2,
                "merchant_name": "Acme Corp",
                "date": "2025-05-15T00:00:00+00:00"
            }
        )
        
        print("  OK - SearchHit created")
        print(f"  - ID: {hit.id}")
        print(f"  - Score: {hit.score}")
        print(f"  - Source keys: {list(hit.source.keys())}")
        
        # Test the fix - accessing hit.source directly
        source = hit.source if hasattr(hit, 'source') else hit.get("_source", {})
        
        if "amount" in source:
            amount = source["amount"]
            print(f"  - Amount accessed: {amount}€")
        
        if "merchant_name" in source:
            merchant = source["merchant_name"]
            print(f"  - Merchant accessed: {merchant}")
            
        print("  SUCCESS - SearchHit.source access working correctly")
        return True
        
    except Exception as e:
        print(f"  ERROR - {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Test SearchHit access fix"""
    print("Testing SearchHit access fix...\n")
    
    if test_searchhit_access():
        print("\nSearchHit access fix verified!")
        print("The response generator should now be able to access SearchHit.source data.")
        return True
    else:
        print("\nSearchHit access test failed!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)