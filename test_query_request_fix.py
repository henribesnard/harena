#!/usr/bin/env python3
"""
Test the QueryGenerationRequest fix specifically
"""
import sys
import os

# Add the conversation_service to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'conversation_service'))
sys.path.insert(0, os.path.dirname(__file__))

from conversation_service.models.contracts.search_service import QueryGenerationRequest

def test_query_request_fix():
    """Test that QueryGenerationRequest works with the correct parameters"""
    
    print("Testing QueryGenerationRequest with correct parameters...")
    
    try:
        # Test data matching the Phase 5 implementation
        intent_dict = {
            "intent_type": "SEARCH_BY_MERCHANT",
            "confidence": 0.95,
            "reasoning": "User wants to know spending at a specific merchant",
            "original_message": "Mes rentrees d'argent en mai ?",
            "category": "spending_analysis",
            "is_supported": True
        }
        
        entities_result = {
            "merchants": ["Carrefour"],
            "dates": {
                "period": "current_month",
                "normalized": {
                    "gte": "2025-05-01",
                    "lte": "2025-05-31"
                }
            },
            "confidence": 0.9
        }
        
        # Create QueryGenerationRequest as done in Phase 5
        query_request = QueryGenerationRequest(
            user_message="Mes rentrees d'argent en mai ?",
            user_id=34,
            intent_type=intent_dict["intent_type"],
            intent_confidence=intent_dict["confidence"],
            entities=entities_result
        )
        
        print("OK - QueryGenerationRequest created successfully")
        print(f"  Intent type: {query_request.intent_type}")
        print(f"  Intent confidence: {query_request.intent_confidence}")
        print(f"  User ID: {query_request.user_id}")
        print(f"  User message: {query_request.user_message}")
        print(f"  Entities keys: {list(query_request.entities.keys())}")
        
        # Verify all required fields are present
        assert query_request.intent_type == "SEARCH_BY_MERCHANT"
        assert query_request.intent_confidence == 0.95
        assert query_request.user_id == 34
        assert query_request.user_message == "Mes rentrees d'argent en mai ?"
        assert "merchants" in query_request.entities
        assert "dates" in query_request.entities
        
        print("OK - All validation checks passed")
        
        # Test that the old broken way would fail
        print("\nTesting that the old broken approach would fail...")
        try:
            # This should fail - passing intent dict instead of individual fields
            broken_request = QueryGenerationRequest(
                user_message="Test message",
                intent=intent_dict,  # Wrong - should be intent_type and intent_confidence
                entities=entities_result,
                user_id=34,
                request_id="test123"  # Wrong - not a valid field
            )
            print("ERROR - Broken request should have failed but didn't!")
            return False
        except Exception as e:
            print(f"OK - Broken request correctly failed: {type(e).__name__}")
        
        print("\nSUCCESS - QueryGenerationRequest fix validation passed!")
        print("Phase 5 should now work correctly with the query generation step")
        return True
        
    except Exception as e:
        print(f"ERROR - QueryGenerationRequest test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = test_query_request_fix()
    sys.exit(0 if result else 1)