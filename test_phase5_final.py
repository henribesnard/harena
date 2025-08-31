#!/usr/bin/env python3
"""
Final test for Phase 5 complete response generation
"""
import sys
import os
import asyncio
from unittest.mock import Mock, patch

# Add the conversation_service to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'conversation_service'))
sys.path.insert(0, os.path.dirname(__file__))

from conversation_service.models.requests.conversation_requests import ConversationRequest
from conversation_service.api.routes.conversation_phase5 import get_dependencies

async def test_response_generation():
    """Test the complete response generation process"""
    
    print("Testing complete Phase 5 response generation...")
    
    try:
        # Initialize dependencies
        deps = get_dependencies()
        agents = deps["agents"]
        personalization_engine = deps["personalization"]
        
        # Test data
        request = ConversationRequest(message="Combien j'ai depense chez Carrefour ce mois-ci ?")
        user_id = 34
        request_id = "test_final_12345"
        
        # Get personalization context
        personalization_context = personalization_engine.get_personalization_context(user_id)
        print(f"OK - Personalization context: {type(personalization_context)}")
        
        # Mock intent data
        intent_dict = {
            "intent_type": "SEARCH_BY_MERCHANT",
            "confidence": 0.95,
            "reasoning": "User wants to know spending at a specific merchant",
            "original_message": request.message,
            "category": "spending_analysis",
            "is_supported": True
        }
        
        # Mock entities data
        entities_result = {
            "merchants": ["Carrefour"],
            "dates": {
                "period": "current_month",
                "normalized": {
                    "gte": "2025-08-01",
                    "lte": "2025-08-31"
                }
            },
            "confidence": 0.9
        }
        
        # Mock search results
        mock_search_response = Mock()
        mock_search_response.hits = [
            {"_source": {"merchant_name": "Carrefour", "amount": -45.67, "date": "2025-08-15"}},
            {"_source": {"merchant_name": "Carrefour", "amount": -23.45, "date": "2025-08-10"}},
            {"_source": {"merchant_name": "Carrefour", "amount": -67.89, "date": "2025-08-05"}}
        ]
        mock_search_response.total_hits = 3
        mock_search_response.aggregations = {
            "total_spent": {"value": -137.01},
            "merchant_analysis": {
                "buckets": [{
                    "key": "Carrefour",
                    "total_spent": {"value": -137.01},
                    "transaction_count": {"value": 3}
                }]
            }
        }
        
        print("Step 1: Testing response generation with full data...")
        
        # Get response generator
        response_generator = agents["response_generator"]
        
        # Mock the DeepSeek client call to avoid actual API calls
        with patch.object(response_generator.deepseek_client, 'chat_completion') as mock_chat:
            mock_chat.return_value = {
                "choices": [{
                    "message": {
                        "content": "D'apres mes calculs, vous avez depense 137,01 euros chez Carrefour ce mois-ci, repartis sur 3 transactions. Votre derniere visite remonte au 15 aout pour 45,67 euros. Cette frequence d'achat chez Carrefour suggere une habitude reguliere d'approvisionnement alimentaire."
                    }
                }]
            }
            
            # Test response generation
            response_content, response_quality, generation_metrics = await response_generator.generate_response(
                user_message=request.message,
                intent=intent_dict,
                entities=entities_result,
                search_results=mock_search_response,
                user_context=personalization_context,
                request_id=request_id
            )
        
        print(f"OK - Response generated:")
        print(f"  Message: {len(response_content.message)} characters")
        print(f"  Insights: {len(response_content.insights)} items")
        print(f"  Suggestions: {len(response_content.suggestions)} items")
        print(f"  Next actions: {len(response_content.next_actions)} items")
        print(f"  Quality score: {response_quality.relevance_score}")
        print(f"  Generation time: {generation_metrics.generation_time_ms}ms")
        
        # Display response content
        print("\nStep 2: Generated response content:")
        print(f"Message: {response_content.message}")
        
        if response_content.structured_data:
            print(f"Structured data: {response_content.structured_data.analysis_type}")
            print(f"  Total amount: {response_content.structured_data.total_amount}")
            print(f"  Transaction count: {response_content.structured_data.transaction_count}")
        
        if response_content.insights:
            print("Insights:")
            for i, insight in enumerate(response_content.insights, 1):
                print(f"  {i}. [{insight.severity}] {insight.title}: {insight.description}")
        
        if response_content.suggestions:
            print("Suggestions:")
            for i, suggestion in enumerate(response_content.suggestions, 1):
                print(f"  {i}. [{suggestion.priority}] {suggestion.title}: {suggestion.description}")
        
        if response_content.next_actions:
            print("Next actions:")
            for i, action in enumerate(response_content.next_actions, 1):
                print(f"  {i}. {action}")
        
        # Verify response completeness
        print("\nStep 3: Validating response completeness...")
        
        assert response_content.message is not None, "Response message should not be None"
        assert len(response_content.message) > 50, "Response message should be substantial"
        assert response_content.insights is not None, "Insights should not be None"
        assert response_content.suggestions is not None, "Suggestions should not be None"
        assert response_content.next_actions is not None, "Next actions should not be None"
        assert response_quality.relevance_score > 0.5, "Quality score should be reasonable"
        assert response_quality.completeness in ["minimal", "partial", "full"], "Completeness should be valid"
        assert response_quality.actionability in ["none", "low", "medium", "high"], "Actionability should be valid"
        
        print("OK - All response validation checks passed")
        
        # Test response without search results (fallback)
        print("\nStep 4: Testing fallback response generation...")
        
        with patch.object(response_generator.deepseek_client, 'chat_completion') as mock_chat_fallback:
            mock_chat_fallback.return_value = {
                "choices": [{
                    "message": {
                        "content": "Je n'ai pas pu recuperer vos donnees de transactions chez Carrefour actuellement. Veuillez reessayer dans quelques instants."
                    }
                }]
            }
            
            fallback_response, fallback_quality, fallback_metrics = await response_generator.generate_response(
                user_message=request.message,
                intent=intent_dict,
                entities=entities_result,
                search_results=None,  # No search results
                user_context=personalization_context,
                request_id=request_id
            )
        
        print(f"OK - Fallback response generated:")
        print(f"  Message: {len(fallback_response.message)} characters")
        print(f"  Insights: {len(fallback_response.insights)} items")
        print(f"  Suggestions: {len(fallback_response.suggestions)} items")
        print(f"  Quality score: {fallback_quality.relevance_score}")
        
        assert fallback_response.message is not None, "Fallback message should not be None"
        assert len(fallback_response.message) > 0, "Fallback message should not be empty"
        
        print("OK - Fallback response validation passed")
        
        print("\nSUCCESS - Phase 5 complete response generation test passed!")
        print(f"Main response: {response_content.message}")
        return True
        
    except Exception as e:
        print(f"ERROR - Response generation test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = asyncio.run(test_response_generation())
    sys.exit(0 if result else 1)