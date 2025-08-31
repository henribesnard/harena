#!/usr/bin/env python3
"""
Test the ProcessingSteps fix for response_generator
"""
import sys
import os

# Add the conversation_service to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'conversation_service'))
sys.path.insert(0, os.path.dirname(__file__))

def test_processing_steps_fix():
    """Test that ProcessingSteps now accepts response_generator"""
    
    print("Testing ProcessingSteps with response_generator...")
    
    try:
        from conversation_service.models.responses.conversation_responses_phase3 import ProcessingSteps
        
        # Test that response_generator is now accepted
        step5 = ProcessingSteps(
            agent="response_generator",
            duration_ms=12663,
            cache_hit=False,
            success=True,
            error_message=None
        )
        
        print("OK - ProcessingSteps created successfully with response_generator")
        print(f"  Agent: {step5.agent}")
        print(f"  Duration: {step5.duration_ms}ms")
        print(f"  Success: {step5.success}")
        
        # Test all Phase 5 agents
        phase5_agents = [
            "intent_classifier",
            "entity_extractor", 
            "query_builder",
            "search_executor",
            "response_generator"  # This should now work
        ]
        
        for agent_name in phase5_agents:
            step = ProcessingSteps(
                agent=agent_name,
                duration_ms=1000,
                cache_hit=False,
                success=True
            )
            print(f"OK - {agent_name} step created successfully")
        
        # Test that invalid agents still fail
        try:
            invalid_step = ProcessingSteps(
                agent="invalid_agent",
                duration_ms=1000,
                cache_hit=False,
                success=True
            )
            print("ERROR - Invalid agent should have failed!")
            return False
        except Exception as e:
            print(f"OK - Invalid agent correctly rejected: {type(e).__name__}")
        
        print("\nSUCCESS - ProcessingSteps fix validation passed!")
        print("Phase 5 should now complete Step 5 (Response Generation) successfully")
        return True
        
    except Exception as e:
        print(f"ERROR - ProcessingSteps test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = test_processing_steps_fix()
    sys.exit(0 if result else 1)