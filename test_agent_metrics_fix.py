#!/usr/bin/env python3
"""
Test the AgentMetrics fix
"""
import sys
import os

# Add the conversation_service to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'conversation_service'))
sys.path.insert(0, os.path.dirname(__file__))

def test_agent_metrics_fix():
    """Test that AgentMetrics now works with all required fields"""
    
    print("Testing AgentMetrics with all required fields...")
    
    try:
        from conversation_service.models.responses.conversation_responses import AgentMetrics
        
        # Test creating AgentMetrics with all required fields
        metrics = AgentMetrics(
            agent_used="phase5_workflow",
            model_used="deepseek-chat",
            tokens_consumed=5100,  # Intent + Query + Response tokens
            processing_time_ms=35000,
            confidence_threshold_met=True,
            cache_hit=False,
            retry_count=0,
            error_count=0,
            detailed_metrics={
                "workflow_success": True,
                "agents_sequence": ["intent_classifier", "entity_extractor", "query_builder", "search_executor", "response_generator"],
                "cache_efficiency": 0.0
            }
        )
        
        print("OK - AgentMetrics created successfully")
        print(f"  Agent: {metrics.agent_used}")
        print(f"  Model: {metrics.model_used}")
        print(f"  Tokens: {metrics.tokens_consumed}")
        print(f"  Processing time: {metrics.processing_time_ms}ms")
        print(f"  Confidence met: {metrics.confidence_threshold_met}")
        print(f"  Cache hit: {metrics.cache_hit}")
        print(f"  Retry count: {metrics.retry_count}")
        print(f"  Error count: {metrics.error_count}")
        print(f"  Workflow success: {metrics.detailed_metrics['workflow_success']}")
        
        # Verify all fields are accessible
        assert metrics.agent_used == "phase5_workflow"
        assert metrics.model_used == "deepseek-chat"
        assert metrics.tokens_consumed == 5100
        assert metrics.processing_time_ms == 35000
        assert metrics.confidence_threshold_met == True
        assert metrics.cache_hit == False
        assert metrics.detailed_metrics["workflow_success"] == True
        
        print("OK - All field validation checks passed")
        
        # Test that missing fields still fail
        try:
            broken_metrics = AgentMetrics(
                agent_used="test",
                # Missing required fields
            )
            print("ERROR - Broken metrics should have failed!")
            return False
        except Exception as e:
            print(f"OK - Broken metrics correctly rejected: {type(e).__name__}")
        
        print("\nSUCCESS - AgentMetrics fix validation passed!")
        print("Phase 5 should now complete without AgentMetrics validation errors")
        return True
        
    except Exception as e:
        print(f"ERROR - AgentMetrics test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = test_agent_metrics_fix()
    sys.exit(0 if result else 1)