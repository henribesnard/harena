#!/usr/bin/env python3
"""
Test script to validate Phase 5 fixes
Tests the search service URL fix and metrics attribute fix
"""
import sys
import os

# Add the conversation_service to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'conversation_service'))
sys.path.insert(0, os.path.dirname(__file__))

def test_search_service_config():
    """Test that SearchServiceConfig uses localhost:8000"""
    print("Testing SearchServiceConfig URL...")
    
    try:
        from conversation_service.core.search_service_client import SearchServiceConfig
        
        # Test default configuration
        config = SearchServiceConfig()
        print(f"  Default base_url: {config.base_url}")
        
        # Should be localhost:8000/api/v1/search now
        expected_url = "http://localhost:8000/api/v1/search"
        if config.base_url == expected_url:
            print("  OK - Search service URL correctly configured for localhost:8000")
            return True
        else:
            print(f"  ERROR - Expected {expected_url}, got {config.base_url}")
            return False
            
    except Exception as e:
        print(f"  ERROR - Error testing SearchServiceConfig: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_query_generation_metrics():
    """Test that QueryGenerationMetrics can be created properly"""
    print("Testing QueryGenerationMetrics creation...")
    
    try:
        from conversation_service.models.responses.conversation_responses_phase3 import QueryGenerationMetrics
        
        # Test creating QueryGenerationMetrics with all required fields
        metrics = QueryGenerationMetrics(
            generation_time_ms=1000,
            validation_time_ms=50,
            optimization_time_ms=30,
            generation_confidence=0.85,
            validation_passed=True,
            optimizations_applied=1,
            estimated_performance="good",
            estimated_results_count=15
        )
        
        print("  OK - QueryGenerationMetrics created successfully")
        print(f"    Generation time: {metrics.generation_time_ms}ms")
        print(f"    Confidence: {metrics.generation_confidence}")
        print(f"    Performance: {metrics.estimated_performance}")
        return True
        
    except Exception as e:
        print(f"  ERROR - Error testing QueryGenerationMetrics: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_processing_steps_response_generator():
    """Test that ProcessingSteps accepts response_generator"""
    print("Testing ProcessingSteps with response_generator...")
    
    try:
        from conversation_service.models.responses.conversation_responses_phase3 import ProcessingSteps
        
        # Test creating ProcessingSteps with response_generator
        step = ProcessingSteps(
            agent="response_generator",
            duration_ms=5000,
            cache_hit=False,
            success=True
        )
        
        print("  OK - ProcessingSteps with response_generator created successfully")
        print(f"    Agent: {step.agent}")
        print(f"    Duration: {step.duration_ms}ms")
        return True
        
    except Exception as e:
        print(f"  ERROR - Error testing ProcessingSteps: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all Phase 5 fix validation tests"""
    print("Testing Phase 5 fixes...\n")
    
    results = []
    
    # Test 1: Search service URL fix
    results.append(test_search_service_config())
    print()
    
    # Test 2: QueryGenerationMetrics fix
    results.append(test_query_generation_metrics())
    print()
    
    # Test 3: ProcessingSteps response_generator fix
    results.append(test_processing_steps_response_generator())
    print()
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"ALL TESTS PASSED ({passed}/{total})")
        print("Phase 5 should now work with:")
        print("  - Correct search service URL (localhost:8000)")
        print("  - Proper QueryGenerationMetrics creation")
        print("  - Support for response_generator agent")
        return True
    else:
        print(f"FAILED: {total - passed} tests failed out of {total}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)