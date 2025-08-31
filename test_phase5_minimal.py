#!/usr/bin/env python3
"""
Test minimal Phase 5 functionality without running full service
"""
import sys
import os
import asyncio
import json

# Add the conversation_service to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'conversation_service'))
sys.path.insert(0, os.path.dirname(__file__))

from conversation_service.models.requests.conversation_requests import ConversationRequest
from conversation_service.api.routes.conversation_phase5 import get_dependencies

async def test_phase5_minimal():
    """Test Phase 5 dependencies and basic workflow"""
    
    print("Testing Phase 5 minimal functionality...")
    
    try:
        # Test dependencies initialization
        print("Step 1: Testing dependencies initialization...")
        deps = get_dependencies()
        print("OK - Dependencies initialized successfully")
        
        # Check agents
        agents = deps["agents"]
        print(f"OK - Agents available: {list(agents.keys())}")
        
        # Test each agent availability
        for name, agent in agents.items():
            if agent:
                print(f"OK - {name}: Available")
            else:
                print(f"ERROR - {name}: Not available")
        
        # Test personalization context
        print("Step 2: Testing personalization engine...")
        personalization = deps["personalization"]
        if personalization:
            context = personalization.get_personalization_context(34)
            print(f"OK - Personalization context: {type(context)}")
        else:
            print("ERROR - Personalization engine not available")
        
        print("OK - Phase 5 minimal test completed successfully")
        return True
        
    except Exception as e:
        print(f"ERROR - Phase 5 test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = asyncio.run(test_phase5_minimal())
    sys.exit(0 if result else 1)