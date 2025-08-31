#!/usr/bin/env python3
"""
Test script to validate imports work after format conversion fix
"""
import sys
import os

# Add the conversation_service to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'conversation_service'))
sys.path.insert(0, os.path.dirname(__file__))

def test_imports():
    """Test that all imports work correctly after the changes"""
    print("Testing imports after format conversion fix...")
    
    try:
        # Test core imports
        from conversation_service.core.search_service_client import SearchServiceClient, SearchServiceConfig
        print("  OK - SearchServiceClient imports")
        
        # Test contract imports  
        from conversation_service.models.contracts.search_service import SearchResponse, SearchHit
        print("  OK - SearchResponse/SearchHit imports")
        
        # Test search executor imports
        from conversation_service.agents.search.search_executor import SearchExecutor
        print("  OK - SearchExecutor imports")
        
        # Test creating client
        client = SearchServiceClient()
        print("  OK - SearchServiceClient instantiation")
        
        # Test format conversion method exists
        if hasattr(client, '_convert_search_service_response'):
            print("  OK - _convert_search_service_response method exists")
        else:
            print("  ERROR - _convert_search_service_response method missing")
            return False
            
        return True
        
    except Exception as e:
        print(f"  ERROR - {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Test imports after format conversion fix"""
    print("Testing imports after format conversion fix...\n")
    
    if test_imports():
        print("\nAll imports working! The format conversion fix is properly integrated.")
        return True
    else:
        print("\nImport test failed!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)