#!/usr/bin/env python3
"""
Test script to validate the SearchServiceClient URL fix
"""
import sys
import os

# Add the conversation_service to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'conversation_service'))
sys.path.insert(0, os.path.dirname(__file__))

def test_search_service_url_construction():
    """Test that SearchServiceConfig constructs correct URLs"""
    print("Testing SearchServiceConfig URL construction...")
    
    try:
        from conversation_service.core.search_service_client import SearchServiceConfig
        
        # Test default configuration
        config = SearchServiceConfig()
        print(f"  Base URL: {config.base_url}")
        print(f"  Search endpoint: {config.search_endpoint}")
        print(f"  Health endpoint: {config.health_endpoint}")
        
        # Expected URLs (the search service expects double /search)
        expected_base_url = "http://localhost:8000/api/v1/search"
        expected_search_url = "http://localhost:8000/api/v1/search/search"
        expected_health_url = "http://localhost:8000/api/v1/search/health"
        
        # Validate URLs
        if config.base_url == expected_base_url:
            print("  OK - Base URL correct")
        else:
            print(f"  ERROR - Expected base URL {expected_base_url}, got {config.base_url}")
            return False
            
        if config.search_endpoint == expected_search_url:
            print("  OK - Search endpoint correct")
        else:
            print(f"  ERROR - Expected search endpoint {expected_search_url}, got {config.search_endpoint}")
            return False
            
        if config.health_endpoint == expected_health_url:
            print("  OK - Health endpoint correct")
        else:
            print(f"  ERROR - Expected health endpoint {expected_health_url}, got {config.health_endpoint}")
            return False
            
        print("  SUCCESS - All URLs constructed correctly")
        return True
        
    except Exception as e:
        print(f"  ERROR - {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Test SearchServiceClient URL fix"""
    print("Testing SearchServiceClient URL fix...\n")
    
    if test_search_service_url_construction():
        print("\nAll tests passed! SearchServiceClient should now connect to the correct endpoint.")
        print("The URL will be http://localhost:8000/api/v1/search/search (with double /search as expected by the service)")
        return True
    else:
        print("\nTest failed!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)