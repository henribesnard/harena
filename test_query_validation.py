#!/usr/bin/env python3
"""
Test script to verify query validation fix
"""
import asyncio
import json
import aiohttp
from typing import Dict, Any

# Test configuration
CONVERSATION_SERVICE_URL = "http://localhost:8001"
TEST_USER_ID = 34

async def test_query_validation():
    """Test that query validation works correctly after the fix"""
    
    async with aiohttp.ClientSession() as session:
        # First test health endpoint
        print("Testing health endpoint...")
        try:
            async with session.get(f"{CONVERSATION_SERVICE_URL}/api/v1/conversation/health") as health_response:
                print(f"Health Status: {health_response.status}")
                if health_response.status == 200:
                    health_data = await health_response.json()
                    print(f"Health Data: {health_data}")
                else:
                    print(f"Health Error: {await health_response.text()}")
        except Exception as e:
            print(f"Health check failed: {e}")
        
        # Test payload that should now work
        test_payload = {
            "user_id": TEST_USER_ID,
            "message": "Montre-moi mes transactions chez Carrefour",
            "session_id": "test_session_123"
        }
        
        print("\nTesting query validation fix...")
        print(f"URL: {CONVERSATION_SERVICE_URL}/api/v1/conversation/{TEST_USER_ID}")
        print(f"Payload: {json.dumps(test_payload, indent=2)}")
        print()
        
        try:
            async with session.post(
                f"{CONVERSATION_SERVICE_URL}/api/v1/conversation/{TEST_USER_ID}",
                json=test_payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                
                print(f"Response Status: {response.status}")
                
                if response.status == 200:
                    result = await response.json()
                    
                    # Check if we have search_execution_result
                    search_result = result.get("search_execution_result")
                    if search_result:
                        query_validation = search_result.get("query_validation")
                        if query_validation:
                            schema_valid = query_validation.get("schema_valid")
                            print(f"Schema Valid: {schema_valid}")
                            
                            if schema_valid:
                                print("[SUCCESS] Query validation is now working!")
                            else:
                                errors = query_validation.get("errors", [])
                                print(f"[FAIL] Schema validation errors: {errors}")
                        
                        # Check if search_results exist and are not null
                        search_results = search_result.get("search_results")
                        if search_results is not None:
                            print(f"[SUCCESS] Search results returned: {len(search_results.get('hits', []))} hits")
                        else:
                            print("[INFO] Search results are null - may be due to empty results or other issues")
                    
                    print("\nFull response:")
                    print(json.dumps(result, indent=2))
                
                else:
                    error_text = await response.text()
                    print(f"[ERROR] HTTP {response.status}: {error_text}")
                    
        except Exception as e:
            print(f"[ERROR] Request failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_query_validation())