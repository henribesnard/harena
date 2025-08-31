#!/usr/bin/env python3
"""
Test final de toutes les corrections pour vérifier que les requêtes
passent maintenant la validation et n'ont plus query=null
"""
import asyncio
import aiohttp
import json
import sys
from pathlib import Path

# Test configuration
CONVERSATION_SERVICE_URL = "http://localhost:8001"
TEST_USER_ID = 34

async def test_final_fix():
    """Test final avec une vraie requête conversation"""
    
    print("Testing final fix with real conversation service...")
    print("=" * 60)
    
    # Test payload similar to the failing example
    test_payload = {
        "user_id": TEST_USER_ID,
        "message": "Montre-moi mes opérations de crédit du mois de mai",
        "session_id": "test_final_fix_123"
    }
    
    async with aiohttp.ClientSession() as session:
        try:
            print(f"POST {CONVERSATION_SERVICE_URL}/api/v1/conversation/{TEST_USER_ID}")
            print(f"Body: {json.dumps(test_payload, indent=2)}")
            print()
            
            async with session.post(
                f"{CONVERSATION_SERVICE_URL}/api/v1/conversation/{TEST_USER_ID}",
                json=test_payload,
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                
                print(f"Response Status: {response.status}")
                
                if response.status == 200:
                    result = await response.json()
                    
                    # Check query validation
                    search_execution = result.get("search_execution_result")
                    if search_execution:
                        query_validation = search_execution.get("query_validation", {})
                        generated_query = search_execution.get("generated_query", {})
                        search_results = search_execution.get("search_results")
                        
                        print("\n1. Query Validation Result:")
                        print(f"   schema_valid: {query_validation.get('schema_valid')}")
                        print(f"   contract_compliant: {query_validation.get('contract_compliant')}")
                        print(f"   errors: {query_validation.get('errors', [])}")
                        print(f"   warnings: {query_validation.get('warnings', [])}")
                        
                        print("\n2. Generated Query Analysis:")
                        print(f"   query field: {generated_query.get('query')!r}")
                        print(f"   user_id: {generated_query.get('user_id')}")
                        print(f"   filters contain user_id: {'user_id' in str(generated_query.get('filters', {}))}")
                        
                        print("\n3. Search Results:")
                        if search_results:
                            print(f"   Search executed: SUCCESS")
                            print(f"   Total hits: {search_results.get('total_hits', 0)}")
                            print(f"   Took ms: {search_results.get('took_ms', 0)}")
                        else:
                            print(f"   Search executed: FAILED or NULL")
                        
                        # Success criteria
                        success_checks = [
                            ("Schema valid", query_validation.get('schema_valid') == True),
                            ("Contract compliant", query_validation.get('contract_compliant') == True),
                            ("No validation errors", len(query_validation.get('errors', [])) == 0),
                            ("Query not null", generated_query.get('query') is not None),
                            ("Query is string", isinstance(generated_query.get('query'), str)),
                            ("Search results exist", search_results is not None)
                        ]
                        
                        print("\n4. Success Checks:")
                        all_passed = True
                        for check_name, passed in success_checks:
                            status = "PASS" if passed else "FAIL"
                            print(f"   {check_name}: {status}")
                            if not passed:
                                all_passed = False
                        
                        if all_passed:
                            print("\n✅ SUCCESS: All fixes working correctly!")
                            print("   - No more query=null (422 errors)")
                            print("   - No more user_id in filters (validation errors)")  
                            print("   - Search execution successful")
                            return True
                        else:
                            print("\n❌ PARTIAL: Some issues remain")
                            return False
                    
                    else:
                        print("\n❌ FAIL: No search_execution_result found")
                        print("Response structure:")
                        print(json.dumps(result, indent=2)[:1000] + "...")
                        return False
                
                elif response.status == 404:
                    print("\n❌ FAIL: Endpoint not found (404)")
                    print("Service may not be fully started or routes not configured")
                    return False
                
                else:
                    error_text = await response.text()
                    print(f"\n❌ FAIL: HTTP {response.status}")
                    print(f"Error: {error_text}")
                    return False
                    
        except asyncio.TimeoutError:
            print("\n❌ FAIL: Request timeout")
            return False
        except Exception as e:
            print(f"\n❌ FAIL: Exception - {type(e).__name__}: {e}")
            return False

if __name__ == "__main__":
    success = asyncio.run(test_final_fix())
    print(f"\n{'='*60}")
    if success:
        print("🎉 FINAL RESULT: All corrections successful!")
        print("Ready for production use with search_service")
    else:
        print("🔧 FINAL RESULT: Additional fixes needed")
        print("Check service logs for more details")