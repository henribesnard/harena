#!/usr/bin/env python3
"""
Test pour vérifier que nos corrections sont bien appliquées
"""
import asyncio
import aiohttp
import json

CONVERSATION_SERVICE_URL = "http://localhost:8001"
TEST_USER_ID = 34

async def test_corrections():
    """Test que nos corrections sont appliquées"""
    
    test_payload = {
        "user_id": TEST_USER_ID,
        "message": "Montre-moi mes transactions de crédit en mai",
        "session_id": "test_corrections_123"
    }
    
    async with aiohttp.ClientSession() as session:
        try:
            print(f"Testing corrections on {CONVERSATION_SERVICE_URL}/api/v1/conversation/{TEST_USER_ID}")
            
            async with session.post(
                f"{CONVERSATION_SERVICE_URL}/api/v1/conversation/{TEST_USER_ID}",
                json=test_payload,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                
                print(f"Response Status: {response.status}")
                
                if response.status == 200:
                    result = await response.json()
                    
                    # Extract search execution info
                    search_execution = result.get("search_execution_result", {})
                    generated_query = search_execution.get("generated_query", {})
                    query_validation = search_execution.get("query_validation", {})
                    
                    print("\n=== QUERY ANALYSIS ===")
                    print(f"Query field value: {generated_query.get('query')!r}")
                    print(f"Query field type: {type(generated_query.get('query'))}")
                    
                    print("\n=== VALIDATION ANALYSIS ===")
                    print(f"Schema valid: {query_validation.get('schema_valid')}")
                    print(f"Contract compliant: {query_validation.get('contract_compliant')}")
                    print(f"Errors: {query_validation.get('errors', [])}")
                    
                    print("\n=== NULL FIELDS COUNT ===")
                    json_str = json.dumps(generated_query)
                    null_count = json_str.count('"null"') + json_str.count(': null')
                    print(f"Total null fields in JSON: {null_count}")
                    
                    print("\n=== SUCCESS CHECKS ===")
                    checks = [
                        ("Query is not None", generated_query.get('query') is not None),
                        ("Query is string", isinstance(generated_query.get('query'), str)),
                        ("Schema validation passes", query_validation.get('schema_valid') == True),
                        ("Contract validation passes", query_validation.get('contract_compliant') == True),
                        ("No validation errors", len(query_validation.get('errors', [])) == 0),
                        ("Minimal null fields", null_count < 20)  # Should be much lower with exclude_none
                    ]
                    
                    all_passed = True
                    for check_name, passed in checks:
                        status = "PASS" if passed else "FAIL"
                        print(f"  {check_name}: {status}")
                        if not passed:
                            all_passed = False
                    
                    if all_passed:
                        print("\nSUCCESS: All corrections are working!")
                        return True
                    else:
                        print("\nPARTIAL: Some corrections may not be applied yet")
                        
                        # Show a sample of the generated query for debugging
                        print(f"\nGenerated query sample (first 500 chars):")
                        print(json_str[:500] + "..." if len(json_str) > 500 else json_str)
                        return False
                
                elif response.status == 404:
                    print("\nERROR: Service endpoint not found (404)")
                    return False
                
                else:
                    error_text = await response.text()
                    print(f"\nERROR: HTTP {response.status} - {error_text}")
                    return False
                    
        except Exception as e:
            print(f"\nEXCEPTION: {type(e).__name__}: {e}")
            return False

if __name__ == "__main__":
    success = asyncio.run(test_corrections())
    if success:
        print("\nFINAL: Corrections successfully applied!")
    else:
        print("\nFINAL: Need to investigate why corrections aren't working")