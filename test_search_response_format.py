#!/usr/bin/env python3
"""
Test script to check the actual response format from search_service
"""
import sys
import os
import json
import asyncio
import aiohttp

async def test_search_service_response_format():
    """Test the actual response format from search_service"""
    print("Testing search_service response format...")
    
    search_query = {
        "user_id": 34,
        "filters": {
            "date": {
                "gte": "2025-05-01",
                "lte": "2025-05-31"
            },
            "transaction_type": "credit"
        },
        "sort": [{"date": {"order": "desc"}}],
        "page_size": 3  # Small sample
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "http://localhost:8000/api/v1/search/search",
                json=search_query
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    print("OK - Search service responded successfully")
                    print(f"Response keys: {list(data.keys())}")
                    print(f"Total results: {data.get('total_results', 'N/A')}")
                    
                    if 'results' in data and data['results']:
                        print(f"First result keys: {list(data['results'][0].keys())}")
                        print("First result structure:")
                        first_result = data['results'][0]
                        for key, value in first_result.items():
                            print(f"  {key}: {type(value).__name__} = {value}")
                    
                    # Check if it matches SearchResponse expectation
                    print("\nExpected SearchResponse format:")
                    print("- hits: List[SearchHit] with _id, _score, _source fields")
                    print("- total_hits: int")
                    print("- aggregations: Optional[Dict]")
                    
                    print(f"\nActual format:")
                    print(f"- results: {type(data.get('results', []))}")
                    if data.get('results'):
                        print(f"- results[0] has _id: {'_id' in data['results'][0]}")
                        print(f"- results[0] has _score: {'_score' in data['results'][0]}")
                        print(f"- results[0] has _source: {'_source' in data['results'][0]}")
                    
                    return True
                else:
                    print(f"ERROR - Search service error: {response.status}")
                    error_text = await response.text()
                    print(f"Error: {error_text}")
                    return False
                    
    except Exception as e:
        print(f"ERROR - Error calling search service: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Test search service response format"""
    print("Testing search service response format mismatch...\n")
    
    success = await test_search_service_response_format()
    
    if success:
        print("\nOK - Test completed - check the format mismatch above")
    else:
        print("\nERROR - Test failed")
        
    return success

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nTest interrupted")
        sys.exit(1)