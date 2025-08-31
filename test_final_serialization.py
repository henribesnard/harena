#!/usr/bin/env python3
"""
Test final complet de toutes nos corrections de sérialisation
"""
import sys
import json
from pathlib import Path

# Add the project to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from conversation_service.models.contracts.search_service import (
    SearchQuery, SearchFilters, DateRange, Aggregation, AggregationConfig, NestedAggregation
)
from conversation_service.models.responses.conversation_responses_phase3 import (
    ConversationResponsePhase3
)
from conversation_service.models.responses.conversation_responses import (
    ConversationResponse, IntentClassificationResult
)
from conversation_service.models.responses.enriched_conversation_responses import (
    EntityExtractionResult
)

def test_complete_serialization_chain():
    """Test de la chaîne complète de sérialisation"""
    
    print("Testing complete serialization chain with all corrections...")
    print("=" * 70)
    
    # 1. Créer une SearchQuery problématique (avec nulls)
    search_query = SearchQuery(
        user_id=34,
        query=None,  # Should become ""
        filters=SearchFilters(
            date=DateRange(
                gte="2025-05-01",
                lte="2025-05-31",
                gt=None,  # Should be excluded
                lt=None   # Should be excluded
            ),
            amount=None,  # Should be excluded
            merchant_name=None,  # Should be excluded
            transaction_type="credit",  # Should be kept
            account_id=None   # Should be excluded
        ),
        aggregations={
            "operation_stats": Aggregation(
                terms=AggregationConfig(
                    field="operation_type.keyword",
                    size=10
                ),
                sum=None,  # Should be excluded
                aggs={
                    "operation_total": NestedAggregation(
                        sum={"field": "amount_abs"},
                        avg=None  # Should be excluded
                    )
                }
            )
        },
        page_size=50
    )
    
    print("1. Testing SearchQuery direct serialization:")
    query_dict = search_query.dict(exclude_none=True)
    query_json = json.dumps(query_dict)
    print(f"   Query field: {query_dict.get('query')!r}")
    print(f"   Contains null: {'null' in query_json}")
    print(f"   Size: {len(query_json)} characters")
    
    # 2. Créer une réponse Phase 3 avec cette SearchQuery
    base_response = ConversationResponse(
        user_id=34,
        session_id="test_123",
        message="Test message",
        intent_result=IntentClassificationResult(
            intent_type="SEARCH_BY_OPERATION_TYPE",
            confidence=0.9,
            reasoning="Test intent"
        ),
        entity_result=EntityExtractionResult(
            entities={"operation_types": ["credit"]},
            extraction_confidence=0.85,
            reasoning="Test entities"
        ),
        processing_status="completed",
        response_text="Test response"
    )
    
    phase3_response = ConversationResponsePhase3(
        **base_response.dict(),
        search_query=search_query
    )
    
    print("\n2. Testing ConversationResponsePhase3 serialization:")
    response_dict = phase3_response.dict()
    response_search_query = response_dict.get("search_query", {})
    
    if response_search_query:
        response_json = json.dumps(response_search_query)
        print(f"   Search query in response - query field: {response_search_query.get('query')!r}")
        print(f"   Search query in response - contains null: {'null' in response_json}")
        print(f"   Search query in response - size: {len(response_json)} characters")
        print(f"   Search query in response - filters: {list(response_search_query.get('filters', {}).keys())}")
    else:
        print("   No search_query in response")
    
    print("\n3. Testing JSON serialization (what API returns):")
    api_json = phase3_response.json()
    api_dict = json.loads(api_json)
    api_search_query = api_dict.get("search_query", {})
    
    if api_search_query:
        print(f"   API JSON - query field: {api_search_query.get('query')!r}")
        print(f"   API JSON - contains null: {'null' in json.dumps(api_search_query)}")
        print(f"   API JSON - size: {len(json.dumps(api_search_query))} characters")
    else:
        print("   No search_query in API JSON")
    
    print("\n4. Verification checks:")
    checks = [
        ("SearchQuery.dict() excludes nulls", 'null' not in json.dumps(query_dict)),
        ("SearchQuery.query is empty string", query_dict.get('query') == ''),
        ("Response search_query excludes nulls", 'null' not in json.dumps(response_search_query)),
        ("Response search_query.query is empty string", response_search_query.get('query') == ''),
        ("API JSON search_query excludes nulls", 'null' not in json.dumps(api_search_query)),
        ("API JSON search_query.query is empty string", api_search_query.get('query') == ''),
        ("Filters are cleaned", len(response_search_query.get('filters', {})) <= 2),  # Only date and transaction_type
        ("Date filter cleaned", 'gt' not in response_search_query.get('filters', {}).get('date', {}))
    ]
    
    all_passed = True
    for check_name, passed in checks:
        status = "PASS" if passed else "FAIL"
        print(f"   {check_name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n5. Final comparison:")
    print(f"   Original SearchQuery serialized: {len(search_query.dict())} chars with nulls")
    print(f"   Clean SearchQuery serialized: {len(query_dict)} chars without nulls")
    print(f"   Response API JSON search_query: {len(json.dumps(api_search_query))} chars")
    print(f"   Size reduction: {(1 - len(json.dumps(api_search_query)) / len(json.dumps(search_query.dict())))*100:.1f}%")
    
    if all_passed:
        print("\n" + "="*70)
        print("SUCCESS: ALL CORRECTIONS WORKING PERFECTLY!")
        print("- query: null -> query: \"\"")
        print("- All null optional fields excluded")
        print("- Response serialization cleaned")
        print("- API JSON output optimized") 
        print("- Ready for production use")
        
        print(f"\nFinal clean search_query that will be sent to API:")
        print(json.dumps(api_search_query, indent=2))
        return True
    else:
        print("\nFAIL: Some corrections not working")
        return False

if __name__ == "__main__":
    success = test_complete_serialization_chain()
    if success:
        print("\nALL CORRECTIONS SUCCESSFULLY APPLIED!")
        print("The generated queries will now pass search_service validation.")
    else:
        print("\nSome corrections need additional work.")