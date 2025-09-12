#!/usr/bin/env python3
"""
Test script to validate response generator filtering changes
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'conversation_service'))

from agents.llm.response_generator import ResponseGenerator, ResponseGenerationRequest
from agents.llm.llm_provider import LLMProviderManager

def test_transaction_filtering():
    """Test que les champs techniques sont bien filtres"""
    
    # Mock transaction with technical and user-friendly fields
    mock_transaction = {
        'user_id': 34,  # Should be filtered out
        'transaction_id': 'tx_12345',  # Should be filtered out 
        'internal_id': 'abc123',  # Should be filtered out
        'amount': 550.0,  # Should be kept
        'date': '2025-01-15',  # Should be kept  
        'merchant_name': 'CARREFOUR MARKET',  # Should be kept
        'primary_description': 'Courses alimentaires',  # Should be kept
        'category': 'Alimentation',  # Should be kept
        'transaction_type': 'debit'  # Should be kept
    }
    
    # Create response generator
    response_gen = ResponseGenerator(llm_manager=None)
    
    # Test filtering
    filtered = response_gen._filter_transaction_data(mock_transaction)
    
    print("Original transaction keys:", sorted(mock_transaction.keys()))
    print("Filtered transaction keys:", sorted(filtered.keys()))
    
    # Validate filtering
    expected_fields = {'amount', 'date', 'merchant_name', 'primary_description', 'category', 'transaction_type'}
    technical_fields = {'user_id', 'transaction_id', 'internal_id'}
    
    # Check that user-friendly fields are kept
    for field in expected_fields:
        if field in mock_transaction and field not in filtered:
            print(f"ERROR: Field '{field}' should be kept but was filtered out")
            return False
    
    # Check that technical fields are removed
    for field in technical_fields:
        if field in filtered:
            print(f"ERROR: Technical field '{field}' should be filtered out but was kept")
            return False
    
    print("SUCCESS: Transaction filtering working correctly")
    return True

def test_system_prompt_changes():
    """Test que les prompts systeme ont ete modifies"""
    
    # Mock request
    request = ResponseGenerationRequest(
        intent_group="transaction_search",
        intent_subtype="filter", 
        user_message="mes dépenses de plus de 500 euros",
        search_results=[],
        conversation_context=[],
        user_profile={'preferences': 'notifications activées'},
        user_id=34
    )
    
    response_gen = ResponseGenerator(llm_manager=None)
    
    # Get template and system prompt
    template = response_gen._select_response_template(request)
    system_prompt = response_gen._build_system_prompt(request, template)
    
    print("\n=== SYSTEM PROMPT ===")
    print(system_prompt)
    print("=====================")
    
    # Validate that personal rules are present
    if "vos transactions" not in system_prompt:
        print("ERROR: 'vos transactions' rule missing from system prompt")
        return False
        
    if "JAMAIS mentionner l'ID utilisateur" not in system_prompt:
        print("ERROR: User ID filtering rule missing from system prompt") 
        return False
        
    if "nom du marchand" not in system_prompt:
        print("ERROR: Merchant name emphasis missing from system prompt")
        return False
    
    # Validate that user ID is NOT in the prompt 
    if "ID: 34" in system_prompt:
        print("ERROR: User ID should not appear in system prompt")
        return False
        
    print("SUCCESS: System prompt changes working correctly")
    return True

def test_template_personalization():
    """Test que les templates sont plus personnels"""
    
    request = ResponseGenerationRequest(
        intent_group="transaction_search",
        intent_subtype="filter",
        user_message="test",
        search_results=[], 
        conversation_context=[],
        user_profile={},
        user_id=34
    )
    
    response_gen = ResponseGenerator(llm_manager=None)
    template = response_gen._select_response_template(request)
    
    print("\n=== TEMPLATE ===")
    print(f"System: {template['system']}")
    print(f"Structure: {template['structure']}")
    print("================")
    
    # Check for personal tone
    if "assistant personnel" not in template['system'].lower():
        print("ERROR: Template should be more personal")
        return False
        
    if "vos" not in template['structure'].lower():
        print("ERROR: Template structure should use personal pronouns")
        return False
        
    print("SUCCESS: Template personalization working correctly")
    return True

if __name__ == "__main__":
    print("Testing Response Generator Improvements")
    print("=" * 50)
    
    success = True
    success &= test_transaction_filtering()
    success &= test_system_prompt_changes()
    success &= test_template_personalization()
    
    print("\n" + "=" * 50)
    if success:
        print("SUCCESS: All tests passed! Response generator improvements working correctly.")
    else:
        print("ERROR: Some tests failed.")
    
    sys.exit(0 if success else 1)