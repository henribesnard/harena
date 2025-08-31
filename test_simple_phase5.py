"""
Test simple Phase 5 - génération de réponse même sans résultats
"""
import asyncio
from datetime import datetime
from conversation_service.agents.financial.response_generator import ResponseGeneratorAgent
from conversation_service.clients.deepseek_client import DeepSeekClient
from conversation_service.models.contracts.search_service import SearchResponse


async def test_empty_results():
    """Test avec résultats vides"""
    
    print("=== TEST PHASE 5 AVEC RESULTATS VIDES ===")
    
    # Données du test
    user_message = "Mes rentrées d'argent en mai ?"
    intent = {"intent_type": "SEARCH_BY_OPERATION_TYPE", "confidence": 0.92}
    entities = {
        "entities": {
            "transaction_types": ["credit"],
            "dates": [{"type": "period", "value": "2025-05", "text": "mai"}]
        }
    }
    
    # Résultats vides (comme dans votre exemple)
    empty_results = SearchResponse(
        hits=[],
        total_hits=0,
        aggregations={},
        took_ms=0,
        query_id="fallback_test",
        timestamp=datetime.now().isoformat()
    )
    
    try:
        client = DeepSeekClient()
        generator = ResponseGeneratorAgent(client)
        
        user_context = {
            "is_returning_user": True,
            "interaction_count": 5,
            "communication_style": "professional_friendly"
        }
        
        print("Génération de réponse en cours...")
        
        response_content, response_quality, metrics = await generator.generate_response(
            user_message=user_message,
            intent=intent,
            entities=entities,
            search_results=empty_results,
            user_context=user_context,
            request_id="test_empty"
        )
        
        print(f"SUCCESS - Réponse générée!")
        print(f"Longueur message: {len(response_content.message)} chars")
        print(f"Insights: {len(response_content.insights)}")
        print(f"Suggestions: {len(response_content.suggestions)}")
        print(f"Qualité: {response_quality.relevance_score:.2f}")
        print(f"Temps génération: {metrics.generation_time_ms}ms")
        
        print("\n=== MESSAGE GENERE ===")
        print(response_content.message)
        print("======================")
        
        if response_content.insights:
            print("\n=== INSIGHTS ===")
            for i, insight in enumerate(response_content.insights, 1):
                print(f"{i}. [{insight.type}] {insight.title}")
                print(f"   {insight.description}")
        
        if response_content.suggestions:
            print("\n=== SUGGESTIONS ===")
            for i, suggestion in enumerate(response_content.suggestions, 1):
                print(f"{i}. [{suggestion.type}] {suggestion.title}")
                print(f"   {suggestion.description}")
        
        return True
        
    except Exception as e:
        print(f"ERREUR: {str(e)}")
        return False


async def main():
    print("DEBUT TEST PHASE 5")
    result = await test_empty_results()
    
    if result:
        print("\nSUCCESS: Phase 5 génère des réponses utiles même sans résultats!")
        print("L'endpoint /conversation/{user_id} devrait maintenant être complet.")
    else:
        print("\nERREUR: Phase 5 nécessite des corrections.")


if __name__ == "__main__":
    asyncio.run(main())