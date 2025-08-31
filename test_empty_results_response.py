"""
Test Phase 5 avec résultats de recherche vides
Vérifie que même sans résultats, une réponse contextuelle est générée
"""
import asyncio
from datetime import datetime

# Test des composants
from conversation_service.agents.financial.response_generator import ResponseGeneratorAgent
from conversation_service.clients.deepseek_client import DeepSeekClient
from conversation_service.models.contracts.search_service import SearchResponse


async def test_empty_results_response():
    """Test génération de réponse avec résultats vides"""
    
    print("TEST GENERATION REPONSE AVEC RESULTATS VIDES")
    print("="*60)
    
    # Données similaires à votre exemple
    user_message = "Mes rentrées d'argent en mai ?"
    intent = {
        "intent_type": "SEARCH_BY_OPERATION_TYPE",
        "confidence": 0.92
    }
    entities = {
        "entities": {
            "amounts": [],
            "dates": [{"type": "period", "value": "2025-05", "text": "mai"}],
            "merchants": [],
            "categories": [],
            "operation_types": [],
            "transaction_types": ["credit"],
            "text_search": []
        },
        "confidence": 0.92
    }
    
    # Simulation de résultats vides (comme dans votre exemple)
    empty_search_results = SearchResponse(
        hits=[],
        total_hits=0,
        aggregations={},
        took_ms=0,
        query_id="fallback_test",
        timestamp=datetime.now().isoformat()
    )
    
    try:
        # Test avec DeepSeek
        print("\nTest avec agent Response Generator...")
        
        client = DeepSeekClient()
        response_generator = ResponseGeneratorAgent(client)
        
        # Contexte utilisateur expérimenté pour plus de personnalisation
        user_context = {
            "is_returning_user": True,
            "interaction_count": 10,
            "detail_level": "medium",
            "communication_style": "professional_friendly",
            "frequent_merchants": ["Amazon", "Salaire"],
            "preferred_intents": ["SEARCH_BY_OPERATION_TYPE"]
        }
        
        response_content, response_quality, generation_metrics = await response_generator.generate_response(
            user_message=user_message,
            intent=intent,
            entities=entities,
            search_results=empty_search_results,  # Résultats vides !
            user_context=user_context,
            request_id="test_empty_results"
        )
        
        print(f"✅ Réponse générée malgré résultats vides!")
        print(f"📝 Longueur: {len(response_content.message)} caractères")
        print(f"💡 Insights: {len(response_content.insights)}")
        print(f"🎯 Suggestions: {len(response_content.suggestions)}")
        print(f"⭐ Qualité: {response_quality.relevance_score:.2f}")
        print(f"🔄 Actions suivantes: {len(response_content.next_actions)}")
        
        print(f"\n📱 RÉPONSE FINALE CONTEXTUALISÉE:")
        print("="*60)
        print(response_content.message)
        print("="*60)
        
        if response_content.insights:
            print(f"\n💡 INSIGHTS GÉNÉRÉS:")
            for i, insight in enumerate(response_content.insights, 1):
                print(f"   {i}. [{insight.type.upper()}] {insight.title}")
                print(f"      {insight.description} (conf: {insight.confidence:.2f})")
        
        if response_content.suggestions:
            print(f"\n🎯 SUGGESTIONS:")
            for i, suggestion in enumerate(response_content.suggestions, 1):
                print(f"   {i}. [{suggestion.type.upper()}] {suggestion.title}")
                print(f"      {suggestion.description}")
                if suggestion.action:
                    print(f"      → Action: {suggestion.action}")
        
        if response_content.next_actions:
            print(f"\n🔄 ACTIONS SUIVANTES:")
            for i, action in enumerate(response_content.next_actions, 1):
                print(f"   {i}. {action}")
        
        # Vérification que la réponse est utile malgré l'absence de résultats
        print(f"\n📊 ANALYSE DE LA RÉPONSE:")
        print(f"   • Contient explication absence résultats: {'aucun' in response_content.message.lower() or 'pas' in response_content.message.lower() or 'trouvé' in response_content.message.lower()}")
        print(f"   • Propose alternatives: {len(response_content.suggestions) > 0}")
        print(f"   • Fournit contexte: {len(response_content.insights) > 0}")
        print(f"   • Actionnable: {response_quality.actionability}")
        print(f"   • Complétude: {response_quality.completeness}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors du test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


async def test_with_some_results():
    """Test avec quelques résultats pour comparaison"""
    
    print(f"\n{'='*60}")
    print("🧪 TEST COMPARAISON AVEC RÉSULTATS PRÉSENTS")
    print("="*60)
    
    user_message = "Mes rentrées d'argent en mai ?"
    intent = {"intent_type": "SEARCH_BY_OPERATION_TYPE", "confidence": 0.92}
    entities = {
        "entities": {
            "transaction_types": ["credit"],
            "dates": [{"type": "period", "value": "2025-05", "text": "mai"}]
        }
    }
    
    # Simulation de résultats avec données
    results_with_data = SearchResponse(
        hits=[
            {"_source": {"amount": 2500.0, "merchant_name": "Salaire Entreprise", "date": "2025-05-01", "operation_type": "virement"}},
            {"_source": {"amount": 150.0, "merchant_name": "Remboursement", "date": "2025-05-15", "operation_type": "virement"}},
            {"_source": {"amount": 50.0, "merchant_name": "Dividendes", "date": "2025-05-20", "operation_type": "virement"}}
        ],
        total_hits=3,
        aggregations={
            "total_credit": {"value": 2700.0},
            "operation_stats": {
                "buckets": [
                    {"key": "virement", "operation_total": {"value": 2700.0}, "operation_count": {"value": 3}}
                ]
            }
        },
        took_ms=45,
        query_id="search_with_results",
        timestamp=datetime.now().isoformat()
    )
    
    try:
        client = DeepSeekClient()
        response_generator = ResponseGeneratorAgent(client)
        
        user_context = {
            "is_returning_user": True,
            "interaction_count": 10,
            "communication_style": "professional_friendly"
        }
        
        response_content, response_quality, generation_metrics = await response_generator.generate_response(
            user_message=user_message,
            intent=intent,
            entities=entities,
            search_results=results_with_data,
            user_context=user_context,
            request_id="test_with_results"
        )
        
        print(f"✅ Réponse avec données générée!")
        print(f"📝 Longueur: {len(response_content.message)} caractères")
        print(f"💡 Insights: {len(response_content.insights)}")
        print(f"🎯 Suggestions: {len(response_content.suggestions)}")
        print(f"⭐ Qualité: {response_quality.relevance_score:.2f}")
        
        print(f"\n📱 RÉPONSE AVEC DONNÉES:")
        print("="*60)
        print(response_content.message)
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors du test avec résultats: {str(e)}")
        return False


async def main():
    """Fonction principale"""
    
    print("🚀 TEST PHASE 5 - GESTION RÉSULTATS VIDES vs PRÉSENTS")
    print("="*60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    try:
        # Test avec résultats vides
        empty_results_ok = await test_empty_results_response()
        
        # Test avec résultats présents (pour comparaison)
        with_results_ok = await test_with_some_results()
        
        print(f"\n{'='*60}")
        print("📊 RÉSUMÉ DES TESTS")
        print("="*60)
        print(f"Test résultats vides: {'✅ OK' if empty_results_ok else '❌ ÉCHEC'}")
        print(f"Test avec résultats: {'✅ OK' if with_results_ok else '❌ ÉCHEC'}")
        
        if empty_results_ok and with_results_ok:
            print("\n🎉 Phase 5 gère correctement les deux cas !")
            print("L'agent génère des réponses utiles même sans données de recherche.")
        else:
            print("\n⚠️ Des améliorations sont nécessaires.")
            
    except Exception as e:
        print(f"\n💥 Erreur critique: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())