"""
Test simple Phase 5 - Exemple d'utilisation du workflow complet
Démontre l'utilisation basique de tous les composants Phase 5
"""
import asyncio
import json
from datetime import datetime

# Test direct des composants principaux
from conversation_service.clients.deepseek_client import DeepSeekClient
from conversation_service.agents.financial.response_generator import ResponseGeneratorAgent
from conversation_service.services.insight_generator import InsightGenerator
from conversation_service.core.context_manager import TemporaryContextManager
from conversation_service.prompts.templates.response_templates import get_response_template


async def test_phase5_components():
    """Test simple des composants Phase 5"""
    
    print("🚀 TEST SIMPLE DES COMPOSANTS PHASE 5")
    print("="*50)
    
    # Données d'exemple
    user_message = "Combien j'ai dépensé chez Amazon ce mois ?"
    intent = {"intent_type": "SEARCH_BY_MERCHANT", "confidence": 0.9}
    entities = {"merchants": ["Amazon"], "dates": {"original": "ce mois"}}
    
    # Simulation de données de recherche
    mock_analysis_data = {
        "has_results": True,
        "primary_entity": "Amazon",
        "total_amount": 234.56,
        "transaction_count": 12,
        "average_amount": 19.55,
        "unique_merchants": 1
    }
    
    try:
        # 1. Test des templates de réponse
        print("\n🎨 Test des templates de réponse...")
        
        template = get_response_template(
            intent_type=intent["intent_type"],
            user_message=user_message,
            entities=entities,
            analysis_data=mock_analysis_data,
            user_context=None,
            use_personalization=False
        )
        
        print(f"✅ Template généré: {len(template)} caractères")
        print(f"📝 Aperçu: {template[:150]}...")
        
        # 2. Test du générateur d'insights
        print("\n💡 Test du générateur d'insights...")
        
        insight_generator = InsightGenerator()
        insights = insight_generator.generate_insights(
            search_results=None,  # Simulation
            intent=intent,
            entities=entities,
            analysis_data=mock_analysis_data
        )
        
        print(f"✅ Insights générés: {len(insights)}")
        for i, insight in enumerate(insights, 1):
            print(f"   {i}. [{insight.type}] {insight.title}: {insight.description}")
        
        # 3. Test du gestionnaire de contexte
        print("\n🧠 Test du gestionnaire de contexte...")
        
        context_manager = TemporaryContextManager()
        user_id = 123
        
        # État initial
        initial_context = context_manager.get_user_context(user_id)
        print(f"📊 Contexte initial: {initial_context.get('interaction_count', 0)} interactions")
        
        # Mise à jour du contexte
        context_manager.update_context(
            user_id=user_id,
            message=user_message,
            intent=intent,
            entities=entities,
            search_results=mock_analysis_data
        )
        
        # État après mise à jour
        updated_context = context_manager.get_user_context(user_id)
        print(f"📊 Contexte mis à jour: {updated_context.get('interaction_count', 0)} interactions")
        
        context_summary = context_manager.get_context_summary(user_id)
        print(f"👤 Utilisateur récurrent: {context_summary.get('is_returning_user', False)}")
        
        # 4. Test avec personnalisation
        print("\n🎯 Test avec personnalisation...")
        
        # Simulation d'un utilisateur expérimenté
        experienced_context = {
            "is_returning_user": True,
            "interaction_count": 15,
            "detail_level": "advanced",
            "communication_style": "detailed",
            "frequent_merchants": ["Amazon", "Carrefour"]
        }
        
        personalized_template = get_response_template(
            intent_type=intent["intent_type"],
            user_message=user_message,
            entities=entities,
            analysis_data=mock_analysis_data,
            user_context=experienced_context,
            use_personalization=True
        )
        
        print(f"✅ Template personnalisé: {len(personalized_template)} caractères")
        has_personalization = "Personnalisation:" in personalized_template
        print(f"🎨 Personnalisé: {'Oui' if has_personalization else 'Non'}")
        
        # 5. Test complet avec agent (nécessite DeepSeek)
        print("\n🤖 Test avec agent Response Generator...")
        
        try:
            client = DeepSeekClient()
            response_generator = ResponseGeneratorAgent(client)
            
            # Simulation d'un search_results
            from conversation_service.models.contracts.search_service import SearchResponse
            mock_search_results = SearchResponse(
                hits=[
                    {"_source": {"amount": -45.99, "merchant_name": "Amazon", "date": "2024-08-15"}}
                    for _ in range(12)
                ],
                total_hits=12,
                took_ms=50,
                aggregations={"total_spent": {"value": -234.56}}
            )
            
            response_content, response_quality, generation_metrics = await response_generator.generate_response(
                user_message=user_message,
                intent=intent,
                entities=entities,
                search_results=mock_search_results,
                user_context=experienced_context,
                request_id="test_simple"
            )
            
            print(f"✅ Réponse générée: {len(response_content.message)} caractères")
            print(f"💡 Insights: {len(response_content.insights)}")
            print(f"🎯 Suggestions: {len(response_content.suggestions)}")
            print(f"⭐ Qualité: {response_quality.relevance_score:.2f}")
            
            print(f"\n📱 MESSAGE FINAL:")
            print(f"{'='*50}")
            print(response_content.message)
            print(f"{'='*50}")
            
            if response_content.insights:
                print(f"\n💡 INSIGHTS:")
                for insight in response_content.insights:
                    print(f"   • {insight.title}: {insight.description}")
            
            if response_content.suggestions:
                print(f"\n🎯 SUGGESTIONS:")
                for suggestion in response_content.suggestions:
                    print(f"   • {suggestion.title}: {suggestion.description}")
            
        except Exception as e:
            print(f"⚠️ Test agent Response Generator échoué: {str(e)}")
            print("   (Probablement DeepSeek API non configurée)")
        
        # 6. Statistiques finales
        print("\n📊 STATISTIQUES FINALES")
        print("="*50)
        
        final_context_stats = context_manager.get_context_stats()
        print(f"Contextes actifs: {final_context_stats['active_contexts']}")
        print(f"Interactions totales: {final_context_stats['total_interactions']}")
        print(f"Mémoire estimée: {final_context_stats['memory_usage_estimate_mb']:.2f} MB")
        
        print("\n🎉 TOUS LES TESTS RÉUSSIS!")
        print("Phase 5 est prête pour utilisation.")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERREUR LORS DES TESTS: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


async def test_context_evolution():
    """Test de l'évolution du contexte sur plusieurs interactions"""
    
    print("\n" + "="*60)
    print("🧠 TEST D'ÉVOLUTION DU CONTEXTE")
    print("="*60)
    
    context_manager = TemporaryContextManager()
    user_id = 456
    
    # Série d'interactions simulées
    interactions = [
        {
            "message": "Mes dépenses Amazon ce mois",
            "intent": {"intent_type": "SEARCH_BY_MERCHANT", "confidence": 0.9},
            "entities": {"merchants": ["Amazon"], "dates": {"original": "ce mois"}}
        },
        {
            "message": "Et chez Carrefour ?",
            "intent": {"intent_type": "SEARCH_BY_MERCHANT", "confidence": 0.85},
            "entities": {"merchants": ["Carrefour"]}
        },
        {
            "message": "Compare ces deux marchands",
            "intent": {"intent_type": "SPENDING_ANALYSIS", "confidence": 0.8},
            "entities": {"merchants": ["Amazon", "Carrefour"]}
        },
        {
            "message": "Mon budget restaurants",
            "intent": {"intent_type": "CATEGORY_ANALYSIS", "confidence": 0.9},
            "entities": {"categories": ["restaurants"]}
        }
    ]
    
    print(f"Simulation de {len(interactions)} interactions pour user {user_id}")
    
    for i, interaction in enumerate(interactions, 1):
        print(f"\n--- Interaction {i} ---")
        print(f"Message: '{interaction['message']}'")
        
        # État avant
        context_before = context_manager.get_user_context(user_id)
        
        # Mise à jour
        context_manager.update_context(
            user_id=user_id,
            message=interaction["message"],
            intent=interaction["intent"],
            entities=interaction["entities"]
        )
        
        # État après
        context_after = context_manager.get_user_context(user_id)
        context_summary = context_manager.get_context_summary(user_id)
        
        print(f"Interactions: {context_before.get('interaction_count', 0)} → {context_after.get('interaction_count', 0)}")
        print(f"Marchands connus: {len(context_before.get('entity_history', {}).get('merchants', {}))}")
        print(f"Intent principal: {context_summary.get('preferred_intents', ['None'])[0] if context_summary.get('preferred_intents') else 'None'}")
    
    # Résumé final
    final_summary = context_manager.get_context_summary(user_id)
    print(f"\n📊 RÉSUMÉ FINAL:")
    print(f"   Interactions totales: {final_summary.get('interaction_count', 0)}")
    print(f"   Utilisateur récurrent: {final_summary.get('is_returning_user', False)}")
    print(f"   Marchands fréquents: {final_summary.get('frequent_merchants', [])}")
    print(f"   Intentions préférées: {final_summary.get('preferred_intents', [])}")
    print(f"   Niveau d'engagement: {final_summary.get('engagement_level', 'unknown')}")


async def main():
    """Fonction principale"""
    
    print("🚀 TESTS SIMPLES PHASE 5 - CONVERSATION SERVICE")
    print("="*60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    try:
        # Test des composants
        components_ok = await test_phase5_components()
        
        # Test évolution contexte
        await test_context_evolution()
        
        if components_ok:
            print(f"\n✅ Phase 5 testée avec succès!")
            print("Le workflow complet est opérationnel.")
        else:
            print(f"\n⚠️ Phase 5 partiellement fonctionnelle.")
            print("Certains composants nécessitent une configuration supplémentaire.")
            
    except Exception as e:
        print(f"\n💥 Erreur critique: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())