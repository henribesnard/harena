"""
Test complet du workflow Phase 5 - Génération de réponses
Test end-to-end du pipeline complet : intentions + entités + requête + résultats + réponse naturelle
"""
import asyncio
import json
import time
from datetime import datetime
from typing import Dict, Any

# Import des composants Phase 5
from conversation_service.clients.deepseek_client import DeepSeekClient
from conversation_service.agents.financial.intent_classifier import IntentClassifierAgent
from conversation_service.agents.financial.entity_extractor import EntityExtractorAgent
from conversation_service.agents.financial.query_builder import QueryBuilderAgent
from conversation_service.agents.search.search_executor import SearchExecutorAgent
from conversation_service.agents.financial.response_generator import ResponseGeneratorAgent

# Services Phase 5
from conversation_service.services.insight_generator import InsightGenerator
from conversation_service.core.context_manager import TemporaryContextManager, PersonalizationEngine
from conversation_service.models.responses.conversation_responses import (
    ConversationResponse, ConversationResponseFactory
)

# Templates
from conversation_service.prompts.templates.response_templates import get_response_template


class Phase5WorkflowTester:
    """Testeur complet du workflow Phase 5"""
    
    def __init__(self):
        # Initialisation des composants
        self.client = DeepSeekClient()
        self.context_manager = TemporaryContextManager()
        self.personalization_engine = PersonalizationEngine(self.context_manager)
        self.insight_generator = InsightGenerator()
        
        # Agents
        self.intent_classifier = IntentClassifierAgent(self.client)
        self.entity_extractor = EntityExtractorAgent(self.client)
        self.query_builder = QueryBuilderAgent(self.client)
        self.search_executor = SearchExecutorAgent()
        self.response_generator = ResponseGeneratorAgent(self.client)
        
        print("🚀 Phase5WorkflowTester initialisé avec tous les composants")
    
    async def test_complete_workflow(self, test_cases: list) -> Dict[str, Any]:
        """Test du workflow complet sur plusieurs cas d'usage"""
        
        results = {
            "test_timestamp": datetime.now().isoformat(),
            "total_cases": len(test_cases),
            "results": [],
            "summary": {
                "successful": 0,
                "failed": 0,
                "average_processing_time_ms": 0,
                "total_insights_generated": 0,
                "total_suggestions_generated": 0
            }
        }
        
        total_time = 0
        
        for i, test_case in enumerate(test_cases):
            print(f"\n{'='*60}")
            print(f"🧪 Test Case {i+1}/{len(test_cases)}: {test_case['description']}")
            print(f"Message: '{test_case['message']}'")
            print(f"User ID: {test_case['user_id']}")
            
            try:
                result = await self._run_single_workflow(test_case)
                results["results"].append(result)
                
                if result["success"]:
                    results["summary"]["successful"] += 1
                    total_time += result["processing_time_ms"]
                    results["summary"]["total_insights_generated"] += result.get("insights_count", 0)
                    results["summary"]["total_suggestions_generated"] += result.get("suggestions_count", 0)
                else:
                    results["summary"]["failed"] += 1
                
                # Affichage résumé
                self._print_test_result(result)
                
            except Exception as e:
                print(f"❌ Erreur inattendue: {str(e)}")
                results["results"].append({
                    "test_case": test_case,
                    "success": False,
                    "error": str(e),
                    "processing_time_ms": 0
                })
                results["summary"]["failed"] += 1
        
        # Calcul moyennes
        if results["summary"]["successful"] > 0:
            results["summary"]["average_processing_time_ms"] = int(
                total_time / results["summary"]["successful"]
            )
        
        self._print_final_summary(results["summary"])
        return results
    
    async def _run_single_workflow(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Exécute le workflow complet pour un cas de test"""
        
        start_time = time.time()
        user_id = test_case["user_id"]
        message = test_case["message"]
        request_id = f"test_{int(time.time() * 1000)}_{user_id}"
        
        result = {
            "test_case": test_case,
            "request_id": request_id,
            "success": False,
            "steps": {},
            "final_response": None,
            "processing_time_ms": 0,
            "insights_count": 0,
            "suggestions_count": 0,
            "response_quality": None
        }
        
        try:
            # Récupération contexte utilisateur
            user_context = self.context_manager.get_user_context(user_id)
            personalization_context = self.personalization_engine.get_personalization_context(user_id)
            
            print(f"📝 Contexte utilisateur: {user_context.get('interaction_count', 0)} interactions")
            
            # ÉTAPE 1: Classification intention
            print("🎯 Étape 1: Classification intention...")
            step1_start = time.time()
            intent_result = await self.intent_classifier.classify_intent(message, request_id=request_id)
            step1_time = int((time.time() - step1_start) * 1000)
            
            result["steps"]["intent_classification"] = {
                "duration_ms": step1_time,
                "success": True,
                "result": intent_result
            }
            print(f"✅ Intent: {intent_result['intent_type']} (conf: {intent_result['confidence']:.2f})")
            
            # ÉTAPE 2: Extraction entités
            print("🔍 Étape 2: Extraction entités...")
            step2_start = time.time()
            entities_result = await self.entity_extractor.extract_entities(
                message, intent_result, request_id=request_id
            )
            step2_time = int((time.time() - step2_start) * 1000)
            
            result["steps"]["entity_extraction"] = {
                "duration_ms": step2_time,
                "success": True,
                "result": entities_result
            }
            print(f"✅ Entités: {list(entities_result.keys())}")
            
            # ÉTAPE 3: Génération requête
            print("🔧 Étape 3: Génération requête...")
            step3_start = time.time()
            generation_response = await self.query_builder.build_search_query(
                user_message=message,
                intent=intent_result,
                entities=entities_result,
                user_id=user_id,
                request_id=request_id
            )
            step3_time = int((time.time() - step3_start) * 1000)
            
            result["steps"]["query_generation"] = {
                "duration_ms": step3_time,
                "success": True,
                "result": {
                    "estimated_performance": generation_response.validation.estimated_performance,
                    "has_filters": bool(generation_response.search_query.filters),
                    "has_aggregations": bool(generation_response.search_query.aggregations)
                }
            }
            print(f"✅ Requête générée (perf: {generation_response.validation.estimated_performance})")
            
            # ÉTAPE 4: Exécution recherche (simulée ou réelle)
            print("🔍 Étape 4: Exécution recherche...")
            step4_start = time.time()
            
            if test_case.get("simulate_search_results"):
                # Simulation des résultats pour tests
                executor_response = self._create_mock_search_response(test_case["simulate_search_results"])
            else:
                executor_response = await self.search_executor.execute_search(
                    generation_response.search_query, request_id=request_id
                )
            
            step4_time = int((time.time() - step4_start) * 1000)
            
            result["steps"]["search_execution"] = {
                "duration_ms": step4_time,
                "success": executor_response.success,
                "result": {
                    "total_hits": executor_response.search_results.total_hits if executor_response.search_results else 0,
                    "has_results": executor_response.success and executor_response.search_results is not None
                }
            }
            
            if executor_response.success:
                print(f"✅ Recherche: {executor_response.search_results.total_hits} résultats")
            else:
                print(f"⚠️ Recherche échouée: {executor_response.error_message}")
            
            # ÉTAPE 5: Génération réponse
            print("💬 Étape 5: Génération réponse...")
            step5_start = time.time()
            
            response_content, response_quality, generation_metrics = await self.response_generator.generate_response(
                user_message=message,
                intent=intent_result,
                entities=entities_result,
                search_results=executor_response.search_results if executor_response.success else None,
                user_context=personalization_context,
                request_id=request_id
            )
            
            step5_time = int((time.time() - step5_start) * 1000)
            
            result["steps"]["response_generation"] = {
                "duration_ms": step5_time,
                "success": True,
                "result": {
                    "message_length": len(response_content.message),
                    "insights_count": len(response_content.insights),
                    "suggestions_count": len(response_content.suggestions),
                    "quality_score": response_quality.relevance_score
                }
            }
            
            print(f"✅ Réponse: {len(response_content.message)} chars, {len(response_content.insights)} insights, {len(response_content.suggestions)} suggestions")
            
            # Mise à jour contexte
            self.context_manager.update_context(
                user_id=user_id,
                message=message,
                intent=intent_result,
                entities=entities_result,
                search_results=executor_response.search_results.__dict__ if executor_response.search_results else None,
                response_generated=response_content.__dict__
            )
            
            # Finalisation résultats
            total_time = int((time.time() - start_time) * 1000)
            
            result.update({
                "success": True,
                "processing_time_ms": total_time,
                "insights_count": len(response_content.insights),
                "suggestions_count": len(response_content.suggestions),
                "response_quality": {
                    "relevance_score": response_quality.relevance_score,
                    "completeness": response_quality.completeness,
                    "actionability": response_quality.actionability
                },
                "final_response": {
                    "message": response_content.message,
                    "structured_data": response_content.structured_data.__dict__ if response_content.structured_data else None,
                    "insights": [insight.__dict__ for insight in response_content.insights],
                    "suggestions": [suggestion.__dict__ for suggestion in response_content.suggestions]
                }
            })
            
            return result
            
        except Exception as e:
            total_time = int((time.time() - start_time) * 1000)
            result.update({
                "success": False,
                "error": str(e),
                "processing_time_ms": total_time
            })
            return result
    
    def _create_mock_search_response(self, mock_data: Dict[str, Any]):
        """Crée une réponse de recherche simulée pour les tests"""
        
        from conversation_service.models.contracts.search_service import SearchResponse
        
        # Simulation basique d'une réponse
        mock_response = type('MockSearchExecutorResponse', (), {
            'success': mock_data.get('success', True),
            'search_results': None,
            'error_message': mock_data.get('error_message'),
            'execution_time_ms': mock_data.get('execution_time_ms', 100)
        })()
        
        if mock_data.get('success', True):
            mock_response.search_results = SearchResponse(
                hits=[
                    {"_source": {"amount": -45.99, "merchant_name": "Amazon", "date": "2024-08-15"}}
                    for _ in range(mock_data.get('total_hits', 5))
                ],
                total_hits=mock_data.get('total_hits', 5),
                took_ms=50,
                aggregations=mock_data.get('aggregations', {
                    "total_spent": {"value": -mock_data.get('total_amount', 234.56)}
                }) if mock_data.get('has_aggregations', True) else None
            )
        
        return mock_response
    
    def _print_test_result(self, result: Dict[str, Any]):
        """Affiche le résultat d'un test"""
        
        if result["success"]:
            print(f"✅ Test réussi en {result['processing_time_ms']}ms")
            print(f"   📊 Qualité: {result['response_quality']['relevance_score']:.2f}")
            print(f"   💡 Insights: {result['insights_count']}")
            print(f"   🎯 Suggestions: {result['suggestions_count']}")
            print(f"   💬 Réponse: {result['final_response']['message'][:100]}...")
        else:
            print(f"❌ Test échoué: {result.get('error', 'Erreur inconnue')}")
    
    def _print_final_summary(self, summary: Dict[str, Any]):
        """Affiche le résumé final des tests"""
        
        print(f"\n{'='*60}")
        print("📊 RÉSUMÉ FINAL DES TESTS PHASE 5")
        print(f"{'='*60}")
        print(f"✅ Tests réussis: {summary['successful']}")
        print(f"❌ Tests échoués: {summary['failed']}")
        print(f"⏱️  Temps moyen: {summary['average_processing_time_ms']}ms")
        print(f"💡 Total insights: {summary['total_insights_generated']}")
        print(f"🎯 Total suggestions: {summary['total_suggestions_generated']}")
        
        success_rate = (summary['successful'] / (summary['successful'] + summary['failed'])) * 100
        print(f"📈 Taux de succès: {success_rate:.1f}%")
        
        if success_rate >= 80:
            print("🎉 Phase 5 fonctionne correctement!")
        elif success_rate >= 60:
            print("⚠️ Phase 5 nécessite des améliorations")
        else:
            print("🚨 Phase 5 nécessite des corrections importantes")
    
    async def test_context_persistence(self, user_id: int = 999) -> Dict[str, Any]:
        """Test de la persistance du contexte entre les interactions"""
        
        print(f"\n{'='*60}")
        print("🧠 TEST DE PERSISTANCE DU CONTEXTE")
        print(f"{'='*60}")
        
        messages = [
            "Combien j'ai dépensé chez Amazon ce mois ?",
            "Et chez Carrefour ?",
            "Compare ces deux marchands",
            "Montre-moi mes dépenses de la semaine dernière"
        ]
        
        context_evolution = []
        
        for i, message in enumerate(messages):
            print(f"\n🔄 Interaction {i+1}: '{message}'")
            
            # État du contexte avant
            context_before = self.context_manager.get_user_context(user_id)
            
            # Simulation d'une interaction
            test_case = {
                "user_id": user_id,
                "message": message,
                "description": f"Interaction {i+1} - Test contexte",
                "simulate_search_results": {
                    "success": True,
                    "total_hits": 5,
                    "total_amount": 156.78,
                    "has_aggregations": True
                }
            }
            
            result = await self._run_single_workflow(test_case)
            
            # État du contexte après
            context_after = self.context_manager.get_user_context(user_id)
            
            context_evolution.append({
                "interaction": i+1,
                "message": message,
                "context_before": {
                    "interaction_count": context_before.get("interaction_count", 0),
                    "recent_queries": len(context_before.get("recent_queries", [])),
                    "merchants_known": len(context_before.get("entity_history", {}).get("merchants", {}))
                },
                "context_after": {
                    "interaction_count": context_after.get("interaction_count", 0),
                    "recent_queries": len(context_after.get("recent_queries", [])),
                    "merchants_known": len(context_after.get("entity_history", {}).get("merchants", {}))
                },
                "success": result["success"]
            })
            
            print(f"   📈 Interactions: {context_before.get('interaction_count', 0)} → {context_after.get('interaction_count', 0)}")
            print(f"   🏪 Marchands connus: {len(context_before.get('entity_history', {}).get('merchants', {}))} → {len(context_after.get('entity_history', {}).get('merchants', {}))}")
        
        # Vérification finale
        final_context = self.context_manager.get_context_summary(user_id)
        print(f"\n📊 CONTEXTE FINAL:")
        print(f"   Interactions totales: {final_context.get('interaction_count', 0)}")
        print(f"   Utilisateur récurrent: {final_context.get('is_returning_user', False)}")
        print(f"   Marchands fréquents: {final_context.get('frequent_merchants', [])}")
        print(f"   Intentions préférées: {final_context.get('preferred_intents', [])}")
        
        return {
            "context_evolution": context_evolution,
            "final_context": final_context,
            "success": all(ce["success"] for ce in context_evolution)
        }
    
    async def test_template_personalization(self) -> Dict[str, Any]:
        """Test de la personnalisation des templates"""
        
        print(f"\n{'='*60}")
        print("🎨 TEST DE PERSONNALISATION DES TEMPLATES")
        print(f"{'='*60}")
        
        # Différents profils utilisateur
        user_profiles = [
            {
                "user_id": 1001,
                "profile": "Novice",
                "context": {
                    "is_returning_user": False,
                    "interaction_count": 1,
                    "detail_level": "basic",
                    "communication_style": "concise"
                }
            },
            {
                "user_id": 1002,
                "profile": "Expert",
                "context": {
                    "is_returning_user": True,
                    "interaction_count": 25,
                    "detail_level": "advanced",
                    "communication_style": "detailed",
                    "frequent_merchants": ["Amazon", "Carrefour", "FNAC"]
                }
            },
            {
                "user_id": 1003,
                "profile": "Régulier",
                "context": {
                    "is_returning_user": True,
                    "interaction_count": 8,
                    "detail_level": "medium",
                    "communication_style": "balanced"
                }
            }
        ]
        
        message = "Combien j'ai dépensé chez Amazon ce mois ?"
        intent = {"intent_type": "SEARCH_BY_MERCHANT", "confidence": 0.9}
        entities = {"merchants": ["Amazon"], "dates": {"original": "ce mois"}}
        analysis_data = {
            "has_results": True,
            "primary_entity": "Amazon", 
            "total_amount": 234.56,
            "transaction_count": 12
        }
        
        template_results = []
        
        for profile in user_profiles:
            print(f"\n👤 Profil: {profile['profile']} (User {profile['user_id']})")
            
            # Générer template personnalisé
            template = get_response_template(
                intent_type=intent["intent_type"],
                user_message=message,
                entities=entities,
                analysis_data=analysis_data,
                user_context=profile["context"],
                use_personalization=True
            )
            
            template_results.append({
                "profile": profile["profile"],
                "user_id": profile["user_id"],
                "template_length": len(template),
                "contains_personalization": "Personnalisation:" in template,
                "template_preview": template[:200] + "..." if len(template) > 200 else template
            })
            
            print(f"   📝 Template généré: {len(template)} caractères")
            print(f"   🎯 Personnalisé: {'Oui' if 'Personnalisation:' in template else 'Non'}")
            print(f"   👀 Aperçu: {template[:100]}...")
        
        return {
            "profiles_tested": len(user_profiles),
            "template_results": template_results,
            "personalization_working": all(tr["contains_personalization"] for tr in template_results)
        }


async def main():
    """Fonction principale de test"""
    
    print("🚀 DÉMARRAGE DES TESTS COMPLETS PHASE 5")
    print("="*60)
    
    # Initialisation du testeur
    tester = Phase5WorkflowTester()
    
    # Cas de test variés
    test_cases = [
        {
            "description": "Recherche marchand avec résultats",
            "user_id": 123,
            "message": "Combien j'ai dépensé chez Amazon ce mois ?",
            "simulate_search_results": {
                "success": True,
                "total_hits": 12,
                "total_amount": 234.56,
                "has_aggregations": True
            }
        },
        {
            "description": "Analyse des dépenses globales",
            "user_id": 124,
            "message": "Mes dépenses de ce mois",
            "simulate_search_results": {
                "success": True,
                "total_hits": 45,
                "total_amount": 1250.30,
                "has_aggregations": True
            }
        },
        {
            "description": "Recherche sans résultats",
            "user_id": 125,
            "message": "Mes achats chez Tesla",
            "simulate_search_results": {
                "success": True,
                "total_hits": 0,
                "total_amount": 0,
                "has_aggregations": False
            }
        },
        {
            "description": "Erreur de recherche",
            "user_id": 126,
            "message": "Mes transactions de la semaine",
            "simulate_search_results": {
                "success": False,
                "error_message": "Service temporairement indisponible"
            }
        },
        {
            "description": "Demande de solde",
            "user_id": 127,
            "message": "Quel est mon solde ?",
            "simulate_search_results": {
                "success": False,
                "error_message": "Informations de solde non disponibles"
            }
        }
    ]
    
    try:
        # Test du workflow complet
        workflow_results = await tester.test_complete_workflow(test_cases)
        
        # Test de la persistance du contexte
        context_results = await tester.test_context_persistence()
        
        # Test de la personnalisation
        personalization_results = await tester.test_template_personalization()
        
        # Résumé global
        print(f"\n{'='*60}")
        print("🏆 RÉSUMÉ GLOBAL DES TESTS PHASE 5")
        print(f"{'='*60}")
        print(f"Workflow complet: {'✅' if workflow_results['summary']['successful'] > 0 else '❌'}")
        print(f"Persistance contexte: {'✅' if context_results['success'] else '❌'}")
        print(f"Personnalisation: {'✅' if personalization_results['personalization_working'] else '❌'}")
        
        # Sauvegarde des résultats
        with open("test_results_phase5.json", "w", encoding="utf-8") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "workflow_results": workflow_results,
                "context_results": context_results,
                "personalization_results": personalization_results
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n📁 Résultats sauvegardés dans: test_results_phase5.json")
        
    except Exception as e:
        print(f"💥 Erreur lors des tests: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())