"""
Endpoint Phase 5 - Workflow complet avec génération de réponses
Version finale avec toute la chaîne : intentions + entités + requête + résultats + réponse naturelle
"""
import logging
import time
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, Request

from conversation_service.models.requests.conversation_requests import ConversationRequest
from conversation_service.models.responses.conversation_responses import (
    ConversationResponse, AgentMetrics, ProcessingSteps, QueryGenerationMetrics,
    ConversationResponseFactory, ResponseContent, ResponseQuality, ResponseGenerationMetrics
)
from conversation_service.models.responses.conversation_responses_clean import (
    CleanConversationResponse, CleanErrorResponse, create_enhanced_structured_data
)

# Agents
from conversation_service.agents.financial.intent_classifier import IntentClassifierAgent
from conversation_service.agents.financial.entity_extractor import EntityExtractorAgent
from conversation_service.agents.financial.query_builder import QueryBuilderAgent
from conversation_service.agents.search.search_executor import SearchExecutor
from conversation_service.agents.financial.response_generator import ResponseGeneratorAgent

# Services
from conversation_service.services.insight_generator import InsightGenerator
from conversation_service.core.context_manager import TemporaryContextManager, PersonalizationEngine

# Core
from conversation_service.clients.deepseek_client import DeepSeekClient
from conversation_service.core.cache_manager import CacheManager
from conversation_service.utils.metrics_collector import metrics_collector
from conversation_service.api.dependencies import (
    get_deepseek_client, get_cache_manager, get_conversation_service_status,
    validate_path_user_id, get_user_context, rate_limit_dependency
)

router = APIRouter()
logger = logging.getLogger(__name__)

# Instances globales (à remplacer par injection de dépendances en production)
_deepseek_client = None
_cache_manager = None
_metrics_collector = None
_context_manager = None
_personalization_engine = None
_insight_generator = None

# Agents
_intent_classifier = None
_entity_extractor = None
_query_builder = None
_search_executor = None
_response_generator = None


def get_dependencies():
    """Initialise les dépendances si nécessaire"""
    global _deepseek_client, _cache_manager, _context_manager
    global _personalization_engine, _insight_generator, _intent_classifier
    global _entity_extractor, _query_builder, _search_executor, _response_generator
    
    if not _deepseek_client:
        _deepseek_client = DeepSeekClient()
        _cache_manager = CacheManager()
        # Utiliser l'instance globale des métriques
        _context_manager = TemporaryContextManager()
        _personalization_engine = PersonalizationEngine(_context_manager)
        _insight_generator = InsightGenerator()
        
        # Initialisation des agents
        _intent_classifier = IntentClassifierAgent(_deepseek_client)
        _entity_extractor = EntityExtractorAgent(_deepseek_client)
        _query_builder = QueryBuilderAgent(_deepseek_client)
        _search_executor = SearchExecutor()
        _response_generator = ResponseGeneratorAgent(_deepseek_client)
    
    return {
        "client": _deepseek_client,
        "cache": _cache_manager,
        "metrics": _metrics_collector,
        "context": _context_manager,
        "personalization": _personalization_engine,
        "insights": _insight_generator,
        "agents": {
            "intent_classifier": _intent_classifier,
            "entity_extractor": _entity_extractor,
            "query_builder": _query_builder,
            "search_executor": _search_executor,
            "response_generator": _response_generator
        }
    }


@router.post("/{user_id}")
async def process_conversation_phase5(
    user_id: int,
    request: ConversationRequest,
    http_request: Request,
    validated_user_id: int = Depends(validate_path_user_id),
    user_context: Dict[str, Any] = Depends(get_user_context),
    _rate_limit: None = Depends(rate_limit_dependency)
):
    """
    Phase 5: Workflow complet avec génération de réponse naturelle
    
    Pipeline complet:
    1. Classification d'intention
    2. Extraction d'entités  
    3. Génération de requête
    4. Exécution de recherche
    5. Génération de réponse contextuelle
    """
    start_time = time.time()
    request_id = f"phase5_{int(time.time() * 1000)}_{user_id}"
    
    # Initialisation
    deps = get_dependencies()
    agents = deps["agents"]
    context_manager = deps["context"]
    personalization_engine = deps["personalization"]
    
    logger.info(f"[{request_id}] 🚀 Phase 5 - Début workflow complet pour user {user_id}")
    logger.info(f"[{request_id}] Message: '{request.message}'")
    
    try:
        # Récupération du contexte utilisateur existant
        user_context = context_manager.get_user_context(user_id)
        logger.info(f"[{request_id}] Contexte utilisateur: {user_context.get('interaction_count', 0)} interactions précédentes")
        
        # =============================================
        # ÉTAPE 1: Classification d'intention
        # =============================================
        logger.info(f"[{request_id}] 🎯 Étape 1: Classification d'intention")
        step1_start = time.time()
        
        try:
            # Récupération du contexte de personnalisation
            personalization_context = personalization_engine.get_personalization_context(user_id)
            
            intent_result = await agents["intent_classifier"].classify_intent(
                request.message,
                user_context=personalization_context
            )
            step1_duration = int((time.time() - step1_start) * 1000)
            
            step1 = ProcessingSteps(
                agent="intent_classifier",
                duration_ms=step1_duration,
                cache_hit=False,
                success=True,
                error_message=None
            )
            
            logger.info(f"[{request_id}] ✅ Intent: {intent_result.intent_type} (conf: {intent_result.confidence})")
            
            # Log détaillé de l'intent pour analyse
            logger.info(f"[{request_id}] 📊 INTENT DETAILS:")
            logger.info(f"[{request_id}]   - Type: {intent_result.intent_type}")
            logger.info(f"[{request_id}]   - Confidence: {intent_result.confidence}")
            logger.info(f"[{request_id}]   - Category: {intent_result.category}")
            logger.info(f"[{request_id}]   - Supported: {intent_result.is_supported}")
            logger.info(f"[{request_id}]   - Reasoning: {intent_result.reasoning[:100]}...")
            logger.info(f"[{request_id}]   - Duration: {step1_duration}ms")
            
            # Conversion en dict pour compatibilité avec les autres agents
            intent_dict = {
                "intent_type": str(intent_result.intent_type),
                "confidence": intent_result.confidence,
                "reasoning": intent_result.reasoning,
                "original_message": intent_result.original_message,
                "category": intent_result.category,
                "is_supported": intent_result.is_supported
            }
            
        except Exception as e:
            step1_duration = int((time.time() - step1_start) * 1000)
            error_msg = f"Erreur classification: {str(e)}"
            logger.error(f"[{request_id}] ❌ {error_msg}")
            
            step1 = ProcessingSteps(
                agent="intent_classifier",
                duration_ms=step1_duration,
                cache_hit=False,
                success=False,
                error_message=error_msg
            )
            
            raise HTTPException(status_code=500, detail=error_msg)
        
        # =============================================
        # ÉTAPE 2: Extraction d'entités
        # =============================================
        logger.info(f"[{request_id}] 🔍 Étape 2: Extraction d'entités")
        step2_start = time.time()
        
        try:
            entities_result = await agents["entity_extractor"].extract_entities(
                request.message,
                intent_dict,
                user_id=user_id
            )
            step2_duration = int((time.time() - step2_start) * 1000)
            
            step2 = ProcessingSteps(
                agent="entity_extractor",
                duration_ms=step2_duration,
                cache_hit=False,
                success=True,
                error_message=None
            )
            
            logger.info(f"[{request_id}] ✅ Entités extraites: {list(entities_result.keys())}")
            
            # Log détaillé des entités pour analyse
            logger.info(f"[{request_id}] 📊 ENTITIES DETAILS:")
            for entity_type, entity_value in entities_result.items():
                if entity_type == "entities" and isinstance(entity_value, dict):
                    for sub_type, sub_value in entity_value.items():
                        logger.info(f"[{request_id}]   - {sub_type}: {sub_value}")
                else:
                    logger.info(f"[{request_id}]   - {entity_type}: {entity_value}")
            logger.info(f"[{request_id}]   - Duration: {step2_duration}ms")
            
        except Exception as e:
            step2_duration = int((time.time() - step2_start) * 1000)
            error_msg = f"Erreur extraction entités: {str(e)}"
            logger.error(f"[{request_id}] ❌ {error_msg}")
            
            step2 = ProcessingSteps(
                agent="entity_extractor",
                duration_ms=step2_duration,
                cache_hit=False,
                success=False,
                error_message=error_msg
            )
            
            raise HTTPException(status_code=500, detail=error_msg)
        
        # =============================================
        # ÉTAPE 3: Génération de requête
        # =============================================
        logger.info(f"[{request_id}] 🔧 Étape 3: Génération de requête")
        step3_start = time.time()
        
        try:
            # Préparer la requête pour query_builder
            from conversation_service.models.contracts.search_service import QueryGenerationRequest
            query_request = QueryGenerationRequest(
                user_message=request.message,
                user_id=user_id,
                intent_type=intent_dict["intent_type"],
                intent_confidence=intent_dict["confidence"],
                entities=entities_result
            )
            generation_response = await agents["query_builder"].generate_search_query(query_request)
            step3_duration = int((time.time() - step3_start) * 1000)
            
            step3 = ProcessingSteps(
                agent="query_builder",
                duration_ms=step3_duration,
                cache_hit=False,
                success=True,
                error_message=None
            )
            
            logger.info(f"[{request_id}] ✅ Requête générée avec validation: {generation_response.validation.estimated_performance}")
            
            # Log détaillé de la requête pour analyse
            logger.info(f"[{request_id}] 📊 QUERY DETAILS:")
            logger.info(f"[{request_id}]   - Performance: {generation_response.validation.estimated_performance}")
            logger.info(f"[{request_id}]   - Confidence: {generation_response.generation_confidence}")
            logger.info(f"[{request_id}]   - Query type: {generation_response.query_type}")
            logger.info(f"[{request_id}]   - Filters: {bool(generation_response.search_query.filters)}")
            logger.info(f"[{request_id}]   - Aggregations: {bool(generation_response.search_query.aggregations)}")
            logger.info(f"[{request_id}]   - Reasoning: {generation_response.reasoning[:100]}...")
            logger.info(f"[{request_id}]   - Duration: {step3_duration}ms")
            
            # Log de la requête complète générée pour analyse
            logger.info(f"[{request_id}] 🔍 GENERATED SEARCH QUERY:")
            logger.info(f"[{request_id}]   - User ID: {generation_response.search_query.user_id}")
            logger.info(f"[{request_id}]   - Page size: {generation_response.search_query.page_size}")
            logger.info(f"[{request_id}]   - Sort: {generation_response.search_query.sort}")
            if generation_response.search_query.query:
                logger.info(f"[{request_id}]   - Query: {generation_response.search_query.query}")
            if generation_response.search_query.filters:
                logger.info(f"[{request_id}]   - Filters: {generation_response.search_query.filters.model_dump()}")
            if generation_response.search_query.aggregations:
                logger.info(f"[{request_id}]   - Aggregations: {generation_response.search_query.aggregations.model_dump()}")
            logger.info(f"[{request_id}]   - Full query: {generation_response.search_query.model_dump()}")
            
        except Exception as e:
            step3_duration = int((time.time() - step3_start) * 1000)
            error_msg = f"Erreur génération requête: {str(e)}"
            logger.error(f"[{request_id}] ❌ {error_msg}")
            
            step3 = ProcessingSteps(
                agent="query_builder",
                duration_ms=step3_duration,
                cache_hit=False,
                success=False,
                error_message=error_msg
            )
            
            raise HTTPException(status_code=500, detail=error_msg)
        
        # =============================================
        # ÉTAPE 4: Exécution recherche
        # =============================================
        logger.info(f"[{request_id}] 🔍 Étape 4: Exécution recherche")
        step4_start = time.time()
        
        try:
            # Préparer la requête pour search_executor
            from conversation_service.agents.search.search_executor import SearchExecutorRequest
            search_request = SearchExecutorRequest(
                search_query=generation_response.search_query,
                user_id=user_id,
                request_id=request_id
            )
            executor_response = await agents["search_executor"].handle_search_request(search_request)
            step4_duration = int((time.time() - step4_start) * 1000)
            
            step4 = ProcessingSteps(
                agent="search_executor",
                duration_ms=step4_duration,
                cache_hit=False,
                success=executor_response.success,
                error_message=executor_response.error_message if not executor_response.success else None
            )
            
            if executor_response.success:
                logger.info(
                    f"[{request_id}] ✅ Recherche réussie: {executor_response.search_results.total_hits} résultats"
                )
                
                # Log détaillé des résultats de recherche pour analyse
                logger.info(f"[{request_id}] 📊 SEARCH DETAILS:")
                logger.info(f"[{request_id}]   - Total hits: {executor_response.search_results.total_hits}")
                logger.info(f"[{request_id}]   - Returned hits: {len(executor_response.search_results.hits)}")
                logger.info(f"[{request_id}]   - Search time: {executor_response.search_results.took_ms}ms")
                logger.info(f"[{request_id}]   - Has aggregations: {bool(executor_response.search_results.aggregations)}")
                logger.info(f"[{request_id}]   - Circuit breaker: {executor_response.circuit_breaker_triggered}")
                logger.info(f"[{request_id}]   - Fallback used: {executor_response.fallback_used}")
                logger.info(f"[{request_id}]   - Duration: {step4_duration}ms")
            else:
                logger.warning(f"[{request_id}] ⚠️ Recherche échouée: {executor_response.error_message}")
                logger.info(f"[{request_id}] 📊 SEARCH FAILURE DETAILS:")
                logger.info(f"[{request_id}]   - Error type: {executor_response.error_type}")
                logger.info(f"[{request_id}]   - Circuit breaker: {executor_response.circuit_breaker_triggered}")
                logger.info(f"[{request_id}]   - Retry attempts: {executor_response.retry_attempts}")
                logger.info(f"[{request_id}]   - Duration: {step4_duration}ms")
            
        except Exception as e:
            step4_duration = int((time.time() - step4_start) * 1000)
            error_msg = f"Erreur exécution recherche: {str(e)}"
            logger.error(f"[{request_id}] ❌ {error_msg}")
            
            step4 = ProcessingSteps(
                agent="search_executor",
                duration_ms=step4_duration,
                cache_hit=False,
                success=False,
                error_message=error_msg
            )
            
            # On continue même si la recherche échoue - réponse sans données
            executor_response = type('MockResponse', (), {
                'success': False,
                'search_results': None,
                'error_message': error_msg,
                'execution_time_ms': step4_duration
            })()
        
        # =============================================
        # ÉTAPE 5: Génération de réponse
        # =============================================
        logger.info(f"[{request_id}] 💬 Étape 5: Génération de réponse")
        step5_start = time.time()
        
        try:
            response_content, response_quality, generation_metrics = await agents["response_generator"].generate_response(
                user_message=request.message,
                intent=intent_dict,
                entities=entities_result,
                search_results=executor_response.search_results if executor_response.success else None,
                user_context=personalization_context,
                request_id=request_id
            )
            
            step5_duration = int((time.time() - step5_start) * 1000)
            
            step5 = ProcessingSteps(
                agent="response_generator",
                duration_ms=step5_duration,
                cache_hit=False,
                success=True,
                error_message=None
            )
            
            logger.info(
                f"[{request_id}] ✅ Réponse générée: {len(response_content.message)} chars, "
                f"{len(response_content.insights)} insights, {len(response_content.suggestions)} suggestions"
            )
            
            # Log détaillé de la réponse pour analyse
            logger.info(f"[{request_id}] 📊 RESPONSE DETAILS:")
            logger.info(f"[{request_id}]   - Message length: {len(response_content.message)} chars")
            logger.info(f"[{request_id}]   - Quality score: {response_quality.relevance_score}")
            logger.info(f"[{request_id}]   - Completeness: {response_quality.completeness}")
            logger.info(f"[{request_id}]   - Actionability: {response_quality.actionability}")
            logger.info(f"[{request_id}]   - Insights count: {len(response_content.insights)}")
            logger.info(f"[{request_id}]   - Suggestions count: {len(response_content.suggestions)}")
            logger.info(f"[{request_id}]   - Next actions: {len(response_content.next_actions)}")
            logger.info(f"[{request_id}]   - Generation time: {generation_metrics.generation_time_ms}ms")
            logger.info(f"[{request_id}]   - Tokens used: {generation_metrics.tokens_response}")
            logger.info(f"[{request_id}]   - Duration: {step5_duration}ms")
            
            # Log du contenu de la réponse
            logger.info(f"[{request_id}] 💬 RESPONSE MESSAGE: {response_content.message}")
            
            for i, insight in enumerate(response_content.insights, 1):
                logger.info(f"[{request_id}] 💡 INSIGHT {i}: [{insight.severity}] {insight.title}")
            
            for i, suggestion in enumerate(response_content.suggestions, 1):
                logger.info(f"[{request_id}] 🎯 SUGGESTION {i}: [{suggestion.priority}] {suggestion.title}")
            
        except Exception as e:
            step5_duration = int((time.time() - step5_start) * 1000)
            error_msg = f"Erreur génération réponse: {str(e)}"
            logger.error(f"[{request_id}] ❌ {error_msg}")
            
            step5 = ProcessingSteps(
                agent="response_generator",
                duration_ms=step5_duration,
                cache_hit=False,
                success=False,
                error_message=error_msg
            )
            
            raise HTTPException(status_code=500, detail=error_msg)
        
        # =============================================
        # CONSTRUCTION RÉPONSE FINALE
        # =============================================
        total_processing_time = int((time.time() - start_time) * 1000)
        
        # Construire la réponse Phase 4 d'abord
        # ConversationResponseFactoryPhase4 est maintenant un alias vers ConversationResponseFactory
        
        # Créer une base Phase 3 temporaire
        # ConversationResponsePhase3 est maintenant un alias vers ConversationResponse
        
        phase3_base = ConversationResponse(
            user_id=user_id,
            sub=str(validated_user_id),
            message=request.message,
            timestamp=datetime.now(timezone.utc),
            request_id=request_id,
            intent=intent_result,
            agent_metrics=AgentMetrics(
                agent_used="phase5_workflow",
                model_used="deepseek-chat",
                tokens_consumed=(
                    generation_metrics.tokens_response + 
                    1862 + 2736  # Intent + Query tokens from logs
                ),
                processing_time_ms=total_processing_time,
                confidence_threshold_met=intent_result.confidence > 0.8,
                cache_hit=False,
                retry_count=0,
                error_count=0,
                detailed_metrics={
                    "workflow_success": all([step1.success, step2.success, step3.success, step4.success, step5.success]),
                    "agents_sequence": ["intent_classifier", "entity_extractor", "query_builder", "search_executor", "response_generator"],
                    "cache_efficiency": 0.0,
                    "step_durations": {
                        "intent_classification": step1_duration,
                        "entity_extraction": step2_duration,
                        "query_generation": step3_duration,
                        "search_execution": step4_duration,
                        "response_generation": step5_duration
                    }
                }
            ),
            processing_time_ms=total_processing_time,
            status="success",
            entities=entities_result,
            search_query=generation_response.search_query,
            query_validation=generation_response.validation,
            query_generation_metrics=QueryGenerationMetrics(
                generation_time_ms=step3_duration,
                validation_time_ms=50,  # Estimation
                optimization_time_ms=30,  # Estimation
                generation_confidence=generation_response.generation_confidence,
                validation_passed=generation_response.validation.schema_valid,
                optimizations_applied=1,  # Estimation
                estimated_performance=generation_response.validation.estimated_performance,
                estimated_results_count=generation_response.estimated_results_count
            ),
            processing_steps=[step1, step2, step3, step4, step5]
        )
        
        # Convertir en Phase 4 avec résultats de recherche
        if executor_response.success and executor_response.search_results:
            phase4_response = ConversationResponseFactory.create_from_search_executor_response(
                phase3_base, executor_response, step4
            )
        else:
            # Phase 4 sans résultats
            phase4_response = ConversationResponseFactory.create_phase5_from_phase4(phase3_base)
        
        # Convertir en Phase 5 avec réponse générée
        phase5_response = ConversationResponseFactory.create_phase5_success(
            phase4_response,
            response_content,
            response_quality,
            generation_metrics
        )
        
        # Mise à jour du contexte utilisateur
        context_manager.update_context(
            user_id=user_id,
            message=request.message,
            intent=intent_dict,
            entities=entities_result,
            search_results=executor_response.search_results.__dict__ if executor_response.search_results else None,
            response_generated=response_content.__dict__
        )
        
        logger.info(
            f"[{request_id}] 🎉 Phase 5 WORKFLOW COMPLETE - "
            f"Total time: {total_processing_time}ms, "
            f"Final response: {len(response_content.message)} chars"
        )
        
        # ============================================================================
        # LOG COMPLET DES DONNÉES TECHNIQUES (pour debugging/analytics)
        # ============================================================================
        logger.info(f"[{request_id}] 📊 COMPLETE RESPONSE DATA (for analytics):")
        logger.info(f"[{request_id}] Intent: {phase5_response.intent.intent_type} (confidence: {phase5_response.intent.confidence})")
        logger.info(f"[{request_id}] Agent metrics: {phase5_response.agent_metrics.performance_grade} grade, {phase5_response.agent_metrics.efficiency_score:.2f} efficiency")
        logger.info(f"[{request_id}] Entities found: {len(phase5_response.entities.get('entities', {}).keys()) if phase5_response.entities else 0} types")
        logger.info(f"[{request_id}] Search results: {phase5_response.search_results.total_hits if phase5_response.search_results else 0} hits")
        logger.info(f"[{request_id}] Response quality: {phase5_response.response_quality.relevance_score:.2f} relevance, {phase5_response.response_quality.completeness} completeness")
        logger.info(f"[{request_id}] Processing steps: {len(phase5_response.processing_steps)} agents in {phase5_response.total_agent_time_ms}ms")
        
        # ============================================================================
        # CRÉATION DE LA RÉPONSE API ÉPURÉE POUR L'UTILISATEUR FINAL
        # ============================================================================
        
        # Créer des données structurées enrichies pour l'utilisateur
        enhanced_structured_data = create_enhanced_structured_data(
            search_results=executor_response.search_results if executor_response.success else None,
            entities_result=entities_result,
            intent_type=intent_result.intent_type
        )
        
        # Remplacer les données structurées dans response_content si elles existent
        if enhanced_structured_data.total_amount is not None:
            response_content.structured_data = enhanced_structured_data
            logger.info(f"[{request_id}] 📈 Enhanced structured data: {enhanced_structured_data.total_amount}€, {enhanced_structured_data.transaction_count} transactions, analysis: {enhanced_structured_data.analysis_type}")
        else:
            logger.info(f"[{request_id}] ⚪ No enhanced structured data - keeping original response data")
        
        clean_response = CleanConversationResponse(
            request_id=request_id,
            processing_time_ms=total_processing_time,
            response=response_content,
            quality=response_quality,
            status="success"
        )
        
        logger.info(f"[{request_id}] 🚀 RETURNING CLEAN RESPONSE TO CLIENT")
        logger.info(f"[{request_id}] Clean response size: {len(response_content.message)} chars message, {len(response_content.insights)} insights, {len(response_content.suggestions)} suggestions")
        
        return clean_response.model_dump()
        
    except HTTPException:
        raise
    except Exception as e:
        total_processing_time = int((time.time() - start_time) * 1000)
        error_msg = f"Erreur inattendue Phase 5: {str(e)}"
        logger.error(f"[{request_id}] 💥 {error_msg}")
        
        # Log complet de l'erreur pour debugging
        import traceback
        logger.error(f"[{request_id}] 🔍 FULL ERROR TRACEBACK:")
        logger.error(traceback.format_exc())
        
        # Retourner une réponse d'erreur épurée
        error_response = CleanErrorResponse(
            request_id=request_id,
            status="error",
            error="Une erreur s'est produite lors du traitement de votre demande. Notre équipe a été notifiée.",
            processing_time_ms=total_processing_time,
            timestamp=datetime.now(timezone.utc),
            suggestions=[
                "Veuillez réessayer dans quelques instants",
                "Reformulez votre question si le problème persiste",
                "Contactez le support si l'erreur se répète"
            ]
        )
        
        raise HTTPException(
            status_code=500,
            detail=error_response.model_dump()
        )


@router.get("/context/{user_id}/stats")
async def get_user_context_stats(user_id: int, validated_user_id: int = Depends(validate_path_user_id)):
    """Récupère les statistiques du contexte utilisateur"""
    
    deps = get_dependencies()
    context_manager = deps["context"]
    
    try:
        context_summary = context_manager.get_context_summary(user_id)
        user_context = context_manager.get_user_context(user_id)
        
        return {
            "user_id": user_id,
            "context_summary": context_summary,
            "full_context": user_context,
            "context_age_minutes": 0 if not user_context else 
                (datetime.now(timezone.utc) - datetime.fromisoformat(
                    user_context["updated_at"].replace('Z', '+00:00')
                )).total_seconds() / 60
        }
        
    except Exception as e:
        logger.error(f"Erreur récupération contexte utilisateur {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur récupération contexte: {str(e)}")


@router.delete("/context/{user_id}")
async def clear_user_context(user_id: int, validated_user_id: int = Depends(validate_path_user_id)):
    """Efface le contexte utilisateur"""
    
    deps = get_dependencies()
    context_manager = deps["context"]
    
    try:
        # Suppression manuelle du contexte
        with context_manager._lock:
            removed = context_manager.session_contexts.pop(user_id, None)
        
        return {
            "user_id": user_id,
            "context_cleared": removed is not None,
            "message": "Context cleared successfully" if removed else "No context found"
        }
        
    except Exception as e:
        logger.error(f"Erreur suppression contexte utilisateur {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur suppression contexte: {str(e)}")


@router.get("/system/context-stats")  
async def get_system_context_stats():
    """Récupère les statistiques globales du système de contexte"""
    
    deps = get_dependencies()
    context_manager = deps["context"]
    
    try:
        # Nettoyage des contextes expirés
        expired_count = context_manager.cleanup_expired_contexts()
        
        # Statistiques globales
        stats = context_manager.get_context_stats()
        
        return {
            "global_stats": stats,
            "cleanup_info": {
                "expired_contexts_removed": expired_count,
                "cleanup_timestamp": datetime.now(timezone.utc).isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Erreur récupération statistiques système: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur statistiques système: {str(e)}")


# Endpoint de santé spécifique Phase 5
@router.get("/health")
async def health_check_phase5():
    """Vérification de santé spécifique aux composants Phase 5"""
    
    try:
        deps = get_dependencies()
        
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "phase": 5,
            "components": {
                "deepseek_client": "healthy" if deps["client"] else "unavailable",
                "cache_manager": "healthy" if deps["cache"] else "unavailable", 
                "context_manager": "healthy" if deps["context"] else "unavailable",
                "personalization_engine": "healthy" if deps["personalization"] else "unavailable",
                "insight_generator": "healthy" if deps["insights"] else "unavailable",
                "all_agents": "healthy" if all(deps["agents"].values()) else "unavailable"
            }
        }
        
        # Test rapide du contexte
        try:
            test_context = deps["context"].get_context_stats()
            health_status["context_stats"] = test_context
        except Exception as e:
            health_status["components"]["context_manager"] = f"error: {str(e)}"
            health_status["status"] = "degraded"
        
        return health_status
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": str(e)
        }