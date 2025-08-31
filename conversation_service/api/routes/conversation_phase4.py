"""
Endpoint Phase 4 - Version complète avec exécution search_service
"""
import logging
import time
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, Request

from conversation_service.models.requests.conversation_requests import ConversationRequest
from conversation_service.models.responses.conversation_responses import ConversationResponse, AgentMetrics
from conversation_service.models.responses.conversation_responses_phase3 import (
    ConversationResponsePhase3, QueryGenerationMetrics, ProcessingSteps,
    ConversationResponseFactory, QueryGenerationError
)
from conversation_service.models.responses.conversation_responses_phase4 import (
    ConversationResponsePhase4, ConversationResponseFactoryPhase4,
    ResilienceMetrics, SearchMetrics, SearchExecutionError
)
from conversation_service.agents.search.search_executor import (
    SearchExecutor, SearchExecutorRequest, SearchExecutorResponse
)
from conversation_service.core.search_service_client import SearchServiceConfig
from conversation_service.agents.financial import intent_classifier as intent_classifier_module
from conversation_service.agents.financial.entity_extractor import EntityExtractorAgent
from conversation_service.agents.financial.query_builder import QueryBuilderAgent
from conversation_service.models.contracts.search_service import QueryGenerationRequest
from conversation_service.clients.deepseek_client import DeepSeekClient
from conversation_service.core.cache_manager import CacheManager
from conversation_service.prompts.harena_intents import HarenaIntentType
from conversation_service.api.dependencies import (
    get_deepseek_client,
    get_cache_manager,
    get_conversation_service_status,
    validate_path_user_id,
    get_user_context,
    rate_limit_dependency
)
from conversation_service.utils.metrics_collector import metrics_collector
from conversation_service.utils.validation_utils import validate_user_message, sanitize_user_input
from config_service.config import settings

logger = logging.getLogger("conversation_service.routes.phase4")

router = APIRouter()

@router.post("/{path_user_id}", response_model=ConversationResponsePhase4)
async def analyze_conversation_phase4(
    path_user_id: int,
    request_data: ConversationRequest,
    request: Request,
    deepseek_client: DeepSeekClient = Depends(get_deepseek_client),
    cache_manager: Optional[CacheManager] = Depends(get_cache_manager),
    validated_user_id: int = Depends(validate_path_user_id),
    user_context: Dict[str, Any] = Depends(get_user_context),
    service_status: dict = Depends(get_conversation_service_status),
    _rate_limit: None = Depends(rate_limit_dependency)
) -> ConversationResponsePhase4:
    """
    Endpoint principal conversation service Phase 4 - Exécution search_service avec résilience
    
    Features Phase 4:
    - Classification intention (Phase 1)
    - Extraction entités (Phase 2) 
    - Génération requête search_service optimisée (Phase 3)
    - Exécution search_service avec résultats réels (Phase 4)
    - Circuit breaker et retry intelligent
    - Cache de résultats avec TTL
    - Fallbacks multi-niveaux
    - Authentification JWT obligatoire compatible user_service
    - Rate limiting par utilisateur avec gestion d'erreur gracieuse
    - Métriques de résilience détaillées
    """
    start_time = time.time()
    request_id = f"{validated_user_id}_{int(start_time * 1000)}_phase4"
    processing_steps = []
    
    # Logging début requête
    client_ip = getattr(request.client, 'host', 'unknown') if request.client else 'unknown'
    logger.info(
        f"[{request_id}] Phase 4 - User: {validated_user_id}, "
        f"IP: {client_ip}, Message: '{request_data.message[:30]}...'"
    )
    
    try:
        # ====================================================================
        # VALIDATION MESSAGE
        # ====================================================================
        
        message_validation = validate_user_message(request_data.message)
        if not message_validation["valid"]:
            metrics_collector.increment_counter("conversation.v4.validation_failed")
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "Message invalide",
                    "errors": message_validation["errors"],
                    "phase": 4,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            )
        
        clean_message = sanitize_user_input(request_data.message)
        
        # ====================================================================
        # ÉTAPE 1: CLASSIFICATION INTENTION
        # ====================================================================
        
        classification_start = time.time()
        try:
            intent_classifier = intent_classifier_module.IntentClassifierAgent(
                deepseek_client=deepseek_client,
                cache_manager=cache_manager
            )
            
            classification_result = await intent_classifier.classify_intent(
                user_message=clean_message,
                user_context=user_context
            )
            
            classification_time = int((time.time() - classification_start) * 1000)
            processing_steps.append(ProcessingSteps(
                agent="intent_classifier",
                duration_ms=classification_time,
                cache_hit=classification_time < 100
            ))
            
        except Exception as e:
            classification_time = int((time.time() - classification_start) * 1000)
            processing_steps.append(ProcessingSteps(
                agent="intent_classifier",
                duration_ms=classification_time,
                cache_hit=False,
                success=False,
                error_message=str(e)
            ))
            
            logger.error(f"[{request_id}] Erreur classification Phase 4: {str(e)}")
            metrics_collector.increment_counter("conversation.v4.classification_failed")
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Erreur classification intention",
                    "phase": 4,
                    "request_id": request_id,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            )
        
        if not classification_result:
            raise HTTPException(status_code=500, detail={"error": "Classification invalide", "phase": 4})
        
        intent_type_value = getattr(classification_result.intent_type, 'value', str(classification_result.intent_type))
        
        # ====================================================================
        # ÉTAPE 2: EXTRACTION ENTITÉS
        # ====================================================================
        
        entities_result = None
        if (classification_result.is_supported and 
            intent_type_value not in [HarenaIntentType.ERROR.value, HarenaIntentType.UNKNOWN.value]):
            
            extraction_start = time.time()
            try:
                entity_extractor = EntityExtractorAgent(
                    deepseek_client=deepseek_client,
                    cache_manager=cache_manager,
                    autogen_mode=False
                )
                
                entities_result = await entity_extractor.extract_entities(
                    user_message=clean_message,
                    intent_result=classification_result,
                    user_id=validated_user_id
                )
                
                extraction_time = int((time.time() - extraction_start) * 1000)
                processing_steps.append(ProcessingSteps(
                    agent="entity_extractor",
                    duration_ms=extraction_time,
                    cache_hit=extraction_time < 150
                ))
                
                logger.info(f"[{request_id}] Entités extraites: {extraction_time}ms")
                
            except Exception as e:
                extraction_time = int((time.time() - extraction_start) * 1000)
                processing_steps.append(ProcessingSteps(
                    agent="entity_extractor",
                    duration_ms=extraction_time,
                    cache_hit=False,
                    success=False,
                    error_message=str(e)
                ))
                
                logger.warning(f"[{request_id}] Extraction entités échouée: {str(e)}")
                entities_result = None
        
        # ====================================================================
        # ÉTAPE 3: GÉNÉRATION REQUÊTE SEARCH_SERVICE
        # ====================================================================
        
        search_query = None
        query_validation = None
        query_metrics = None
        
        if (classification_result.is_supported and entities_result and
            intent_type_value not in [HarenaIntentType.ERROR.value, HarenaIntentType.UNKNOWN.value]):
            
            generation_start = time.time()
            try:
                query_builder = QueryBuilderAgent(
                    deepseek_client=deepseek_client,
                    cache_manager=cache_manager,
                    autogen_mode=False
                )
                
                generation_request = QueryGenerationRequest(
                    user_id=validated_user_id,
                    intent_type=intent_type_value,
                    intent_confidence=classification_result.confidence,
                    entities=entities_result.get("entities", {}),
                    user_message=clean_message,
                    context=user_context
                )
                
                generation_response = await query_builder.generate_search_query(generation_request)
                
                generation_time = int((time.time() - generation_start) * 1000)
                processing_steps.append(ProcessingSteps(
                    agent="query_builder",
                    duration_ms=generation_time,
                    cache_hit=generation_time < 200
                ))
                
                search_query = generation_response.search_query
                query_validation = generation_response.validation
                
                query_metrics = QueryGenerationMetrics(
                    generation_time_ms=generation_time,
                    validation_time_ms=50,
                    optimization_time_ms=100,
                    generation_confidence=generation_response.generation_confidence,
                    validation_passed=generation_response.validation.schema_valid and generation_response.validation.contract_compliant,
                    optimizations_applied=len(generation_response.validation.optimization_applied),
                    estimated_performance=generation_response.validation.estimated_performance,
                    estimated_results_count=generation_response.estimated_results_count
                )
                
                logger.info(
                    f"[{request_id}] Requête générée: {generation_time}ms, "
                    f"Validation: {query_validation.schema_valid}, "
                    f"Optimisations: {len(query_validation.optimization_applied)}"
                )
                metrics_collector.increment_counter("conversation.v4.query_generation.success")
                
            except Exception as e:
                generation_time = int((time.time() - generation_start) * 1000)
                processing_steps.append(ProcessingSteps(
                    agent="query_builder",
                    duration_ms=generation_time,
                    cache_hit=False,
                    success=False,
                    error_message=str(e)
                ))
                
                logger.error(f"[{request_id}] Génération requête échouée: {str(e)}")
                metrics_collector.increment_counter("conversation.v4.query_generation.failed")
                
                search_query = None
                query_validation = None
        
        # ====================================================================
        # ÉTAPE 4: EXÉCUTION SEARCH_SERVICE AVEC RÉSILIENCE
        # ====================================================================
        
        search_results = None
        resilience_metrics = None
        search_metrics = None
        search_execution_error = None
        
        if search_query and query_validation and query_validation.schema_valid:
            
            search_start = time.time()
            try:
                # Configuration client search_service
                search_config = SearchServiceConfig(
                    base_url=getattr(settings, 'SEARCH_SERVICE_BASE_URL', 'http://localhost:8080'),
                    timeout_seconds=30.0,
                    max_retries=3,
                    circuit_breaker_enabled=True,
                    fallback_enabled=True
                )
                
                # Initialisation Search Executor
                search_executor = SearchExecutor(
                    search_service_config=search_config,
                    cache_enabled=True,
                    cache_ttl_seconds=300
                )
                
                # Requête exécution search
                executor_request = SearchExecutorRequest(
                    search_query=search_query,
                    user_id=validated_user_id,
                    request_id=request_id,
                    timeout_seconds=25.0,
                    enable_fallback=True,
                    validate_before_search=True
                )
                
                # Exécution avec SearchExecutor
                executor_response = await search_executor.handle_search_request(
                    executor_request, 
                    ctx=None  # Pas de contexte AutoGen nécessaire
                )
                
                search_execution_time = int((time.time() - search_start) * 1000)
                
                # Processing step pour search execution
                search_processing_step = ProcessingSteps(
                    agent="search_executor",
                    duration_ms=search_execution_time,
                    cache_hit=executor_response.execution_time_ms < 100,
                    success=executor_response.success
                )
                
                if executor_response.success and executor_response.search_results:
                    # Succès Phase 4 complète
                    search_results = executor_response.search_results
                    
                    # Métriques résilience
                    resilience_metrics = ResilienceMetrics(
                        circuit_breaker_triggered=executor_response.circuit_breaker_triggered,
                        retry_attempts=executor_response.retry_attempts,
                        cache_hit=executor_response.execution_time_ms < 100,
                        fallback_used=executor_response.fallback_used,
                        search_execution_time_ms=executor_response.execution_time_ms
                    )
                    
                    # Métriques search
                    search_metrics = SearchMetrics(
                        total_hits=search_results.total_hits,
                        returned_hits=len(search_results.hits),
                        has_aggregations=bool(search_results.aggregations),
                        aggregations_count=len(search_results.aggregations) if search_results.aggregations else 0,
                        search_service_took_ms=search_results.took_ms,
                        network_latency_ms=max(0, executor_response.execution_time_ms - search_results.took_ms),
                        parsing_time_ms=10,
                        results_relevance=ConversationResponseFactoryPhase4._map_performance_to_relevance(executor_response.estimated_performance)
                    )
                    
                    logger.info(
                        f"[{request_id}] ✅ Search execution réussie: {search_results.total_hits} résultats, "
                        f"Temps: {executor_response.execution_time_ms}ms, "
                        f"Fallback: {executor_response.fallback_used}"
                    )
                    metrics_collector.increment_counter("conversation.v4.search_execution.success")
                    
                else:
                    # Erreur ou pas de résultats
                    search_execution_error = SearchExecutionError(
                        error_type=executor_response.error_type or "unknown",
                        error_message=executor_response.error_message or "Unknown search error",
                        error_component="search_executor",
                        recovery_attempted=executor_response.fallback_used,
                        recovery_successful=executor_response.success and executor_response.fallback_used,
                        timestamp=datetime.now(timezone.utc)
                    )
                    
                    logger.warning(
                        f"[{request_id}] ⚠️ Search execution échouée: {executor_response.error_type}, "
                        f"Fallback: {executor_response.fallback_used}"
                    )
                    metrics_collector.increment_counter("conversation.v4.search_execution.failed")
                
                # Cleanup resources
                await search_executor.cleanup()
                
            except Exception as e:
                search_execution_time = int((time.time() - search_start) * 1000)
                search_processing_step = ProcessingSteps(
                    agent="search_executor",
                    duration_ms=search_execution_time,
                    cache_hit=False,
                    success=False,
                    error_message=str(e)
                )
                
                search_execution_error = SearchExecutionError(
                    error_type="unexpected_error",
                    error_message=f"Search execution failed: {str(e)}",
                    error_component="search_executor",
                    recovery_attempted=False,
                    recovery_successful=False,
                    timestamp=datetime.now(timezone.utc)
                )
                
                logger.error(f"[{request_id}] Erreur search execution: {str(e)}")
                metrics_collector.increment_counter("conversation.v4.search_execution.error")
            
            # Ajouter processing step search
            processing_steps.append(search_processing_step)
        
        # ====================================================================
        # CONSTRUCTION RÉPONSE FINALE PHASE 4
        # ====================================================================
        
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        # Métriques agent principales
        agent_metrics = AgentMetrics(
            agent_used="multi_agent_pipeline_phase4",
            cache_hit=any(step.cache_hit for step in processing_steps),
            model_used=getattr(settings, 'DEEPSEEK_CHAT_MODEL', 'deepseek-chat'),
            tokens_consumed=await _estimate_tokens_consumption_safe(clean_message, classification_result),
            processing_time_ms=processing_time_ms,
            confidence_threshold_met=classification_result.confidence >= 0.5
        )
        
        # Créer réponse base Phase 3
        phase3_response = ConversationResponsePhase3(
            user_id=validated_user_id,
            sub=user_context.get("user_id", validated_user_id),
            message=clean_message,
            timestamp=datetime.now(timezone.utc),
            processing_time_ms=processing_time_ms,
            intent=classification_result,
            agent_metrics=agent_metrics,
            entities=entities_result,
            search_query=search_query,
            query_validation=query_validation,
            query_generation_metrics=query_metrics,
            processing_steps=processing_steps,
            phase=3,
            request_id=request_id
        )
        
        # Création réponse Phase 4 selon résultats
        if search_results and resilience_metrics and not search_execution_error:
            # Succès Phase 4 complète avec résultats
            response = ConversationResponseFactoryPhase4.create_phase4_success(
                base_response=phase3_response,
                search_results=search_results,
                resilience_metrics=resilience_metrics,
                search_metrics=search_metrics
            )
            
            logger.info(
                f"[{request_id}] ✅ Phase 4 complète: {intent_type_value}, "
                f"Confiance: {classification_result.confidence:.2f}, "
                f"Résultats: {search_results.total_hits}, "
                f"Temps: {processing_time_ms}ms"
            )
            metrics_collector.increment_counter("conversation.v4.complete_success")
            
        elif search_execution_error:
            # Phase 4 avec erreur search mais données Phase 3 disponibles
            response = ConversationResponseFactoryPhase4.create_phase4_error(
                base_response=phase3_response,
                error=search_execution_error,
                resilience_metrics=resilience_metrics,
                partial_results=search_results
            )
            
            logger.info(
                f"[{request_id}] ⚠️ Phase 4 avec erreur search: {intent_type_value}, "
                f"Erreur: {search_execution_error.error_type}, "
                f"Temps: {processing_time_ms}ms"
            )
            metrics_collector.increment_counter("conversation.v4.search_error")
            
        elif search_query and query_validation:
            # Fallback Phase 3 (requête générée mais pas exécutée)
            response = ConversationResponsePhase4(
                user_id=phase3_response.user_id,
                sub=phase3_response.sub,
                message=phase3_response.message,
                timestamp=phase3_response.timestamp,
                request_id=phase3_response.request_id,
                intent=phase3_response.intent,
                agent_metrics=phase3_response.agent_metrics,
                processing_time_ms=phase3_response.processing_time_ms,
                status=phase3_response.status,
                warnings=["Search execution skipped - fallback to Phase 3"],
                entities=phase3_response.entities,
                search_query=phase3_response.search_query,
                query_validation=phase3_response.query_validation,
                query_generation_metrics=phase3_response.query_generation_metrics,
                processing_steps=phase3_response.processing_steps,
                agent_metrics_detailed=phase3_response.agent_metrics_detailed,
                phase=3
            )
            
            logger.info(
                f"[{request_id}] ⚠️ Fallback Phase 3 (search skipped): {intent_type_value}, "
                f"Temps: {processing_time_ms}ms"
            )
            metrics_collector.increment_counter("conversation.v4.fallback_phase3")
            
        else:
            # Fallback Phase 2 (pas de requête générée)
            response = ConversationResponsePhase4(
                user_id=phase3_response.user_id,
                sub=phase3_response.sub,
                message=phase3_response.message,
                timestamp=phase3_response.timestamp,
                request_id=phase3_response.request_id,
                intent=phase3_response.intent,
                agent_metrics=phase3_response.agent_metrics,
                processing_time_ms=phase3_response.processing_time_ms,
                status=phase3_response.status,
                warnings=["Query generation failed - fallback to Phase 2"],
                entities=phase3_response.entities,
                processing_steps=phase3_response.processing_steps,
                agent_metrics_detailed=phase3_response.agent_metrics_detailed,
                phase=2
            )
            
            logger.info(
                f"[{request_id}] ⚠️ Fallback Phase 2: {intent_type_value}, "
                f"Temps: {processing_time_ms}ms"
            )
            metrics_collector.increment_counter("conversation.v4.fallback_phase2")
        
        # Métriques détaillées
        await _collect_comprehensive_metrics_safe(request_id, classification_result, processing_time_ms, agent_metrics)
        metrics_collector.increment_counter("conversation.v4.requests.total")
        metrics_collector.record_histogram("conversation.v4.processing_time", processing_time_ms)
        
        return response
        
    except HTTPException:
        raise
        
    except Exception as e:
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        metrics_collector.increment_counter("conversation.v4.errors.technical")
        metrics_collector.record_histogram("conversation.v4.error_time", processing_time_ms)
        
        logger.error(
            f"[{request_id}] ❌ Erreur technique Phase 4: {type(e).__name__}: {str(e)[:200]}, "
            f"User: {validated_user_id}, Time: {processing_time_ms}ms",
            exc_info=True
        )
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Erreur interne du service conversation Phase 4",
                "request_id": request_id,
                "phase": 4,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )


async def _estimate_tokens_consumption_safe(
    user_message: str, 
    classification_result
) -> int:
    """Estimation tokens consommés avec gestion d'erreur robuste"""
    try:
        input_tokens = len(user_message.split()) * 1.3
        system_prompt_tokens = 200
        few_shot_examples_tokens = 150
        
        try:
            reasoning_tokens = len(classification_result.reasoning.split()) * 1.3 if classification_result.reasoning else 50
        except (AttributeError, TypeError):
            reasoning_tokens = 50
        
        json_structure_tokens = 20
        
        total_estimated = int(
            input_tokens + system_prompt_tokens + 
            few_shot_examples_tokens + reasoning_tokens + 
            json_structure_tokens
        )
        
        return max(50, min(total_estimated, 4000))
        
    except Exception as e:
        logger.debug(f"Erreur estimation tokens: {str(e)}")
        return 200


async def _collect_comprehensive_metrics_safe(
    request_id: str,
    classification_result,
    processing_time_ms: int,
    agent_metrics: AgentMetrics
) -> None:
    """Collection centralisée métriques avec gestion d'erreur robuste"""
    try:
        # Métriques de base
        metrics_collector.increment_counter("conversation.v4.requests.total")
        metrics_collector.record_histogram("conversation.v4.processing_time", processing_time_ms)
        
        # Métriques par intention
        try:
            intent_type = getattr(classification_result.intent_type, "value", classification_result.intent_type)
            if intent_type:
                safe_intent = str(intent_type).replace('.', '_').replace(' ', '_')[:50]
                metrics_collector.increment_counter(f"conversation.v4.intent.{safe_intent}")
        except Exception:
            pass
        
        # Métriques qualité
        try:
            if hasattr(classification_result, 'confidence') and classification_result.confidence is not None:
                confidence = float(classification_result.confidence)
                if 0 <= confidence <= 1:
                    metrics_collector.record_gauge("conversation.v4.intent.confidence", confidence)
        except Exception:
            pass
        
        # Métriques cache
        try:
            if hasattr(agent_metrics, 'cache_hit'):
                if agent_metrics.cache_hit:
                    metrics_collector.increment_counter("conversation.v4.cache.hits")
                else:
                    metrics_collector.increment_counter("conversation.v4.cache.misses")
        except Exception:
            pass
        
        # Métriques performance par tranche
        try:
            if processing_time_ms < 1000:  # < 1s
                metrics_collector.increment_counter("conversation.v4.performance.fast")
            elif processing_time_ms < 3000:  # < 3s
                metrics_collector.increment_counter("conversation.v4.performance.normal")
            else:
                metrics_collector.increment_counter("conversation.v4.performance.slow")
        except Exception:
            pass
        
    except Exception as e:
        logger.debug(f"[{request_id}] Erreur collection métriques: {str(e)}")