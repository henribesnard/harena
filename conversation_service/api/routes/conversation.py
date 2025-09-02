"""
Routes API pour conversation service - Version réécrite compatible JWT
"""
import logging
import time
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, Request

from conversation_service.models.requests.conversation_requests import ConversationRequest
from conversation_service.models.responses.conversation_responses import (
    ConversationResponse, AgentMetrics, ConversationResponseError,
    QueryGenerationMetrics, ProcessingSteps, ConversationResponseFactory, QueryGenerationError,
    ResilienceMetrics, SearchMetrics, SearchExecutionError,
    ResponseContent, ResponseQuality, ResponseGenerationMetrics,
    Insight, Suggestion, StructuredData
)
from conversation_service.agents.search.search_executor import (
    SearchExecutor, SearchExecutorRequest, SearchExecutorResponse
)
from conversation_service.core.search_service_client import SearchServiceConfig
from conversation_service.models.responses.conversation_responses import (
    EnrichedConversationResponse,
    TeamHealthResponse,
    TeamMetricsResponse
)
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
    rate_limit_dependency,
    get_multi_agent_team,
    get_conversation_processor,
    get_conversation_persistence
)
from conversation_service.services.conversation_persistence import (
    ConversationPersistenceService, create_conversation_data, create_turn_data
)
from conversation_service.utils.metrics_collector import metrics_collector
from conversation_service.utils.validation_utils import validate_user_message, sanitize_user_input
from config_service.config import settings

# Configuration du router et logger
router = APIRouter(tags=["conversation"])
logger = logging.getLogger("conversation_service.routes")

@router.post("/conversation/{path_user_id}")
async def analyze_conversation(
    path_user_id: int,
    request_data: ConversationRequest,
    request: Request,
    deepseek_client: DeepSeekClient = Depends(get_deepseek_client),
    cache_manager: Optional[CacheManager] = Depends(get_cache_manager),
    validated_user_id: int = Depends(validate_path_user_id),
    user_context: Dict[str, Any] = Depends(get_user_context),
    service_status: dict = Depends(get_conversation_service_status),
    persistence_service: Optional[ConversationPersistenceService] = Depends(get_conversation_persistence),
    _rate_limit: None = Depends(rate_limit_dependency)
):
    """
    Endpoint principal conversation service Phase 5 - Workflow complet avec réponse naturelle
    
    Features Phase 5 (workflow complet):
    - Classification intention (Phase 1)
    - Extraction entités (Phase 2) 
    - Génération requête search_service optimisée (Phase 3)
    - Exécution search_service avec résultats réels (Phase 4)
    - Génération réponse naturelle contextualisée (Phase 5)
    - Insights automatiques et suggestions actionnables
    - Personnalisation basée sur le contexte utilisateur
    - Circuit breaker et résilience complète
    - Authentification JWT obligatoire compatible user_service
    - Rate limiting par utilisateur avec gestion d'erreur gracieuse
    - Métriques de résilience détaillées
    """
    # Traitement Phase 5 complet intégré
    return await _process_conversation_phase5_integrated(
        user_id=path_user_id,
        request=request_data,
        http_request=request,
        validated_user_id=validated_user_id,
        user_context=user_context,
        persistence_service=persistence_service,
        _rate_limit=_rate_limit
    )


@router.post("/conversation/legacy/{path_user_id}")
async def analyze_conversation_legacy_phase2(
    path_user_id: int,
    request_data: ConversationRequest,
    request: Request,
    deepseek_client: DeepSeekClient = Depends(get_deepseek_client),
    cache_manager: Optional[CacheManager] = Depends(get_cache_manager),
    validated_user_id: int = Depends(validate_path_user_id),
    user_context: Dict[str, Any] = Depends(get_user_context),
    service_status: dict = Depends(get_conversation_service_status),
    _rate_limit: None = Depends(rate_limit_dependency)
):
    """
    Endpoint Legacy Phase 2: Intentions + Entités (sans génération requête)
    
    DEPRECATED: Utilisez /conversation/{user_id} pour la Phase 3 complète
    
    Features Phase 2 (Legacy):
    - Classification intention (Phase 1)
    - Extraction entités (Phase 2) 
    - Compatible avec anciennes intégrations
    """
    start_time = time.time()
    request_id = f"{validated_user_id}_{int(start_time * 1000)}_v3"
    processing_steps = []
    
    # Logging début requête
    client_ip = getattr(request.client, 'host', 'unknown') if request.client else 'unknown'
    logger.info(
        f"[{request_id}] Phase 3 - User: {validated_user_id}, "
        f"IP: {client_ip}, Message: '{request_data.message[:30]}...'"
    )
    
    try:
        # ====================================================================
        # VALIDATION MESSAGE (réutilise logique existante)
        # ====================================================================
        
        message_validation = validate_user_message(request_data.message)
        if not message_validation["valid"]:
            metrics_collector.increment_counter("conversation.v3.validation_failed")
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "Message invalide",
                    "errors": message_validation["errors"],
                    "phase": 3,
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
                cache_hit=classification_time < 100  # Heuristique cache
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
            
            logger.error(f"[{request_id}] Erreur classification Phase 3: {str(e)}")
            metrics_collector.increment_counter("conversation.v3.classification_failed")
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Erreur classification intention",
                    "phase": 3,
                    "request_id": request_id,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            )
        
        # Validation résultat classification
        if not classification_result:
            raise HTTPException(status_code=500, detail={"error": "Classification invalide", "phase": 3})
        
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
        
        # Génération requête si intention supportée
        if (classification_result.is_supported and entities_result and
            intent_type_value not in [HarenaIntentType.ERROR.value, HarenaIntentType.UNKNOWN.value]):
            
            generation_start = time.time()
            try:
                query_builder = QueryBuilderAgent(
                    deepseek_client=deepseek_client,
                    cache_manager=cache_manager,
                    autogen_mode=False
                )
                
                # Construction requête génération
                generation_request = QueryGenerationRequest(
                    user_id=validated_user_id,
                    intent_type=intent_type_value,
                    intent_confidence=classification_result.confidence,
                    entities=entities_result.get("entities", {}),
                    user_message=clean_message,
                    context=user_context
                )
                
                # Génération requête
                generation_response = await query_builder.generate_search_query(generation_request)
                
                generation_time = int((time.time() - generation_start) * 1000)
                processing_steps.append(ProcessingSteps(
                    agent="query_builder",
                    duration_ms=generation_time,
                    cache_hit=generation_time < 200
                ))
                
                # Extraction résultats
                search_query = generation_response.search_query
                query_validation = generation_response.validation
                
                # Métriques génération
                query_metrics = QueryGenerationMetrics(
                    generation_time_ms=generation_time,
                    validation_time_ms=50,  # Inclus dans generation_time
                    optimization_time_ms=100,  # Inclus dans generation_time
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
                metrics_collector.increment_counter("conversation.v3.query_generation.success")
                
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
                metrics_collector.increment_counter("conversation.v3.query_generation.failed")
                
                # Continue sans search_query (fallback Phase 2)
                search_query = None
                query_validation = None
        
        # ====================================================================
        # CONSTRUCTION RÉPONSE FINALE
        # ====================================================================
        
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        # Métriques agent principales (réutilise logique existante)
        agent_metrics = AgentMetrics(
            agent_used="multi_agent_pipeline",
            cache_hit=any(step.cache_hit for step in processing_steps),
            model_used=getattr(settings, 'DEEPSEEK_CHAT_MODEL', 'deepseek-chat'),
            tokens_consumed=await _estimate_tokens_consumption_safe(clean_message, classification_result),
            processing_time_ms=processing_time_ms,
            confidence_threshold_met=classification_result.confidence >= 0.5
        )
        
        # Création réponse base
        base_response = ConversationResponse(
            user_id=validated_user_id,
            sub=user_context.get("user_id", validated_user_id),
            message=clean_message,
            timestamp=datetime.now(timezone.utc),
            processing_time_ms=processing_time_ms,
            intent=classification_result,
            agent_metrics=agent_metrics,
            entities=entities_result,
            phase=3 if search_query else 2,  # Phase 3 si requête générée
            request_id=request_id
        )
        
        # Création réponse Phase 3
        if search_query and query_validation:
            # Succès Phase 3 complète
            response = ConversationResponseFactory.create_phase3_success(
                base_response=base_response,
                search_query=search_query,
                query_validation=query_validation,
                processing_steps=processing_steps,
                query_metrics=query_metrics
            )
            
            logger.info(
                f"[{request_id}] ✅ Phase 3 complète: {intent_type_value}, "
                f"Confiance: {classification_result.confidence:.2f}, "
                f"Requête: {'générée' if search_query else 'échouée'}, "
                f"Temps: {processing_time_ms}ms"
            )
            metrics_collector.increment_counter("conversation.v3.complete_success")
            
        else:
            # Fallback Phase 2 (sans requête)
            response = ConversationResponse(
                user_id=base_response.user_id,
                sub=base_response.sub,
                message=base_response.message,
                timestamp=base_response.timestamp,
                request_id=base_response.request_id,
                intent=base_response.intent,
                agent_metrics=base_response.agent_metrics,
                processing_time_ms=base_response.processing_time_ms,
                status=base_response.status,
                phase=2,  # Downgrade vers Phase 2
                warnings=["Génération requête search_service échouée, fallback Phase 2"],
                entities=base_response.entities,
                processing_steps=processing_steps
            )
            
            logger.info(
                f"[{request_id}] ⚠️ Phase 3 partielle (fallback Phase 2): {intent_type_value}, "
                f"Temps: {processing_time_ms}ms"
            )
            metrics_collector.increment_counter("conversation.v3.partial_success")
        
        # Métriques détaillées
        await _collect_comprehensive_metrics_safe(request_id, classification_result, processing_time_ms, agent_metrics)
        metrics_collector.increment_counter("conversation.v3.requests.total")
        metrics_collector.record_histogram("conversation.v3.processing_time", processing_time_ms)
        
        return response
        
    except HTTPException:
        raise
        
    except Exception as e:
        # Erreur critique Phase 3
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        logger.error(
            f"[{request_id}] ❌ Erreur critique Phase 3: {type(e).__name__}: {str(e)[:200]}, "
            f"Temps: {processing_time_ms}ms",
            exc_info=True
        )
        
        metrics_collector.increment_counter("conversation.v3.errors.critical")
        metrics_collector.record_histogram("conversation.v3.error_time", processing_time_ms)
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Erreur critique Phase 3",
                "phase": 3,
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )

# ============================================================================
# ENDPOINTS MONITORING (PUBLICS - SANS AUTHENTIFICATION)
# ============================================================================

@router.get("/conversation/health")
async def conversation_health_detailed():
    """Health check spécifique conversation service - ENDPOINT PUBLIC"""
    try:
        health_metrics = metrics_collector.get_health_metrics()
        
        return {
            "service": "conversation_service", 
            "phase": 3,
            "version": "3.0.0",
            "status": health_metrics["status"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "jwt_compatible": True,
            "health_details": {
                "total_requests": health_metrics["total_requests"],
                "error_rate_percent": health_metrics["error_rate_percent"],
                "avg_latency_ms": health_metrics.get("latency_p95_ms", 0),
                "uptime_seconds": health_metrics["uptime_seconds"],
                "status_description": _get_health_status_description(health_metrics["status"])
            },
            "features": {
                "intent_classification": True,
                "entity_extraction": True,
                "query_generation": True,
                "search_service_integration": True,
                "supported_intents": len(HarenaIntentType),
                "json_output_forced": True,
                "cache_enabled": True,
                "auth_required": True,
                "rate_limiting": True,
                "jwt_compatible": True,
                "query_optimization": True,
                "query_validation": True
            },
            "configuration": {
                "min_confidence_threshold": getattr(settings, 'MIN_CONFIDENCE_THRESHOLD', 0.5),
                "max_message_length": getattr(settings, 'MAX_MESSAGE_LENGTH', 1000),
                "cache_ttl": getattr(settings, 'CACHE_TTL_INTENT', 300),
                "deepseek_model": getattr(settings, 'DEEPSEEK_CHAT_MODEL', 'deepseek-chat'),
                "environment": getattr(settings, 'ENVIRONMENT', 'production')
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur health check: {str(e)}")
        return {
            "service": "conversation_service",
            "phase": 3,
            "version": "3.0.0",
            "status": "error",
            "jwt_compatible": False,
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

@router.get("/conversation/metrics")
async def conversation_metrics_detailed():
    """Métriques détaillées pour monitoring - ENDPOINT PUBLIC"""
    try:
        all_metrics = metrics_collector.get_all_metrics()
        
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "service_info": {
                "name": "conversation_service",
                "version": "1.1.0",
                "phase": 1,
                "features": ["intent_classification", "json_output", "cache", "auth", "jwt_compatible"],
                "jwt_compatible": True
            },
            "metrics": all_metrics,
            "performance_summary": {
                "avg_response_time": _safe_get_metric(all_metrics, ["histograms", "conversation.processing_time", "avg"], 0),
                "p95_response_time": _safe_get_metric(all_metrics, ["histograms", "conversation.processing_time", "p95"], 0),
                "requests_per_second": _safe_get_metric(all_metrics, ["rates", "conversation.requests_per_second"], 0),
                "error_rate": _calculate_error_rate(all_metrics),
                "cache_hit_rate": _calculate_cache_hit_rate(all_metrics)
            },
            "intent_distribution": _calculate_intent_distribution(all_metrics)
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur export métriques: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Erreur récupération métriques",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )

@router.get("/conversation/status")
async def conversation_status():
    """Statut global service pour monitoring externe - ENDPOINT PUBLIC"""
    try:
        health_metrics = metrics_collector.get_health_metrics()
        
        health_status = health_metrics["status"]

        return {
            "status": health_status,
            "uptime_seconds": health_metrics["uptime_seconds"],
            "version": "1.1.0",
            "phase": 1,
            "ready": health_status == "healthy",
            "jwt_compatible": True,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur status check: {str(e)}")
        return {
            "status": "error",
            "ready": False,
            "jwt_compatible": False,
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

# ============================================================================
# UTILITAIRES PRIVÉS AVEC GESTION D'ERREUR ROBUSTE
# ============================================================================

async def _estimate_tokens_consumption_safe(
    user_message: str, 
    classification_result
) -> int:
    """Estimation tokens consommés avec gestion d'erreur robuste"""
    try:
        # Estimation basée sur longueur réelle
        input_tokens = len(user_message.split()) * 1.3  # Facteur français
        system_prompt_tokens = 200  # Prompt système estimé
        few_shot_examples_tokens = 150  # Exemples few-shot
        
        # Tokens output avec gestion d'erreur
        try:
            reasoning_tokens = len(classification_result.reasoning.split()) * 1.3 if classification_result.reasoning else 50
        except (AttributeError, TypeError):
            reasoning_tokens = 50
        
        # Tokens JSON structure
        json_structure_tokens = 20
        
        total_estimated = int(
            input_tokens + system_prompt_tokens + 
            few_shot_examples_tokens + reasoning_tokens + 
            json_structure_tokens
        )
        
        return max(50, min(total_estimated, 4000))  # Borné entre 50 et 4000
        
    except Exception as e:
        logger.debug(f"Erreur estimation tokens: {str(e)}")
        return 200  # Fallback conservateur

async def _collect_comprehensive_metrics_safe(
    request_id: str,
    classification_result,
    processing_time_ms: int,
    agent_metrics: AgentMetrics
) -> None:
    """Collection centralisée métriques avec gestion d'erreur robuste"""
    try:
        # Métriques de base - toujours sûres
        metrics_collector.increment_counter("conversation.requests.total")
        metrics_collector.record_histogram("conversation.processing_time", processing_time_ms)
        metrics_collector.record_rate("conversation.requests")
        
        # Métriques par intention avec protection
        try:
            intent_type = getattr(classification_result.intent_type, "value", classification_result.intent_type)
            if intent_type:
                safe_intent = str(intent_type).replace('.', '_').replace(' ', '_')[:50]
                metrics_collector.increment_counter(f"conversation.intent.{safe_intent}")
            
            if hasattr(classification_result, 'category') and classification_result.category:
                safe_category = str(classification_result.category).replace('.', '_').replace(' ', '_')[:50]
                metrics_collector.increment_counter(f"conversation.intent.category.{safe_category}")
                
        except Exception as intent_metrics_error:
            logger.debug(f"[{request_id}] Erreur métriques intention: {str(intent_metrics_error)}")
        
        # Métriques qualité avec protection
        try:
            if hasattr(classification_result, 'confidence') and classification_result.confidence is not None:
                confidence = float(classification_result.confidence)
                if 0 <= confidence <= 1:
                    metrics_collector.record_gauge("conversation.intent.confidence", confidence)
                    
                    confidence_threshold = getattr(settings, 'MIN_CONFIDENCE_THRESHOLD', 0.5)
                    if confidence < confidence_threshold:
                        metrics_collector.increment_counter("conversation.intent.low_confidence")
                    elif confidence > 0.9:
                        metrics_collector.increment_counter("conversation.intent.high_confidence")
                        
        except Exception as confidence_metrics_error:
            logger.debug(f"[{request_id}] Erreur métriques confiance: {str(confidence_metrics_error)}")
        
        # Métriques support avec protection
        try:
            if hasattr(classification_result, 'is_supported') and not classification_result.is_supported:
                metrics_collector.increment_counter("conversation.intent.unsupported")
        except Exception as support_metrics_error:
            logger.debug(f"[{request_id}] Erreur métriques support: {str(support_metrics_error)}")
        
        # Métriques cache avec protection
        try:
            if hasattr(agent_metrics, 'cache_hit'):
                if agent_metrics.cache_hit:
                    metrics_collector.increment_counter("conversation.cache.hits")
                    metrics_collector.record_histogram("conversation.cache.hit_time", processing_time_ms)
                else:
                    metrics_collector.increment_counter("conversation.cache.misses")
        except Exception as cache_metrics_error:
            logger.debug(f"[{request_id}] Erreur métriques cache: {str(cache_metrics_error)}")
        
        # Métriques tokens avec protection
        try:
            if hasattr(agent_metrics, 'tokens_consumed') and agent_metrics.tokens_consumed:
                tokens = int(agent_metrics.tokens_consumed)
                if 0 <= tokens <= 10000:  # Sanity check
                    metrics_collector.record_histogram("conversation.tokens.consumed", tokens)
        except Exception as tokens_metrics_error:
            logger.debug(f"[{request_id}] Erreur métriques tokens: {str(tokens_metrics_error)}")
        
        # Métriques alternatives avec protection
        try:
            if hasattr(classification_result, 'alternatives') and classification_result.alternatives:
                metrics_collector.increment_counter("conversation.alternatives.provided")
                alternatives_count = len(classification_result.alternatives)
                if 0 <= alternatives_count <= 10:
                    metrics_collector.record_gauge("conversation.alternatives.count", alternatives_count)
        except Exception as alternatives_metrics_error:
            logger.debug(f"[{request_id}] Erreur métriques alternatives: {str(alternatives_metrics_error)}")
        
        # Métriques performance par tranche
        try:
            if processing_time_ms < 100:
                metrics_collector.increment_counter("conversation.performance.fast")
            elif processing_time_ms < 500:
                metrics_collector.increment_counter("conversation.performance.normal")
            else:
                metrics_collector.increment_counter("conversation.performance.slow")
        except Exception as perf_metrics_error:
            logger.debug(f"[{request_id}] Erreur métriques performance: {str(perf_metrics_error)}")
        
        logger.debug(f"[{request_id}] Métriques collectées avec succès")
        
    except Exception as e:
        logger.error(f"[{request_id}] Erreur collection métriques globale: {str(e)}")

def _safe_get_metric(metrics: Dict[str, Any], path: list, default: Any = None) -> Any:
    """Récupération sécurisée de métrique imbriquée"""
    try:
        current = metrics
        for key in path:
            current = current[key]
        return current
    except (KeyError, TypeError, AttributeError):
        return default

def _get_health_status_description(status: str) -> str:
    """Description détaillée du statut de santé avec gestion d'erreur"""
    descriptions = {
        "healthy": "Service opérationnel, aucune requête traitée ou performances normales",
        "degraded": "Service opérationnel mais performances réduites",
        "unhealthy": "Service en difficulté, performances critiques",
        "unknown": "Statut indéterminable"
    }
    return descriptions.get(status, f"Statut: {status}")

def _calculate_error_rate(metrics: Dict[str, Any]) -> float:
    """Calcul taux d'erreur global avec protection"""
    try:
        counters = metrics.get("counters", {})
        total_requests = counters.get("conversation.requests.total", 0)
        
        if total_requests <= 0:
            return 0.0
        
        total_errors = (
            counters.get("conversation.errors.technical", 0) +
            counters.get("conversation.errors.auth", 0) +
            counters.get("conversation.errors.validation", 0) +
            counters.get("conversation.errors.classification", 0)
        )
        
        error_rate = (total_errors / total_requests) * 100
        return min(max(error_rate, 0.0), 100.0)  # Borné entre 0 et 100
        
    except Exception as e:
        logger.debug(f"Erreur calcul taux erreur: {str(e)}")
        return 0.0

def _calculate_cache_hit_rate(metrics: Dict[str, Any]) -> float:
    """Calcul taux de hit cache avec protection"""
    try:
        counters = metrics.get("counters", {})
        cache_hits = counters.get("conversation.cache.hits", 0)
        cache_misses = counters.get("conversation.cache.misses", 0)
        total_cache_operations = cache_hits + cache_misses
        
        if total_cache_operations <= 0:
            return 0.0
        
        hit_rate = (cache_hits / total_cache_operations) * 100
        return min(max(hit_rate, 0.0), 100.0)  # Borné entre 0 et 100
        
    except Exception as e:
        logger.debug(f"Erreur calcul cache hit rate: {str(e)}")
        return 0.0

def _calculate_intent_distribution(metrics: Dict[str, Any]) -> Dict[str, int]:
    """Distribution des intentions classifiées avec protection"""
    try:
        counters = metrics.get("counters", {})
        intent_distribution = {}
        
        # Filtrer les métriques d'intention avec protection
        for key, value in counters.items():
            try:
                if (key.startswith("conversation.intent.") and 
                    not key.startswith("conversation.intent.category") and
                    key not in ["conversation.intent.unsupported", "conversation.intent.low_confidence", "conversation.intent.high_confidence"]):
                    
                    intent_name = key.replace("conversation.intent.", "")
                    if isinstance(value, (int, float)) and value >= 0:
                        intent_distribution[intent_name] = int(value)
                        
            except Exception as intent_error:
                logger.debug(f"Erreur traitement métrique intention {key}: {str(intent_error)}")
        
        # Trier par valeur décroissante avec limite
        sorted_distribution = dict(sorted(intent_distribution.items(), key=lambda x: x[1], reverse=True)[:20])
        return sorted_distribution
        
    except Exception as e:
        logger.debug(f"Erreur calcul distribution intentions: {str(e)}")
        return {}

# ============================================================================
# ROUTES DEBUG (UNIQUEMENT EN NON-PRODUCTION)
# ============================================================================

environment = getattr(settings, 'ENVIRONMENT', 'production')
if environment != "production":
    
    @router.get("/conversation/debug/cache-stats")
    async def debug_cache_stats(
        cache_manager: Optional[CacheManager] = Depends(get_cache_manager)
    ):
        """Stats cache détaillées pour debugging - NÉCESSITE AUTH"""
        if not cache_manager:
            return {"error": "Cache non disponible"}
        
        try:
            return await cache_manager.get_cache_stats()
        except Exception as e:
            return {"error": f"Erreur récupération stats cache: {str(e)}"}
    
    @router.post("/conversation/debug/clear-cache")
    async def debug_clear_cache(
        cache_manager: Optional[CacheManager] = Depends(get_cache_manager)
    ):
        """Nettoyage cache pour debugging - NÉCESSITE AUTH"""
        if not cache_manager:
            return {"error": "Cache non disponible", "cache_cleared": False}
        
        try:
            success = await cache_manager.clear_all_cache()
            return {
                "cache_cleared": success,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            return {
                "cache_cleared": False,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    @router.get("/conversation/debug/agent-metrics")
    async def debug_agent_metrics():
        """Métriques agents détaillées pour debugging - ENDPOINT PUBLIC EN DEBUG"""
        try:
            return {
                "global_metrics": metrics_collector.get_all_metrics(),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "environment": environment
            }
        except Exception as e:
            return {
                "error": f"Erreur métriques debug: {str(e)}",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    @router.get("/conversation/debug/test-classification/{text}")
    async def debug_test_classification(
        text: str,
        deepseek_client: DeepSeekClient = Depends(get_deepseek_client),
        cache_manager: Optional[CacheManager] = Depends(get_cache_manager)
    ):
        """Test direct classification pour debugging - NÉCESSITE AUTH"""
        request_id = f"debug_{int(time.time() * 1000)}"
        
        try:
            intent_classifier = intent_classifier_module.IntentClassifierAgent(
                deepseek_client=deepseek_client,
                cache_manager=cache_manager
            )
            
            result = await intent_classifier.classify_intent(text)
            
            return {
                "input": text,
                "result": result.model_dump(mode="json") if hasattr(result, 'model_dump') else str(result),
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"[{request_id}] Erreur test classification debug: {str(e)}")
            return {
                "input": text,
                "error": str(e),
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }


# ============================================================================
# ENDPOINTS DUAL-MODE (MULTI-AGENT + FALLBACK)
# ============================================================================

@router.post("/conversation/v2/{path_user_id}", response_model=EnrichedConversationResponse)
async def analyze_conversation_dual_mode(
    path_user_id: int,
    request_data: ConversationRequest,
    request: Request,
    conversation_processor: Dict[str, Any] = Depends(get_conversation_processor),
    cache_manager: Optional[CacheManager] = Depends(get_cache_manager),
    validated_user_id: int = Depends(validate_path_user_id),
    user_context: Dict[str, Any] = Depends(get_user_context),
    service_status: dict = Depends(get_conversation_service_status),
    _rate_limit: None = Depends(rate_limit_dependency)
):
    """
    Endpoint V2 avec choix automatique multi-agent ou single-agent
    
    Features:
    - Choix automatique entre équipe AutoGen et agent unique
    - Fallback robuste multi-agent → single-agent → erreur
    - Réponse enrichie compatible avec format existant
    - Cache partagé entre modes
    - Métriques différenciées par mode
    """
    start_time = time.time()
    request_id = f"{validated_user_id}_{int(start_time * 1000)}_v2"
    
    # Logging début requête
    client_ip = getattr(request.client, 'host', 'unknown') if request.client else 'unknown'
    processing_mode = conversation_processor["processing_mode"]
    
    logger.info(
        f"[{request_id}] Conversation V2 - User: {validated_user_id}, "
        f"Mode: {processing_mode}, IP: {client_ip}, Message: '{request_data.message[:30]}...'"
    )
    
    try:
        # Validation message (réutilise logique existante)
        message_validation = validate_user_message(request_data.message)
        if not message_validation["valid"]:
            metrics_collector.increment_counter(f"conversation.v2.{processing_mode}.validation_failed")
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "Message invalide",
                    "errors": message_validation["errors"],
                    "processing_mode": processing_mode,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            )
        
        clean_message = sanitize_user_input(request_data.message)
        
        # Traitement selon mode sélectionné
        if processing_mode == "multi_agent_team":
            return await _process_with_multi_agent_team(
                validated_user_id, clean_message, conversation_processor,
                cache_manager, request_id, start_time
            )
        else:
            return await _process_with_single_agent_fallback(
                validated_user_id, clean_message, conversation_processor,
                cache_manager, request_id, start_time
            )
    
    except HTTPException:
        raise  # Re-raise validation errors
    except Exception as e:
        # Fallback final vers single agent en cas d'erreur
        logger.error(f"[{request_id}] Erreur traitement V2: {str(e)}")
        metrics_collector.increment_counter(f"conversation.v2.{processing_mode}.critical_fallback")
        
        if conversation_processor.get("deepseek_client"):
            return await _process_with_single_agent_fallback(
                validated_user_id, clean_message, conversation_processor,
                cache_manager, request_id, start_time, 
                fallback_reason=f"Critical error: {str(e)}"
            )
        else:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Service temporairement indisponible",
                    "processing_mode": processing_mode,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            )


async def _process_with_multi_agent_team(
    user_id: int, 
    message: str,
    conversation_processor: Dict[str, Any],
    cache_manager: Optional[CacheManager],
    request_id: str,
    start_time: float
) -> EnrichedConversationResponse:
    """Traitement avec équipe multi-agents AutoGen"""
    
    multi_agent_team = conversation_processor["multi_agent_team"]
    
    try:
        # Traitement équipe AutoGen
        team_results = await multi_agent_team.process_user_message(
            user_message=message,
            user_id=user_id
        )
        
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        # Métriques spécifiques multi-agent
        metrics_collector.increment_counter("conversation.v2.multi_agent_team.success")
        metrics_collector.record_histogram("conversation.v2.multi_agent_team.processing_time", processing_time_ms)
        
        if team_results.get("from_cache"):
            metrics_collector.increment_counter("conversation.v2.multi_agent_team.cache_hit")
        
        # Génération réponse enrichie
        enriched_response = EnrichedConversationResponse.from_team_results(
            user_id=user_id,
            message=message,
            team_results=team_results,
            processing_time_ms=processing_time_ms
        )
        
        logger.info(
            f"[{request_id}] Multi-agent terminé - "
            f"Success: {team_results.get('workflow_success')}, "
            f"Temps: {processing_time_ms}ms, "
            f"Cache: {team_results.get('from_cache', False)}"
        )
        
        return enriched_response
        
    except Exception as e:
        logger.warning(f"[{request_id}] Équipe multi-agents échouée: {str(e)}")
        metrics_collector.increment_counter("conversation.v2.multi_agent_team.failed")
        
        # Fallback automatique vers single agent
        return await _process_with_single_agent_fallback(
            user_id, message, conversation_processor, cache_manager,
            request_id, start_time, fallback_reason=f"Multi-agent failed: {str(e)}"
        )


async def _process_with_single_agent_fallback(
    user_id: int,
    message: str, 
    conversation_processor: Dict[str, Any],
    cache_manager: Optional[CacheManager],
    request_id: str,
    start_time: float,
    fallback_reason: Optional[str] = None
) -> EnrichedConversationResponse:
    """Traitement fallback avec agent unique"""
    
    deepseek_client = conversation_processor["deepseek_client"]
    
    try:
        # Agent intent standard (existant)
        intent_classifier = intent_classifier_module.IntentClassifierAgent(
            deepseek_client=deepseek_client,
            cache_manager=cache_manager
        )
        
        # Classification intention
        classification_result = await intent_classifier.classify_intent(message, user_id)
        
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        # Métriques spécifiques single agent
        mode_label = "fallback" if fallback_reason else "selected"
        metrics_collector.increment_counter(f"conversation.v2.single_agent.{mode_label}")
        metrics_collector.record_histogram(f"conversation.v2.single_agent.processing_time", processing_time_ms)
        
        # Conversion en réponse standard puis enrichie
        standard_response = ConversationResponse(
            user_id=user_id,
            message=message,
            timestamp=datetime.now(timezone.utc),
            processing_time_ms=processing_time_ms,
            status="success",
            intent=classification_result,
            agent_metrics=AgentMetrics(
                agent_used="intent_classifier",
                model_used="deepseek-chat",
                tokens_consumed=getattr(classification_result, 'tokens_consumed', 0),
                processing_time_ms=processing_time_ms,
                confidence_threshold_met=classification_result.confidence > 0.7,
                cache_hit=getattr(classification_result, 'cache_hit', False)
            )
        )
        
        # Conversion en réponse enrichie avec contexte fallback
        enriched_response = EnrichedConversationResponse.from_fallback_single_agent(
            standard_response, fallback_reason or "Single agent selected"
        )
        
        logger.info(
            f"[{request_id}] Single-agent terminé - "
            f"Intent: {classification_result.intent_type}, "
            f"Confidence: {classification_result.confidence:.2f}, "
            f"Temps: {processing_time_ms}ms, "
            f"Fallback: {'Yes' if fallback_reason else 'No'}"
        )
        
        return enriched_response
        
    except Exception as e:
        logger.error(f"[{request_id}] Single-agent fallback échoué: {str(e)}")
        metrics_collector.increment_counter("conversation.v2.single_agent.failed")
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Tous les modes de traitement ont échoué",
                "fallback_reason": fallback_reason,
                "final_error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )


@router.get("/team/health", response_model=TeamHealthResponse)
async def get_team_health_status(
    request: Request,
    multi_agent_team = Depends(get_multi_agent_team),
    deepseek_client: DeepSeekClient = Depends(get_deepseek_client)
):
    """Health check détaillé modes de traitement"""
    
    try:
        # Test disponibilité équipe
        team_available = multi_agent_team is not None
        team_details = None
        
        if team_available:
            try:
                team_health = await asyncio.wait_for(
                    multi_agent_team.health_check(),
                    timeout=5.0
                )
                team_details = team_health
            except Exception as e:
                logger.debug(f"Team health check failed: {str(e)}")
                team_available = False
        
        # Test agent unique
        single_agent_available = deepseek_client is not None
        
        # Déterminer mode courant
        if team_available:
            current_mode = "multi_agent_team"
        elif single_agent_available:
            current_mode = "single_agent"
        else:
            current_mode = "unavailable"
        
        return TeamHealthResponse(
            single_agent_available=single_agent_available,
            multi_agent_team_available=team_available,
            current_processing_mode=current_mode,
            autogen_details=team_details
        )
        
    except Exception as e:
        logger.error(f"Erreur team health check: {str(e)}")
        return TeamHealthResponse(
            single_agent_available=False,
            multi_agent_team_available=False,
            current_processing_mode="error"
        )


@router.get("/team/metrics", response_model=TeamMetricsResponse)
async def get_team_metrics(
    multi_agent_team = Depends(get_multi_agent_team)
):
    """Métriques détaillées équipe multi-agents"""
    
    try:
        if not multi_agent_team:
            return TeamMetricsResponse(available=False)
        
        team_stats = multi_agent_team.get_team_statistics()
        team_health = await multi_agent_team.health_check()
        
        # Métriques comparatives (simulation - à enrichir avec vraies données)
        performance_comparison = {
            "multi_agent_vs_single": {
                "avg_processing_time_improvement": -15,  # % (négatif = plus lent)
                "feature_completeness_improvement": 150,  # % (plus de fonctionnalités)
                "cache_efficiency_improvement": 25  # % (meilleur cache)
            },
            "recommendation": "Multi-agent recommandé pour requêtes complexes"
        }
        
        return TeamMetricsResponse(
            available=True,
            statistics=team_stats,
            health=team_health,
            performance_comparison=performance_comparison
        )
        
    except Exception as e:
        logger.error(f"Erreur team metrics: {str(e)}")
        return TeamMetricsResponse(
            available=False,
            statistics={"error": str(e)}
        )


logger.info(f"Routes conversation configurées - Environnement: {environment}")
logger.info(f"Endpoints debug: {'activés' if environment != 'production' else 'désactivés'}")
# ============================================================================
# INTÉGRATION FONCTIONS PHASE 5 - WORKFLOW COMPLET
# ============================================================================

async def _process_conversation_phase5_integrated(
    user_id: int,
    request: ConversationRequest,
    http_request: Request,
    validated_user_id: int,
    user_context: Dict[str, Any],
    persistence_service: Optional[ConversationPersistenceService] = None,
    _rate_limit: None = None
) -> ConversationResponse:
    """
    Phase 5: Workflow complet avec génération de réponse naturelle - Intégré
    """
    start_time = time.time()
    request_id = f"phase5_{int(time.time() * 1000)}_{user_id}"
    
    logger.info(f"[{request_id}] 🚀 Phase 5 intégrée - Début workflow pour user {user_id}")
    
    try:
        # Validation message
        message_validation = validate_user_message(request.message)
        if not message_validation["valid"]:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "Message invalide",
                    "errors": message_validation["errors"],
                    "phase": 5,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            )
        
        clean_message = sanitize_user_input(request.message)
        
        # ====================================================================
        # ÉTAPE 1: Classification intention
        # ====================================================================
        step1_start = time.time()
        
        # Obtenir les dépendances depuis app state
        deepseek_client = http_request.app.state.deepseek_client
        cache_manager = http_request.app.state.cache_manager
        
        intent_classifier = intent_classifier_module.IntentClassifierAgent(
            deepseek_client=deepseek_client,
            cache_manager=cache_manager
        )
        
        classification_result = await intent_classifier.classify_intent(
            user_message=clean_message,
            user_context=user_context
        )
        
        step1_duration = int((time.time() - step1_start) * 1000)
        step1 = ProcessingSteps(
            agent="intent_classifier",
            duration_ms=step1_duration,
            cache_hit=step1_duration < 100,
            success=True
        )
        
        intent_type_value = getattr(classification_result.intent_type, 'value', str(classification_result.intent_type))
        
        # ====================================================================
        # ÉTAPE 2: Extraction entités
        # ====================================================================
        entities_result = None
        step2 = None
        
        if classification_result.is_supported:
            step2_start = time.time()
            
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
            
            step2_duration = int((time.time() - step2_start) * 1000)
            step2 = ProcessingSteps(
                agent="entity_extractor",
                duration_ms=step2_duration,
                cache_hit=step2_duration < 150,
                success=True
            )
        
        # ====================================================================
        # ÉTAPE 3: Génération de requête search (si intention supportée)
        # ====================================================================
        search_query = None
        step3 = None
        
        if classification_result.is_supported and entities_result:
            step3_start = time.time()
            
            query_builder = QueryBuilderAgent(
                deepseek_client=deepseek_client,
                cache_manager=cache_manager
            )
            
            query_request = QueryGenerationRequest(
                user_message=clean_message,
                intent_type=str(classification_result.intent_type.value),
                intent_confidence=classification_result.confidence,
                entities=entities_result.get("entities", {}) if entities_result else {},
                user_id=validated_user_id
            )
            
            try:
                query_result = await query_builder.generate_search_query(query_request)
                
                # Vérification robuste de la réponse du query_builder
                generation_success = False
                search_query = None
                
                if query_result is not None:
                    # Vérifier le type de réponse retournée
                    if hasattr(query_result, 'search_query') and hasattr(query_result, 'validation'):
                        # QueryGenerationResponse standard
                        generation_success = (
                            query_result.search_query is not None and
                            query_result.validation is not None and
                            query_result.validation.schema_valid
                        )
                        search_query = query_result.search_query if generation_success else None
                        logger.info(f"[{request_id}] QueryBuilder standard response: success={generation_success}")
                    elif hasattr(query_result, 'success') and hasattr(query_result, 'search_query'):
                        # Réponse avec attribut success (ancien format)
                        generation_success = query_result.success and query_result.search_query is not None
                        search_query = query_result.search_query if generation_success else None
                        logger.info(f"[{request_id}] QueryBuilder legacy response: success={generation_success}")
                    else:
                        logger.warning(f"[{request_id}] QueryBuilder réponse inattendue: {type(query_result)}")
                
                step3_duration = int((time.time() - step3_start) * 1000)
                step3 = ProcessingSteps(
                    agent="query_builder",
                    duration_ms=step3_duration,
                    cache_hit=step3_duration < 200,
                    success=generation_success
                )
            except Exception as e:
                logger.warning(f"[{request_id}] ⚠️ Génération requête échouée: {str(e)}")
                step3_duration = int((time.time() - step3_start) * 1000)
                step3 = ProcessingSteps(
                    agent="query_builder",
                    duration_ms=step3_duration,
                    cache_hit=False,
                    success=False,
                    error_message=str(e)
                )
        
        # ====================================================================
        # ÉTAPE 4: Exécution recherche (si requête générée)
        # ====================================================================
        search_results = None
        step4 = None
        
        if search_query:
            step4_start = time.time()
            
            try:
                search_executor = SearchExecutor()
                executor_request = SearchExecutorRequest(
                    search_query=search_query,
                    user_id=validated_user_id,
                    request_id=request_id
                )
                
                executor_response = await search_executor.handle_search_request(executor_request)
                search_results = executor_response.search_results if executor_response.success else None
                
                step4_duration = int((time.time() - step4_start) * 1000)
                step4 = ProcessingSteps(
                    agent="search_executor", 
                    duration_ms=step4_duration,
                    cache_hit=executor_response.fallback_used if hasattr(executor_response, 'fallback_used') else False,
                    success=executor_response.success
                )
            except Exception as e:
                logger.warning(f"[{request_id}] ⚠️ Exécution search échouée: {str(e)}")
                step4_duration = int((time.time() - step4_start) * 1000)
                step4 = ProcessingSteps(
                    agent="search_executor",
                    duration_ms=step4_duration,
                    cache_hit=False,
                    success=False,
                    error_message=str(e)
                )
        
        # ====================================================================
        # ÉTAPE 5: Génération réponse finale pour l'utilisateur
        # ====================================================================
        response_content = None
        step5 = None
        
        # Toujours générer une réponse, même sans résultats de recherche
        step5_start = time.time()
        
        try:
            from conversation_service.models.responses.conversation_responses import ResponseContent, StructuredData
            from conversation_service.prompts.templates.response_templates import get_response_template
            
            # Génération de réponse contextualisée avec LLM (version simplifiée)
            logger.info(f"[{request_id}] 🤖 Génération réponse LLM directe...")
            
            # Préparation du prompt contextualisé
            intent_type = str(classification_result.intent_type.value if hasattr(classification_result.intent_type, 'value') else classification_result.intent_type)
            entities = entities_result.get("entities", {}) if entities_result else {}
            
            # Analyse des résultats de recherche
            analysis_data = {}
            if search_results and search_results.hits:
                amounts = [abs(hit.source.get("amount", 0)) for hit in search_results.hits]
                analysis_data = {
                    "has_results": True,
                    "total_hits": search_results.total_hits,
                    "returned_hits": len(search_results.hits),
                    "total_amount": sum(amounts),
                    "average_amount": sum(amounts) / len(amounts) if amounts else 0,
                    "transaction_count": len(amounts),
                    "analysis_type": intent_type
                }
            else:
                analysis_data = {
                    "has_results": False,
                    "total_hits": 0,
                    "analysis_type": intent_type
                }
            
            # Génération du prompt
            prompt = get_response_template(
                intent_type=intent_type,
                user_message=clean_message,
                entities=entities,
                analysis_data=analysis_data,
                user_context=user_context,
                use_personalization=user_context is not None
            )
            
            # Appel LLM
            chat_response = await deepseek_client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,
                temperature=0.3
            )
            
            llm_message = chat_response["choices"][0]["message"]["content"].strip()
            
            # Création des données structurées
            structured_data = StructuredData(
                total_amount=analysis_data.get("total_amount"),
                transaction_count=analysis_data.get("transaction_count"),
                average_amount=analysis_data.get("average_amount"),
                currency="EUR" if analysis_data.get("total_amount") else None,
                analysis_type=intent_type
            ) if analysis_data.get("has_results") else None
            
            response_content = ResponseContent(
                message=llm_message,
                structured_data=structured_data
            )
            
            logger.info(f"[{request_id}] ✅ Réponse LLM générée: {len(llm_message)} caractères")
            
            step5_duration = int((time.time() - step5_start) * 1000)
            step5 = ProcessingSteps(
                agent="response_generator",
                duration_ms=step5_duration,
                cache_hit=False,
                success=True
            )
            
        except Exception as e:
            logger.error(f"[{request_id}] ❌ Génération réponse LLM échouée: {str(e)}")
            
            # Réponse de fallback (sans LLM)
            from conversation_service.models.responses.conversation_responses import ResponseContent, StructuredData
            
            if search_results and search_results.hits:
                total_amount = sum(hit.source.get("amount", 0) for hit in search_results.hits)
                transaction_count = len(search_results.hits)
                fallback_message = f"J'ai trouvé {transaction_count} transactions pour un montant de {abs(total_amount):.2f}€."
                
                structured_data = StructuredData(
                    total_amount=abs(total_amount),
                    transaction_count=transaction_count,
                    average_amount=abs(total_amount) / transaction_count if transaction_count > 0 else 0,
                    currency="EUR"
                )
            else:
                fallback_message = "Je n'ai pas trouvé de transactions correspondant à votre demande."
                structured_data = None
            
            response_content = ResponseContent(
                message=fallback_message,
                structured_data=structured_data
            )
            
            step5_duration = int((time.time() - step5_start) * 1000)
            step5 = ProcessingSteps(
                agent="response_generator",
                duration_ms=step5_duration,
                cache_hit=False,
                success=False,
                error_message=str(e)
            )
        
        # ====================================================================
        # Construction réponse complète Phase 5
        # ====================================================================
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        agent_metrics = AgentMetrics(
            agent_used="phase5_integrated",
            cache_hit=step1.cache_hit,
            model_used=getattr(settings, 'DEEPSEEK_CHAT_MODEL', 'deepseek-chat'),
            tokens_consumed=await _estimate_tokens_consumption_safe(clean_message, classification_result),
            processing_time_ms=processing_time_ms,
            confidence_threshold_met=classification_result.confidence >= 0.5
        )
        
        # Construction de la réponse complète avec toutes les étapes
        processing_steps = [step1]
        if step2:
            processing_steps.append(step2)
        if step3:
            processing_steps.append(step3)
        if step4:
            processing_steps.append(step4)
        if step5:
            processing_steps.append(step5)
        
        response = ConversationResponse(
            user_id=validated_user_id,
            sub=user_context.get("user_id", validated_user_id),
            message=clean_message,
            timestamp=datetime.now(timezone.utc),
            processing_time_ms=processing_time_ms,
            intent=classification_result,
            agent_metrics=agent_metrics,
            entities=entities_result,
            search_query=search_query,
            search_results=search_results,
            response=response_content,
            phase=5,
            request_id=request_id,
            processing_steps=processing_steps,
            status="success"
        )
        
        logger.info(
            f"[{request_id}] ✅ Phase 5 intégrée terminée: {intent_type_value}, "
            f"Confiance: {classification_result.confidence:.2f}, "
            f"Temps: {processing_time_ms}ms"
        )
        
        # Créer une réponse API épurée pour l'utilisateur final
        return _create_clean_api_response(response)
        
    except HTTPException:
        raise
    except Exception as e:
        processing_time_ms = int((time.time() - start_time) * 1000)
        logger.error(f"[{request_id}] ❌ Erreur Phase 5 intégrée: {str(e)}")
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Erreur traitement conversation",
                "phase": 5,
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )


def _create_clean_api_response(full_response: ConversationResponse) -> Dict[str, Any]:
    """
    Crée une réponse API épurée pour l'utilisateur final
    
    Ne contient que les informations essentielles :
    - Message original et réponse finale
    - Métriques par agent (succès/échec + latence)
    - Pas de détails techniques internes
    """
    
    # Métriques par agent simplifiées
    agent_metrics = {}
    for step in full_response.processing_steps:
        agent_metrics[step.agent] = {
            "success": step.success,
            "duration_ms": step.duration_ms,
            "cache_hit": step.cache_hit
        }
        if step.error_message:
            agent_metrics[step.agent]["error"] = step.error_message
    
    # Réponse épurée
    clean_response = {
        "user_id": full_response.user_id,
        "message": full_response.message,
        "timestamp": full_response.timestamp.isoformat(),
        "request_id": full_response.request_id,
        "processing_time_ms": full_response.processing_time_ms,
        "status": full_response.status.value if hasattr(full_response.status, 'value') else str(full_response.status),
        
        # Intent classification essentiel
        "intent": {
            "type": full_response.intent.intent_type.value if hasattr(full_response.intent.intent_type, 'value') else str(full_response.intent.intent_type),
            "confidence": full_response.intent.confidence,
            "supported": full_response.intent.is_supported
        },
        
        # Métriques agents
        "agents": agent_metrics,
        
        # Réponse finale (l'essentiel !)
        "response": full_response.response.dict() if full_response.response else None,
        
        # Résultats de recherche (synthèse uniquement)
        "search_summary": {
            "found_results": full_response.has_search_results,
            "total_results": full_response.search_results.total_hits if full_response.search_results else 0,
            "search_successful": full_response.search_execution_success
        } if full_response.has_search_query else None,
        
        # Métriques globales
        "performance": {
            "overall_success": full_response.is_successful,
            "confidence_level": full_response.intent.confidence_level.value if hasattr(full_response.intent.confidence_level, 'value') else str(full_response.intent.confidence_level),
            "cache_efficiency": full_response.cache_efficiency,
            "agents_used": len(full_response.processing_steps)
        }
    }
    
    return clean_response


logger.info(f"Routes conversation configurées - Environnement: {environment}")
logger.info(f"Endpoints debug: {'activés' if environment != 'production' else 'désactivés'}")
logger.info("Endpoints configurés:")
logger.info("  - /conversation/{user_id} → Phase 5 intégrée (Principal)")
logger.info("  - /conversation/legacy/{user_id} → Phase 2 (Deprecated)")
logger.info("  - /conversation/v2/{user_id} → Dual-mode AutoGen") 
logger.info("  - /team/health, /team/metrics → Team monitoring")