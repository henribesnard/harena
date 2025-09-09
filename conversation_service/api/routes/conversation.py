"""
Routes API pour conversation service - Version réécrite compatible JWT
"""
import logging
import time
import asyncio
import json
from typing import Dict, Any, Optional, List, AsyncGenerator
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, Request, Query
from fastapi.responses import StreamingResponse

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
# Old intent_classifier module removed - using streamlined architecture
# QueryGenerationRequest removed - using deterministic query building
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
    # get_multi_agent_team removed - streamlined architecture
    get_conversation_processor,
    get_conversation_persistence
)
from conversation_service.api.middleware.auth_middleware import get_current_user_id
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


@router.post("/conversation/{path_user_id}/stream")
async def analyze_conversation_stream(
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
    Endpoint de streaming pour conversation service Phase 5 - Réponse LLM en temps réel
    
    Exécute le workflow complet en arrière-plan (étapes 1-4) puis stream uniquement
    la réponse LLM finale chunk par chunk.
    
    Workflow interne (non streamé):
    - Étape 1: Classification d'intention
    - Étape 2: Extraction d'entités  
    - Étape 3: Génération de requête search
    - Étape 4: Exécution de la recherche
    - Étape 5: Génération de réponse finale (STREAMÉE)
    
    Format de stream simplifié:
    data: {"chunk": "texte partiel de la réponse"}
    data: {"chunk": "suite du texte"}
    data: {"done": true}
    
    Ou en cas d'erreur:
    data: {"error": "description de l'erreur"}
    """
    
    async def generate_conversation_stream() -> AsyncGenerator[str, None]:
        """Générateur de stream pour la réponse LLM uniquement"""
        
        start_time = time.time()
        request_id = f"stream_{int(time.time() * 1000)}_{validated_user_id}"
        
        try:
            # Validation du message (sans streaming)
            message_validation = validate_user_message(request_data.message)
            if not message_validation["valid"]:
                error_data = {
                    "error": "Message invalide",
                    "details": message_validation["errors"]
                }
                yield f"data: {json.dumps(error_data)}\n\n"
                return
            
            clean_message = sanitize_user_input(request_data.message)
            
            # ====================================================================
            # ÉTAPES 1-4: Traitement en interne (sans streaming)
            # ====================================================================
            
            # Obtenir les dépendances depuis app state
            deepseek_client = request.app.state.deepseek_client
            cache_manager = request.app.state.cache_manager
            
            # Extraire le token d'authentification pour search_service
            auth_token = None
            authorization = request.headers.get("Authorization")
            if authorization and authorization.startswith("Bearer "):
                auth_token = authorization[7:]  # Enlever "Bearer "
            
            # ÉTAPE 1: Classification intention + extraction entités unifiées
            from conversation_service.agents.financial.intent_entity_classifier import IntentEntityClassifier
            
            intent_entity_classifier = IntentEntityClassifier(
                deepseek_client=deepseek_client,
                cache_manager=cache_manager
            )
            
            unified_result = await intent_entity_classifier.execute(
                input_data=clean_message,
                context=user_context
            )
            
            classification_result = unified_result["intent_result"]
            entities_result = unified_result["entity_result"]
            
            intent_type_value = getattr(classification_result.intent_type, 'value', str(classification_result.intent_type))
            
            # ÉTAPE 2: Construction requête déterministe (logique pure)
            search_query = None
            if classification_result.is_supported:
                from conversation_service.core.deterministic_query_builder import DeterministicQueryBuilder
                
                query_builder = DeterministicQueryBuilder()
                
                try:
                    search_query = query_builder.build_query(
                        intent_result=classification_result,
                        entity_result=entities_result,
                        user_id=validated_user_id,
                        context=user_context
                    )
                    
                    # Validation de la requête construite
                    if search_query and not query_builder.validate_query(search_query):
                        search_query = None
                        
                except Exception as e:
                    logger.warning(f"[{request_id}] Erreur construction requête: {str(e)}")
                    search_query = None
            
            # ÉTAPE 3: Exécution recherche (si requête disponible)
            search_results = None
            if search_query:
                try:
                    search_executor = SearchExecutor()
                    executor_request = SearchExecutorRequest(
                        search_query=search_query,
                        user_id=validated_user_id,
                        request_id=request_id,
                        auth_token=auth_token
                    )
                    
                    executor_response = await search_executor.handle_search_request(executor_request)
                    search_results = executor_response.search_results if executor_response.success else None
                except Exception as e:
                    logger.warning(f"[{request_id}] Erreur exécution search: {str(e)}")
                    search_results = None
            
            # ====================================================================
            # ÉTAPE 4: Génération et streaming de la réponse LLM SEULEMENT
            # ====================================================================
            
            # Préparer les données pour la génération
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
            
            # Récupération historique contextualisé (en silence)
            conversation_history = await _get_contextualized_conversation_history(
                user_id=validated_user_id,
                current_message=clean_message,
                current_search_results=search_results,
                persistence_service=persistence_service,
                deepseek_client=deepseek_client,
                request_id=request_id,
                max_context_tokens=125000
            )
            
            # Génération du prompt
            from conversation_service.prompts.templates.response_templates import get_response_template
            
            prompt = get_response_template(
                intent_type=intent_type,
                user_message=clean_message,
                entities=entities,
                analysis_data=analysis_data,
                user_context=user_context,
                conversation_history=conversation_history,
                use_personalization=user_context is not None
            )
            
            # ====================================================================
            # STREAMING DE LA RÉPONSE LLM UNIQUEMENT
            # ====================================================================
            full_response_text = ""
            async for chunk in deepseek_client.chat_completion_stream(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,
                temperature=0.3
            ):
                if chunk.get("choices") and len(chunk["choices"]) > 0:
                    delta = chunk["choices"][0].get("delta", {})
                    if "content" in delta and delta["content"]:
                        content_chunk = delta["content"]
                        full_response_text += content_chunk
                        
                        # Stream uniquement le contenu de la réponse
                        yield f"data: {json.dumps({'chunk': content_chunk})}\n\n"
            
            # ====================================================================
            # PERSISTANCE EN ARRIÈRE-PLAN (sans streaming)
            # ====================================================================
            if persistence_service and full_response_text:
                try:
                    conversation = persistence_service.get_or_create_active_conversation(
                        user_id=validated_user_id,
                        conversation_title=f"Conversation du {datetime.now().strftime('%d/%m/%Y %H:%M')}"
                    )
                    
                    # Construction des données complètes pour le dataset
                    processing_time_ms = int((time.time() - start_time) * 1000)
                    
                    turn_data = {
                        "request_id": request_id,
                        "processing_time_ms": processing_time_ms,
                        "phase": 6,
                        "stream_mode": True,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "user_query": clean_message,
                        "intent_detected": {
                            "intent_type": intent_type_value,
                            "confidence": classification_result.confidence,
                            "is_supported": classification_result.is_supported,
                            "reasoning": getattr(classification_result, 'reasoning', '')
                        },
                        "entities_extracted": entities_result.get("entities", {}) if entities_result else {},
                        "query_generated": search_query.dict() if search_query else None,
                        "response_generated": full_response_text,
                        "search_results_count": search_results.total_hits if search_results else 0,
                        "client_feedback": {
                            "rating": None,
                            "comment": None,
                            "feedback_date": None
                        }
                    }
                    
                    persistence_service.add_conversation_turn(
                        conversation_id=conversation.id,
                        user_message=clean_message,
                        assistant_response=full_response_text,
                        turn_data=turn_data
                    )
                except Exception as e:
                    logger.error(f"[{request_id}] Erreur persistence: {str(e)}")
            
            # Signal de fin de stream
            yield f"data: {json.dumps({'done': True})}\n\n"
            
        except Exception as e:
            logger.error(f"[{request_id}] Erreur streaming: {str(e)}")
            error_response = {
                "error": str(e),
                "request_id": request_id
            }
            yield f"data: {json.dumps(error_response)}\n\n"
    
    return StreamingResponse(
        generate_conversation_stream(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Pour Nginx
        }
    )


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

@router.post("/conversation/feedback/{user_id}/{conversation_id}/{turn_number}")
async def submit_conversation_feedback(
    user_id: int,
    conversation_id: int, 
    turn_number: int,
    feedback: Dict[str, Any],
    validated_user_id: int = Depends(get_current_user_id),
    persistence_service: Optional[ConversationPersistenceService] = Depends(get_conversation_persistence)
):
    """
    Permet à l'utilisateur de donner un feedback sur un tour de conversation
    
    Payload attendu:
    {
        "rating": "positive" | "negative", 
        "comment": "Commentaire optionnel"
    }
    """
    try:
        # Vérification des droits utilisateur
        if validated_user_id != user_id:
            raise HTTPException(status_code=403, detail="Accès non autorisé")
        
        if not persistence_service:
            raise HTTPException(status_code=500, detail="Service de persistence non disponible")
        
        # Récupérer le tour de conversation
        conversation = persistence_service.get_conversation_with_turns(conversation_id)
        if not conversation or conversation.user_id != user_id:
            raise HTTPException(status_code=404, detail="Conversation non trouvée")
        
        # Trouver le tour spécifique
        target_turn = next((turn for turn in conversation.turns if turn.turn_number == turn_number), None)
        if not target_turn:
            raise HTTPException(status_code=404, detail="Tour de conversation non trouvé")
        
        # Mettre à jour le feedback dans les données JSON
        turn_data = target_turn.data or {}
        turn_data["client_feedback"] = {
            "rating": feedback.get("rating"),  # positive/negative
            "comment": feedback.get("comment"),
            "feedback_date": datetime.now(timezone.utc).isoformat()
        }
        
        # Sauvegarder les modifications
        target_turn.data = turn_data
        target_turn.updated_at = datetime.now(timezone.utc)
        persistence_service.db.commit()
        
        return {
            "success": True,
            "message": "Feedback enregistré avec succès",
            "conversation_id": conversation_id,
            "turn_number": turn_number,
            "feedback": turn_data["client_feedback"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur enregistrement feedback: {str(e)}")
        raise HTTPException(status_code=500, detail="Erreur interne du serveur")

@router.get("/conversation/dataset/{user_id}")
async def get_training_dataset(
    user_id: int,
    validated_user_id: int = Depends(get_current_user_id),
    persistence_service: Optional[ConversationPersistenceService] = Depends(get_conversation_persistence),
    limit: int = Query(100, description="Nombre maximum d'échantillons à retourner"),
    with_feedback_only: bool = Query(False, description="Retourner seulement les échantillons avec feedback")
):
    """
    Récupère les données d'entraînement pour un utilisateur
    
    Retourne les triplets (requête_utilisateur, intention+entités, query_elasticsearch) 
    pour entraîner un modèle de génération de requêtes
    """
    try:
        # Vérification des droits utilisateur
        if validated_user_id != user_id:
            raise HTTPException(status_code=403, detail="Accès non autorisé")
        
        if not persistence_service:
            raise HTTPException(status_code=500, detail="Service de persistence non disponible")
        
        # Récupérer toutes les conversations de l'utilisateur
        conversations = persistence_service.get_user_conversations(user_id)
        
        training_samples = []
        for conversation in conversations:
            conversation_with_turns = persistence_service.get_conversation_with_turns(conversation.id)
            
            for turn in conversation_with_turns.turns:
                turn_data = turn.data or {}
                
                # Filtrer par feedback si demandé
                if with_feedback_only:
                    client_feedback = turn_data.get("client_feedback", {})
                    if not client_feedback.get("rating"):
                        continue
                
                # Extraire les données d'entraînement
                if all(key in turn_data for key in ["user_query", "intent_detected", "entities_extracted", "query_generated"]):
                    sample = {
                        # Identifiants
                        "conversation_id": conversation.id,
                        "turn_number": turn.turn_number,
                        "timestamp": turn.created_at.isoformat(),
                        
                        # Données d'entraînement
                        "input_query": turn_data["user_query"],
                        "target_intent": turn_data["intent_detected"],
                        "target_entities": turn_data["entities_extracted"], 
                        "target_elasticsearch_query": turn_data["query_generated"],
                        
                        # Métadonnées de qualité
                        "success_rate": turn_data.get("success_rate", {}),
                        "processing_time_ms": turn_data.get("processing_time_ms", 0),
                        "client_feedback": turn_data.get("client_feedback", {}),
                        
                        # Pour validation
                        "response_generated": turn_data.get("response_generated"),
                        "search_results_count": turn_data.get("search_results_count", 0)
                    }
                    training_samples.append(sample)
        
        # Limiter les résultats
        training_samples = training_samples[-limit:] if limit else training_samples
        
        return {
            "user_id": user_id,
            "total_samples": len(training_samples),
            "with_feedback_filter": with_feedback_only,
            "samples": training_samples
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur récupération dataset: {str(e)}")
        raise HTTPException(status_code=500, detail="Erreur interne du serveur")

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
        """Test direct classification avec nouvelle architecture - NÉCESSITE AUTH"""
        request_id = f"debug_{int(time.time() * 1000)}"
        
        try:
            from conversation_service.agents.financial.intent_entity_classifier import IntentEntityClassifier
            
            classifier = IntentEntityClassifier(
                deepseek_client=deepseek_client,
                cache_manager=cache_manager
            )
            
            result = await classifier.execute(
                input_data=text,
                context={"user_id": 1}  # Debug context
            )
            
            return {
                "input": text,
                "intent_result": result["intent_result"].model_dump(mode="json") if hasattr(result["intent_result"], 'model_dump') else str(result["intent_result"]),
                "entity_result": result["entity_result"],
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "architecture": "streamlined"
            }
            
        except Exception as e:
            logger.error(f"[{request_id}] Erreur test classification debug: {str(e)}")
            return {
                "input": text,
                "error": str(e),
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "architecture": "streamlined"
            }


# ============================================================================
# LEGACY MULTI-AGENT ENDPOINTS - DISABLED (using streamlined architecture)
# ============================================================================

# Multi-agent team endpoints removed as part of architecture streamlining
# The new architecture uses:
# 1. IntentEntityClassifier (1 LLM call for both tasks)
# 2. DeterministicQueryBuilder (no LLM, pure logic)
# 3. ResponseGenerator (existing LLM call)
#
# This reduces latency from ~80s to expected ~5-10s


logger.debug(f"Conversation routes configured - Environment: {environment}")

# ============================================================================
# GESTION CONTEXTE CONVERSATION AVEC LIMITE 128K TOKENS
# ============================================================================

async def _get_contextualized_conversation_history(
    user_id: int,
    current_message: str,
    current_search_results: Optional[Any],
    persistence_service: Optional[ConversationPersistenceService],
    deepseek_client: DeepSeekClient,
    request_id: str,
    max_context_tokens: int = 125000
) -> Optional[Dict[str, Any]]:
    """
    Récupère et résume l'historique des conversations avec gestion intelligente des 128k tokens.
    
    Priorité dans l'allocation des tokens:
    1. Message utilisateur actuel (priorité absolue)
    2. Résultats de recherche actuels (priorité absolue) 
    3. Historique résumé (selon espace restant)
    
    Args:
        user_id: ID de l'utilisateur
        current_message: Message actuel de l'utilisateur
        current_search_results: Résultats de recherche actuels
        persistence_service: Service de persistance
        deepseek_client: Client DeepSeek pour résumer
        request_id: ID de la requête
        max_context_tokens: Limite maximum de tokens (128k par défaut)
        
    Returns:
        Dict contenant l'historique résumé ou None si pas d'historique
    """
    
    if not persistence_service:
        logger.debug(f"[{request_id}] Pas de service de persistance, historique ignoré")
        return None
    
    try:
        # Estimation des tokens obligatoires (priorité absolue)
        current_message_tokens = _estimate_text_tokens(current_message)
        current_results_tokens = 0
        
        if current_search_results and current_search_results.hits:
            # Estimation des tokens des résultats de recherche
            results_text = ""
            for hit in current_search_results.hits[:10]:  # Limiter à 10 résultats pour estimation
                source = hit.source or {}
                results_text += f"Date: {source.get('date', '')}, Montant: {source.get('amount', 0)}€, Marchand: {source.get('merchant', '')}, Description: {source.get('description', '')}\n"
            current_results_tokens = _estimate_text_tokens(results_text)
        
        # Tokens obligatoires + marge de sécurité pour le prompt système et la réponse
        reserved_tokens = current_message_tokens + current_results_tokens + 5000  # 5k pour système + réponse
        available_tokens_for_history = max_context_tokens - reserved_tokens
        
        logger.debug(f"[{request_id}] Tokens - Message: {current_message_tokens}, Résultats: {current_results_tokens}, Disponible historique: {available_tokens_for_history}")
        
        # Si pas assez de place pour l'historique, retourner None
        if available_tokens_for_history < 2000:  # Minimum 2k tokens pour un historique utile
            logger.debug(f"[{request_id}] Pas assez de tokens pour historique ({available_tokens_for_history} < 2000)")
            return None
        
        # Récupérer les conversations récentes de l'utilisateur (exclure la conversation actuelle)
        recent_conversations = persistence_service.get_user_conversations(
            user_id=user_id,
            limit=20,  # Dernières 20 conversations max
            offset=0
        )
        
        if not recent_conversations:
            logger.debug(f"[{request_id}] Aucun historique trouvé")
            return None
        
        # Construire l'historique brut des dernières conversations
        raw_history = []
        for conversation in recent_conversations[:10]:  # Limiter à 10 conversations
            try:
                conv_with_turns = persistence_service.get_conversation_with_turns(
                    conversation_id=conversation.id,
                    user_id=user_id
                )
                
                if conv_with_turns and conv_with_turns.turns:
                    # Prendre les 3 derniers tours de chaque conversation
                    recent_turns = sorted(conv_with_turns.turns, key=lambda x: x.turn_number)[-3:]
                    
                    for turn in recent_turns:
                        raw_history.append({
                            "user_message": turn.user_message,
                            "assistant_response": turn.assistant_response,
                            "timestamp": turn.created_at.isoformat(),
                            "intent": turn.data.get("intent_detected", {}).get("intent_type", "unknown") if turn.data else "unknown"
                        })
            except Exception as e:
                logger.debug(f"[{request_id}] Erreur récupération conversation {conversation.id}: {str(e)}")
                continue
        
        if not raw_history:
            logger.debug(f"[{request_id}] Aucun tour de conversation dans l'historique")
            return None
        
        # Trier par timestamp et prendre les plus récents
        raw_history = sorted(raw_history, key=lambda x: x["timestamp"], reverse=True)
        
        # Résumer l'historique en respectant la limite de tokens
        summarized_history = await _summarize_conversation_history(
            raw_history=raw_history,
            max_tokens=available_tokens_for_history,
            deepseek_client=deepseek_client,
            request_id=request_id
        )
        
        if summarized_history:
            logger.info(f"[{request_id}] ✅ Historique contextualisé intégré - {len(raw_history)} tours résumés")
            return {
                "summarized_context": summarized_history,
                "raw_turns_count": len(raw_history),
                "estimated_tokens": _estimate_text_tokens(summarized_history),
                "context_priority": "search_results_first"
            }
        
        return None
        
    except Exception as e:
        logger.error(f"[{request_id}] ❌ Erreur récupération historique: {str(e)}")
        return None


async def _summarize_conversation_history(
    raw_history: List[Dict[str, Any]],
    max_tokens: int,
    deepseek_client: DeepSeekClient,
    request_id: str
) -> Optional[str]:
    """
    Résume l'historique des conversations en respectant la limite de tokens
    """
    
    if not raw_history:
        return None
    
    try:
        # Construire le texte brut de l'historique
        history_text = ""
        for turn in raw_history[:20]:  # Max 20 tours les plus récents
            history_text += f"User: {turn['user_message'][:200]}...\n"  # Tronquer à 200 chars
            history_text += f"Assistant: {turn['assistant_response'][:300]}...\n"  # Tronquer à 300 chars
            history_text += f"Intent: {turn['intent']}\n---\n"
        
        # Estimation des tokens du texte brut
        estimated_tokens = _estimate_text_tokens(history_text)
        
        # Si l'historique brut tient dans la limite, le retourner tel quel
        if estimated_tokens <= max_tokens * 0.8:  # Garde 20% de marge
            logger.debug(f"[{request_id}] Historique brut utilisé ({estimated_tokens} tokens)")
            return f"HISTORIQUE DES CONVERSATIONS PRÉCÉDENTES:\n{history_text}"
        
        # Sinon, demander à DeepSeek de résumer
        summary_prompt = f"""Résume cet historique de conversations financières en gardant les informations essentielles pour le contexte :

{history_text}

Instructions :
- Résume en maximum {max_tokens // 4} mots
- Garde les patterns récurrents de demandes
- Mentionne les montants et marchands fréquemment discutés
- Conserve le style de communication de l'utilisateur
- Format concis et structuré

Résumé contextualisé :"""
        
        chat_response = await deepseek_client.chat_completion(
            messages=[{"role": "user", "content": summary_prompt}],
            max_tokens=min(1000, max_tokens // 4),  # Max 1000 tokens pour le résumé
            temperature=0.1  # Très factuel
        )
        
        summary = chat_response["choices"][0]["message"]["content"].strip()
        
        logger.debug(f"[{request_id}] Historique résumé par LLM ({_estimate_text_tokens(summary)} tokens)")
        return f"CONTEXTE DES CONVERSATIONS PRÉCÉDENTES:\n{summary}"
        
    except Exception as e:
        logger.error(f"[{request_id}] ❌ Erreur résumé historique: {str(e)}")
        return None


def _estimate_text_tokens(text: str) -> int:
    """
    Estimation du nombre de tokens dans un texte
    
    Utilise une approximation basée sur:
    - 1 token ≈ 4 caractères pour l'anglais
    - 1 token ≈ 3 caractères pour le français (plus dense)
    - Majoration de 20% pour la sécurité
    """
    if not text:
        return 0
    
    # Estimation de base : 3 chars = 1 token en français
    base_estimation = len(text) / 3.0
    
    # Majoration de sécurité de 20%
    safe_estimation = int(base_estimation * 1.2)
    
    return max(1, safe_estimation)  # Au moins 1 token


# ============================================================================
# INTÉGRATION NOUVELLE ARCHITECTURE - WORKFLOW STREAMLINÉ 3 ÉTAPES
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
    Nouvelle architecture streamlinée - 3 étapes au lieu de 5
    
    ÉTAPE 1: Classification intention + extraction entités (1 seul appel LLM)
    ÉTAPE 2: Construction requête déterministe (logique pure, pas de LLM)
    ÉTAPE 3: Génération réponse (après exécution search)
    """
    start_time = time.time()
    request_id = f"streamlined_{int(time.time() * 1000)}_{user_id}"
    
    logger.debug(f"[{request_id}] Streamlined workflow start for user {user_id}")
    
    try:
        # Validation message
        message_validation = validate_user_message(request.message)
        if not message_validation["valid"]:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "Message invalide",
                    "errors": message_validation["errors"],
                    "phase": 6,  # Nouvelle phase
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            )
        
        clean_message = sanitize_user_input(request.message)
        
        # Obtenir les dépendances depuis app state
        deepseek_client = http_request.app.state.deepseek_client
        cache_manager = http_request.app.state.cache_manager
        
        # Extraire le token d'authentification pour search_service
        auth_token = None
        authorization = http_request.headers.get("Authorization")
        if authorization and authorization.startswith("Bearer "):
            auth_token = authorization[7:]  # Enlever "Bearer "
        
        # ====================================================================
        # ÉTAPE 1: Classification intention + extraction entités unifiées
        # ====================================================================
        step1_start = time.time()
        
        from conversation_service.agents.financial.intent_entity_classifier import IntentEntityClassifier
        
        intent_entity_classifier = IntentEntityClassifier(
            deepseek_client=deepseek_client,
            cache_manager=cache_manager
        )
        
        unified_result = await intent_entity_classifier.execute(
            input_data=clean_message,
            context=user_context
        )
        
        classification_result = unified_result["intent_result"]
        entities_result = unified_result["entity_result"]
        
        step1_duration = int((time.time() - step1_start) * 1000)
        step1 = ProcessingSteps(
            agent="intent_entity_classifier",
            duration_ms=step1_duration,
            cache_hit=step1_duration < 200,  # Plus large car 2 tâches
            success=True
        )
        
        intent_type_value = getattr(classification_result.intent_type, 'value', str(classification_result.intent_type))
        
        # ====================================================================
        # ÉTAPE 2: Construction requête déterministe (logique pure)
        # ====================================================================
        search_query = None
        step2 = None
        
        if classification_result.is_supported:
            step2_start = time.time()
            
            from conversation_service.core.deterministic_query_builder import DeterministicQueryBuilder
            
            query_builder = DeterministicQueryBuilder()
            
            try:
                search_query = query_builder.build_query(
                    intent_result=classification_result,
                    entity_result=entities_result,
                    user_id=validated_user_id,
                    context=user_context
                )
                
                # Validation de la requête construite
                generation_success = query_builder.validate_query(search_query) if search_query else False
                
                step2_duration = int((time.time() - step2_start) * 1000)
                step2 = ProcessingSteps(
                    agent="deterministic_query_builder",
                    duration_ms=step2_duration,
                    cache_hit=False,  # Pas de cache pour logique déterministe
                    success=generation_success
                )
                
                logger.info(f"[{request_id}] Deterministic query built: success={generation_success}")
                
            except Exception as e:
                logger.warning(f"[{request_id}] ⚠️ Construction requête échouée: {str(e)}")
                step2_duration = int((time.time() - step2_start) * 1000)
                step2 = ProcessingSteps(
                    agent="deterministic_query_builder",
                    duration_ms=step2_duration,
                    cache_hit=False,
                    success=False,
                    error_message=str(e)
                )
                search_query = None
        
        # ====================================================================
        # ÉTAPE 3: Exécution recherche (si requête générée)
        # ====================================================================
        search_results = None
        resilience_metrics = None
        step3 = None
        
        # Exécution de la recherche si une requête a été générée
        if search_query:
            step3_start = time.time()
            
            try:
                search_executor = SearchExecutor()
                executor_request = SearchExecutorRequest(
                    search_query=search_query,
                    user_id=validated_user_id,
                    request_id=request_id,
                    auth_token=auth_token
                )
                
                executor_response = await search_executor.handle_search_request(executor_request)
                search_results = executor_response.search_results if executor_response.success else None
                
                # Calculer la durée d'abord
                step3_duration = int((time.time() - step3_start) * 1000)
                
                # Récupération ou création des métriques de résilience depuis SearchExecutor
                resilience_metrics = getattr(executor_response, 'resilience_metrics', None)
                
                # Créer des métriques par défaut si pas disponibles mais recherche réussie
                if resilience_metrics is None and executor_response.success:
                    from conversation_service.models.responses.conversation_responses import ResilienceMetrics
                    resilience_metrics = ResilienceMetrics(
                        circuit_breaker_triggered=False,
                        fallback_used=getattr(executor_response, 'fallback_used', False),
                        retry_attempts=0,
                        search_execution_time_ms=getattr(executor_response, 'execution_time_ms', step3_duration)
                    )
                
                step3 = ProcessingSteps(
                    agent="search_executor", 
                    duration_ms=step3_duration,
                    cache_hit=executor_response.fallback_used if hasattr(executor_response, 'fallback_used') else False,
                    success=executor_response.success
                )
            except Exception as e:
                logger.warning(f"[{request_id}] ⚠️ Exécution search échouée: {str(e)}")
                resilience_metrics = None  # Pas de métriques en cas d'exception
                step3_duration = int((time.time() - step3_start) * 1000)
                step3 = ProcessingSteps(
                    agent="search_executor",
                    duration_ms=step3_duration,
                    cache_hit=False,
                    success=False,
                    error_message=str(e)
                )
        
        # ====================================================================
        # NOUVELLE ÉTAPE: Génération réponse finale pour l'utilisateur
        # ====================================================================
        response_content = None
        step4 = None
        
        # Toujours générer une réponse, même sans résultats de recherche
        step4_start = time.time()
        
        try:
            from conversation_service.models.responses.conversation_responses import ResponseContent, StructuredData
            from conversation_service.prompts.templates.response_templates import get_response_template
            
            # Génération de réponse contextualisée avec LLM (version avec historique)
            logger.debug(f"[{request_id}] Generating LLM response with conversation history...")
            
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
            
            # Récupération et résumé de l'historique des conversations avec gestion 128k tokens
            conversation_history = await _get_contextualized_conversation_history(
                user_id=validated_user_id,
                current_message=clean_message,
                current_search_results=search_results,
                persistence_service=persistence_service,
                deepseek_client=deepseek_client,
                request_id=request_id,
                max_context_tokens=125000  # Garde 3k tokens pour la réponse
            )
            
            # Génération du prompt avec historique intégré
            prompt = get_response_template(
                intent_type=intent_type,
                user_message=clean_message,
                entities=entities,
                analysis_data=analysis_data,
                user_context=user_context,
                conversation_history=conversation_history,
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
            
            step4_duration = int((time.time() - step4_start) * 1000)
            step4 = ProcessingSteps(
                agent="response_generator",
                duration_ms=step4_duration,
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
            
            step4_duration = int((time.time() - step4_start) * 1000)
            step4 = ProcessingSteps(
                agent="response_generator",
                duration_ms=step4_duration,
                cache_hit=False,
                success=False,
                error_message=str(e)
            )
        
        # ====================================================================
        # Construction réponse complète - Architecture streamlinée
        # ====================================================================
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        agent_metrics = AgentMetrics(
            agent_used="streamlined_architecture",
            cache_hit=step1.cache_hit,
            model_used=getattr(settings, 'DEEPSEEK_CHAT_MODEL', 'deepseek-chat'),
            tokens_consumed=await _estimate_tokens_consumption_safe(clean_message, classification_result),
            processing_time_ms=processing_time_ms,
            confidence_threshold_met=classification_result.confidence >= 0.5
        )
        
        # Construction de la réponse complète avec toutes les étapes streamlinées
        processing_steps = [step1]
        if step2:
            processing_steps.append(step2)
        if step3:
            processing_steps.append(step3)
        if step4:
            processing_steps.append(step4)
        
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
            resilience_metrics=resilience_metrics,
            response=response_content,
            phase=6,
            request_id=request_id,
            processing_steps=processing_steps,
            status="success"
        )
        
        logger.info(f"[{request_id}] Conversation completed: {intent_type_value} ({processing_time_ms}ms)")
        
        # ====================================================================
        # PERSISTANCE: Enregistrement en base de données
        # ====================================================================
        if persistence_service:
            try:
                # Obtenir ou créer une conversation active pour l'utilisateur
                conversation = persistence_service.get_or_create_active_conversation(
                    user_id=validated_user_id,
                    conversation_title=f"Conversation du {datetime.now().strftime('%d/%m/%Y %H:%M')}"
                )
                
                # Ajouter ce tour de conversation 
                assistant_response_text = response_content.message if response_content else "Erreur de génération de réponse"
                
                # Construction des données complètes pour le dataset d'entraînement
                turn_data = {
                    # Métadonnées de traitement
                    "request_id": request_id,
                    "processing_time_ms": processing_time_ms,
                    "phase": 6,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    
                    # === DONNÉES POUR DATASET D'ENTRAÎNEMENT ===
                    
                    # 1. Requête utilisateur (déjà dans user_message mais dupliquée pour clarté)
                    "user_query": clean_message,
                    
                    # 2. Intention détectée (JSON complet)
                    "intent_detected": {
                        "intent_type": intent_type_value,
                        "confidence": classification_result.confidence,
                        "is_supported": classification_result.is_supported,
                        "reasoning": getattr(classification_result, 'reasoning', '')
                    },
                    
                    # 3. Entités extraites (JSON complet)
                    "entities_extracted": entities_result.get("entities", {}) if entities_result else {},
                    
                    # 4. Requête Elasticsearch générée (JSON complet)
                    "query_generated": search_query.dict() if search_query else None,
                    
                    # 5. Réponse générée (déjà dans assistant_response mais dupliquée)  
                    "response_generated": assistant_response_text,
                    
                    # 6. Taux de succès global
                    "success_rate": {
                        "workflow_success": all([
                            step1.success,
                            step2.success if step2 else True,
                            step3.success if step3 else True, 
                            step4.success if step4 else True
                        ]),
                        "agents_success_count": sum([
                            1 if step1.success else 0,
                            1 if (step2 and step2.success) else 0,
                            1 if (step3 and step3.success) else 0,
                            1 if (step4 and step4.success) else 0
                        ]),
                        "agents_total_count": 4
                    },
                    
                    # 7. Placeholder pour feedback client (sera mis à jour via une API dédiée)
                    "client_feedback": {
                        "rating": None,  # positive/negative
                        "comment": None,
                        "feedback_date": None
                    },
                    
                    # === DONNÉES TECHNIQUES EXISTANTES ===
                    "search_results_count": search_results.total_hits if search_results else 0,
                    "agent_metrics": {
                        "intent_entity_classifier": {"success": step1.success, "duration_ms": step1.duration_ms},
                        "deterministic_query_builder": {"success": step2.success if step2 else False, "duration_ms": step2.duration_ms if step2 else 0},
                        "search_executor": {"success": step3.success if step3 else False, "duration_ms": step3.duration_ms if step3 else 0},
                        "response_generator": {"success": step4.success if step4 else False, "duration_ms": step4.duration_ms if step4 else 0}
                    }
                }
                
                persistence_service.add_conversation_turn(
                    conversation_id=conversation.id,
                    user_message=clean_message,
                    assistant_response=assistant_response_text,
                    turn_data=turn_data
                )
                
                logger.debug(f"[{request_id}] Conversation saved - ID: {conversation.id}")
                
            except Exception as e:
                logger.error(f"[{request_id}] ❌ Erreur persistence conversation: {str(e)}")
                # On continue même si la persistence échoue
        
        # Créer une réponse API épurée pour l'utilisateur final
        return _create_clean_api_response(response)
        
    except HTTPException:
        raise
    except Exception as e:
        processing_time_ms = int((time.time() - start_time) * 1000)
        logger.error(f"[{request_id}] ❌ Erreur architecture streamlinée: {str(e)}")
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Erreur traitement conversation",
                "phase": 6,
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
        
        # Debug - Entités et query générée
        "debug_info": {
            "entities_detected": full_response.entities.get("entities") if full_response.entities else {},
            "search_query": full_response.search_query.dict(exclude_none=True) if hasattr(full_response, 'search_query') and full_response.search_query else None,
            "aggregations_returned": bool(full_response.search_results.aggregations) if full_response.search_results else False,
            "aggregation_keys": list(full_response.search_results.aggregations.keys()) if full_response.search_results and full_response.search_results.aggregations else []
        },
        
        # Métriques globales
        "performance": {
            "overall_success": full_response.is_successful,
            "confidence_level": full_response.intent.confidence_level.value if hasattr(full_response.intent.confidence_level, 'value') else str(full_response.intent.confidence_level),
            "cache_efficiency": full_response.cache_efficiency,
            "agents_used": len(full_response.processing_steps)
        }
    }
    
    return clean_response


logger.debug(f"Conversation routes configured - Environment: {environment}")