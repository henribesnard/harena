"""
🌐 Routes API - Endpoints de Détection d'Intention

Routes FastAPI pour l'API de détection d'intention avec tous les endpoints
nécessaires : détection, métriques, santé, batch processing.
"""

import time
import logging
from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import PlainTextResponse, JSONResponse
from conversation_service.models.intent import (
    IntentRequest, IntentResponse, BatchIntentRequest, BatchIntentResponse,
    HealthResponse, MetricsResponse, ErrorResponse
)
from conversation_service.models.exceptions import IntentDetectionError, ValidationError
from conversation_service.services.intent_detection.detector import OptimizedIntentService
from conversation_service.clients.cache.memory_cache import IntelligentMemoryCache
from conversation_service.utils.monitoring.intent_metrics import IntentMetricsCollector, record_intent_request
from conversation_service.api.dependencies import (
    get_intent_service, get_cache_manager, get_metrics_manager,
    get_user_context, public_endpoint_validator, admin_endpoint_validator,
    get_configuration
)

logger = logging.getLogger(__name__)

# Création du router
router = APIRouter(prefix="/api/v1", tags=["intent-detection"])


def create_error_response(
    error_type: str,
    message: str,
    details: Dict[str, Any] = None,
    request_id: str = None
) -> ErrorResponse:
    """
    Crée une réponse d'erreur structurée
    
    Args:
        error_type: Type d'erreur
        message: Message d'erreur
        details: Détails techniques optionnels
        request_id: ID de la requête
        
    Returns:
        ErrorResponse structurée
    """
    return ErrorResponse(  # ✅ UTILISÉ
        error=error_type,
        message=message,
        details=details or {},
        timestamp=time.time(),
        request_id=request_id
    )


@router.post("/detect-intent", response_model=IntentResponse)
async def detect_intent_endpoint(
    request: IntentRequest,
    http_request: Request,
    intent_service: OptimizedIntentService = Depends(get_intent_service),
    user_context: Dict[str, Any] = Depends(get_user_context),
    metrics_collector: IntentMetricsCollector = Depends(get_metrics_manager),  # ✅ UTILISÉ
    validation: Dict[str, Any] = Depends(public_endpoint_validator)  # ✅ UTILISÉ
):
    """
    🎯 Endpoint principal : Détection d'intention
    
    Détecte l'intention d'une requête utilisateur avec extraction d'entités
    et suggestions contextuelles.
    
    - **query**: Texte utilisateur à analyser (1-500 caractères)
    - **user_id**: Identifiant utilisateur optionnel pour analytics
    - **use_deepseek_fallback**: Autoriser fallback DeepSeek si règles insuffisantes
    - **force_method**: Forcer utilisation méthode spécifique (debug)
    - **enable_cache**: Utiliser cache si disponible
    
    Returns:
        IntentResponse avec intention, confiance, entités et suggestions
    """
    start_time = time.time()
    request_id = validation.get("request_id", f"req_{int(time.time())}")  # ✅ UTILISÉ
    
    try:
        # Validation des données d'entrée
        if not validation.get("is_valid", True):  # ✅ UTILISÉ
            error_response = create_error_response(
                "VALIDATION_ERROR",
                "Validation de la requête échouée",
                details=validation.get("errors", {}),
                request_id=request_id
            )
            return JSONResponse(
                status_code=400,
                content=error_response.dict()
            )
        
        # Ajout contexte utilisateur à la requête
        if user_context.get("user_id") and not request.user_id:
            request.user_id = user_context["user_id"]
        
        # Détection d'intention
        result = await intent_service.detect_intent(request)
        
        # Création réponse
        response = IntentResponse(**result)
        
        # Enregistrement métriques avec collecteur  # ✅ UTILISÉ
        await metrics_collector.record_request(
            query=request.query,
            intent=response.intent,
            confidence=response.confidence,
            processing_time_ms=response.processing_time_ms,
            method_used=response.method_used,
            user_id=str(request.user_id) if request.user_id else None,
            request_id=request_id,
            cached=response.cached,
            entities_count=len(response.entities)
        )
        
        # Enregistrement métriques legacy (background)
        record_intent_request(
            query=request.query,
            result=result,
            user_id=str(request.user_id) if request.user_id else None
        )
        
        # Logging pour audit avec validation info  # ✅ UTILISÉ
        logger.info(
            f"Intent detected: {response.intent} "
            f"(confidence: {response.confidence:.3f}, "
            f"method: {response.method_used}, "
            f"time: {response.processing_time_ms:.1f}ms) "
            f"for user {user_context.get('user_id', 'anonymous')} "
            f"[req_id: {request_id}, validation: {validation.get('validation_time_ms', 0):.1f}ms]"
        )
        
        return response
        
    except ValidationError as e:
        logger.warning(f"Validation error: {e}")
        
        # Métriques d'erreur  # ✅ UTILISÉ
        await metrics_collector.record_error(
            error_type="validation_error",
            query=request.query,
            user_id=str(request.user_id) if request.user_id else None,
            request_id=request_id
        )
        
        error_response = create_error_response(
            "VALIDATION_ERROR",
            str(e),
            details={"field": getattr(e, 'field_name', None)},
            request_id=request_id
        )
        
        return JSONResponse(
            status_code=400,
            content=error_response.dict()
        )
    
    except IntentDetectionError as e:
        logger.error(f"Intent detection error: {e}")
        
        # Métriques d'erreur  # ✅ UTILISÉ
        await metrics_collector.record_error(
            error_type="intent_detection_error",
            query=request.query,
            user_id=str(request.user_id) if request.user_id else None,
            request_id=request_id,
            error_details={"attempted_methods": getattr(e, 'attempted_methods', [])}
        )
        
        error_response = create_error_response(
            "INTENT_DETECTION_ERROR",
            str(e),
            details={
                "query_truncated": request.query[:50] + "..." if len(request.query) > 50 else request.query,
                "attempted_methods": getattr(e, 'attempted_methods', [])
            },
            request_id=request_id
        )
        
        return JSONResponse(
            status_code=422,
            content=error_response.dict()
        )
    
    except Exception as e:
        logger.error(f"Unexpected error in intent detection: {e}")
        
        # Métriques d'erreur système  # ✅ UTILISÉ
        await metrics_collector.record_error(
            error_type="system_error",
            query=request.query,
            user_id=str(request.user_id) if request.user_id else None,
            request_id=request_id,
            error_details={"exception_type": type(e).__name__}
        )
        
        error_response = create_error_response(
            "INTERNAL_SERVER_ERROR",
            "Une erreur inattendue s'est produite",
            details={"exception_type": type(e).__name__},
            request_id=request_id
        )
        
        return JSONResponse(
            status_code=500,
            content=error_response.dict()
        )


@router.post("/batch-detect", response_model=BatchIntentResponse)
async def batch_detect_intent_endpoint(
    request: BatchIntentRequest,
    http_request: Request,
    intent_service: OptimizedIntentService = Depends(get_intent_service),
    user_context: Dict[str, Any] = Depends(get_user_context),
    metrics_collector: IntentMetricsCollector = Depends(get_metrics_manager),  # ✅ UTILISÉ
    validation: Dict[str, Any] = Depends(public_endpoint_validator)  # ✅ UTILISÉ
):
    """
    📦 Endpoint batch : Détection d'intention multiple
    
    Traite plusieurs requêtes en parallèle pour optimiser le débit.
    Maximum 100 requêtes par batch.
    
    - **queries**: Liste des textes à analyser
    - **user_id**: Identifiant utilisateur pour toutes les requêtes
    - **use_deepseek_fallback**: Autoriser fallback DeepSeek
    - **parallel_processing**: Traitement parallèle (recommandé)
    
    Returns:
        BatchIntentResponse avec résultats et métriques batch
    """
    start_time = time.time()
    request_id = validation.get("request_id", f"batch_{int(time.time())}")  # ✅ UTILISÉ
    
    try:
        # Validation spécifique batch  # ✅ UTILISÉ
        if not validation.get("is_valid", True):
            error_response = create_error_response(
                "BATCH_VALIDATION_ERROR",
                "Validation du batch échouée",
                details={
                    "errors": validation.get("errors", {}),
                    "batch_size": len(request.queries) if request.queries else 0
                },
                request_id=request_id
            )
            return JSONResponse(
                status_code=400,
                content=error_response.dict()
            )
        
        # Validation taille batch
        if len(request.queries) > 100:
            error_response = create_error_response(
                "BATCH_TOO_LARGE",
                "Maximum 100 requêtes par batch",
                details={"received": len(request.queries), "maximum": 100},
                request_id=request_id
            )
            return JSONResponse(
                status_code=400,
                content=error_response.dict()
            )
        
        logger.info(f"Starting batch processing: {len(request.queries)} queries [req_id: {request_id}]")
        
        # Enregistrement début batch  # ✅ UTILISÉ
        await metrics_collector.record_batch_start(
            batch_size=len(request.queries),
            user_id=str(request.user_id) if request.user_id else user_context.get("user_id"),
            request_id=request_id
        )
        
        # Traitement batch
        results = await intent_service.batch_detect_intent(
            queries=request.queries,
            user_id=str(request.user_id) if request.user_id else user_context.get("user_id"),
            use_deepseek_fallback=request.use_deepseek_fallback
        )
        
        # Conversion en IntentResponse objects
        intent_responses = []
        successful_requests = 0
        failed_requests = 0
        
        for result in results:
            try:
                if "error" not in result:
                    intent_responses.append(IntentResponse(**result))
                    successful_requests += 1
                else:
                    # Créer réponse d'erreur
                    error_response_item = IntentResponse(
                        intent="UNKNOWN",
                        intent_code="UNKNOWN", 
                        confidence=0.0,
                        processing_time_ms=0.0,
                        method_used="error",
                        query=result.get("query", ""),
                        entities={},
                        suggestions=[],
                        cost_estimate=0.0
                    )
                    intent_responses.append(error_response_item)
                    failed_requests += 1
            except Exception as e:
                logger.warning(f"Error creating response for batch item: {e}")
                failed_requests += 1
        
        total_processing_time = (time.time() - start_time) * 1000
        
        # Métriques batch
        batch_metrics = {
            "total_queries": len(request.queries),
            "avg_processing_time_ms": round(
                sum(r.processing_time_ms for r in intent_responses) / len(intent_responses), 2
            ) if intent_responses else 0,
            "total_cost": sum(r.cost_estimate for r in intent_responses),
            "method_distribution": {},
            "intent_distribution": {},
            "validation_time_ms": validation.get("validation_time_ms", 0)  # ✅ UTILISÉ
        }
        
        # Distribution méthodes et intentions
        for response in intent_responses:
            method = response.method_used
            batch_metrics["method_distribution"][method] = batch_metrics["method_distribution"].get(method, 0) + 1
            
            intent = response.intent
            batch_metrics["intent_distribution"][intent] = batch_metrics["intent_distribution"].get(intent, 0) + 1
        
        # Enregistrement fin batch  # ✅ UTILISÉ
        await metrics_collector.record_batch_completion(
            batch_size=len(request.queries),
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            total_processing_time_ms=total_processing_time,
            request_id=request_id
        )
        
        # Réponse batch
        batch_response = BatchIntentResponse(
            results=intent_responses,
            batch_metrics=batch_metrics,
            total_processing_time_ms=total_processing_time,
            successful_requests=successful_requests,
            failed_requests=failed_requests
        )
        
        logger.info(
            f"Batch completed: {successful_requests}/{len(request.queries)} successful "
            f"in {total_processing_time:.1f}ms [req_id: {request_id}]"
        )
        
        return batch_response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch processing error: {e}")
        
        # Métriques d'erreur batch  # ✅ UTILISÉ
        await metrics_collector.record_error(
            error_type="batch_processing_error",
            user_id=str(request.user_id) if request.user_id else user_context.get("user_id"),
            request_id=request_id,
            error_details={
                "batch_size": len(request.queries) if request.queries else 0,
                "exception_type": type(e).__name__
            }
        )
        
        error_response = create_error_response(
            "BATCH_PROCESSING_ERROR",
            str(e),
            details={
                "batch_size": len(request.queries) if request.queries else 0,
                "exception_type": type(e).__name__
            },
            request_id=request_id
        )
        
        return JSONResponse(
            status_code=500,
            content=error_response.dict()
        )


@router.get("/health", response_model=HealthResponse)
async def health_check_endpoint(
    intent_service: OptimizedIntentService = Depends(get_intent_service),
    cache_manager: IntelligentMemoryCache = Depends(get_cache_manager),
    metrics_collector: IntentMetricsCollector = Depends(get_metrics_manager)  # ✅ UTILISÉ
):
    """
    🏥 Endpoint santé : Vérification état du service
    
    Vérifie l'état de tous les composants du service et retourne
    les métriques de base pour monitoring.
    
    Returns:
        HealthResponse avec statut et métriques essentielles
    """
    try:
        # Vérification santé complète
        health_status = await intent_service.health_check()
        
        # Métriques de base
        service_metrics = intent_service.get_metrics()
        cache_stats = cache_manager.get_stats()
        
        # Métriques collecteur  # ✅ UTILISÉ
        collector_stats = await metrics_collector.get_health_metrics()
        
        # Construction réponse santé
        response = HealthResponse(
            status=health_status["status"],
            service_name="conversation-service",
            version="2.0.0",
            timestamp=time.time(),
            
            # Statuts composants
            rule_engine_status=health_status["components"]["rule_engine"],
            deepseek_client_status=health_status["components"]["deepseek_client"],
            cache_status=health_status["components"]["cache"],
            
            # Métriques de base avec collecteur
            total_requests=max(
                service_metrics.get("total_requests", 0),
                collector_stats.get("total_requests", 0)
            ),
            average_latency_ms=service_metrics.get("avg_latency_ms", 0.0),
            cache_hit_rate=cache_stats["cache_performance"]["hit_rate"],
            
            # Configuration
            deepseek_fallback_enabled=health_status["configuration"]["deepseek_enabled"],
            cache_enabled=health_status["configuration"]["cache_enabled"]
        )
        
        # Log statut santé avec métriques collecteur  # ✅ UTILISÉ
        if health_status["status"] != "healthy":
            logger.warning(
                f"Service health status: {health_status['status']} "
                f"(collector metrics: {collector_stats.get('status', 'unknown')})"
            )
        
        return response
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        
        # Réponse d'erreur minimale
        return HealthResponse(
            status="error",
            service_name="conversation-service",
            version="2.0.0",
            timestamp=time.time(),
            rule_engine_status="unknown",
            deepseek_client_status="unknown",
            cache_status="unknown"
        )


@router.get("/metrics", response_model=MetricsResponse)
async def metrics_endpoint(
    detailed: bool = Query(False, description="Inclure métriques détaillées"),
    intent_service: OptimizedIntentService = Depends(get_intent_service),
    metrics_collector: IntentMetricsCollector = Depends(get_metrics_manager),  # ✅ UTILISÉ
    validation: Dict[str, Any] = Depends(admin_endpoint_validator)  # ✅ UTILISÉ
):
    """
    📊 Endpoint métriques : Analytics et performance
    
    Retourne métriques complètes du service pour monitoring
    et optimisation des performances.
    
    - **detailed**: Inclure analytics détaillées (admin seulement)
    
    Returns:
        MetricsResponse avec métriques complètes ou basiques
    """
    try:
        # Vérification permissions admin  # ✅ UTILISÉ
        if detailed and not validation.get("is_admin", False):
            error_response = create_error_response(
                "INSUFFICIENT_PERMISSIONS",
                "Métriques détaillées réservées aux administrateurs",
                details={"required_role": "admin", "current_role": validation.get("user_role", "user")},
                request_id=validation.get("request_id")
            )
            return JSONResponse(
                status_code=403,
                content=error_response.dict()
            )
        
        # Métriques service principal
        service_metrics = intent_service.get_metrics()
        
        if detailed:
            # Métriques détaillées (admin)  # ✅ UTILISÉ
            comprehensive_report = await metrics_collector.get_comprehensive_report()
            
            response = MetricsResponse(
                **service_metrics,
                **comprehensive_report.get("historical_analysis", {}),
                component_metrics=service_metrics.get("component_metrics", {}),
                detailed_analytics={
                    "intent_analytics": comprehensive_report.get("intent_analytics", {}),
                    "method_analytics": comprehensive_report.get("method_analytics", {}),
                    "user_analytics": comprehensive_report.get("user_analytics", {}),
                    "real_time_performance": comprehensive_report.get("real_time_performance", {}),
                    "admin_access_time": time.time(),
                    "access_validated_by": validation.get("validator_id", "unknown")
                }
            )
        else:
            # Métriques basiques (public) avec données collecteur  # ✅ UTILISÉ
            basic_collector_metrics = await metrics_collector.get_basic_metrics()
            
            response = MetricsResponse(
                **service_metrics,
                collector_metrics=basic_collector_metrics
            )
        
        return response
        
    except Exception as e:
        logger.error(f"Metrics endpoint error: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Metrics unavailable",
                "message": "Impossible de récupérer les métriques"
            }
        )


@router.get("/supported-intents")
async def supported_intents_endpoint(
    config_data: Dict[str, Any] = Depends(get_configuration),
    validation: Dict[str, Any] = Depends(public_endpoint_validator)  # ✅ UTILISÉ
):
    """
    📋 Endpoint intentions : Liste des intentions supportées
    
    Retourne la liste complète des intentions financières supportées
    avec leurs métadonnées.
    
    Returns:
        Dict avec intentions supportées et configuration
    """
    from ...config import get_supported_intents
    
    supported_intents = get_supported_intents()
    
    return {
        "supported_intents": list(supported_intents.keys()),
        "intent_details": supported_intents,
        "total_intents": len(supported_intents),
        "categories": {
            "financial": [
                "ACCOUNT_BALANCE", "SEARCH_BY_CATEGORY", "BUDGET_ANALYSIS",
                "TRANSFER", "SEARCH_BY_DATE", "CARD_MANAGEMENT"
            ],
            "conversational": ["GREETING", "HELP", "GOODBYE"],
            "fallback": ["UNKNOWN"]
        },
        "service_info": {
            "version": config_data["version"],
            "deepseek_enabled": config_data["deepseek_enabled"],
            "request_validated": validation.get("is_valid", False),  # ✅ UTILISÉ
            "validation_time_ms": validation.get("validation_time_ms", 0)  # ✅ UTILISÉ
        }
    }


@router.post("/cache/clear")
async def clear_cache_endpoint(
    cache_manager: IntelligentMemoryCache = Depends(get_cache_manager),
    metrics_collector: IntentMetricsCollector = Depends(get_metrics_manager),  # ✅ UTILISÉ
    validation: Dict[str, Any] = Depends(admin_endpoint_validator)  # ✅ UTILISÉ
):
    """
    🗑️ Endpoint admin : Vider le cache
    
    Vide complètement le cache mémoire. Utile pour maintenance
    ou après mise à jour de modèles.
    
    Returns:
        Statistiques de nettoyage
    """
    try:
        # Vérification permissions  # ✅ UTILISÉ
        if not validation.get("is_admin", False):
            error_response = create_error_response(
                "INSUFFICIENT_PERMISSIONS",
                "Opération réservée aux administrateurs",
                details={"required_role": "admin"},
                request_id=validation.get("request_id")
            )
            return JSONResponse(
                status_code=403,
                content=error_response.dict()
            )
        
        entries_removed = cache_manager.clear()
        
        # Enregistrement action admin  # ✅ UTILISÉ
        await metrics_collector.record_admin_action(
            action="cache_clear",
            admin_user=validation.get("user_id", "unknown"),
            details={"entries_removed": entries_removed},
            request_id=validation.get("request_id")
        )
        
        logger.info(
            f"Cache cleared by admin {validation.get('user_id', 'unknown')}: "
            f"{entries_removed} entries removed"
        )
        
        return {
            "status": "success",
            "message": "Cache vidé avec succès",
            "entries_removed": entries_removed,
            "timestamp": time.time(),
            "admin_user": validation.get("user_id", "unknown")  # ✅ UTILISÉ
        }
        
    except Exception as e:
        logger.error(f"Cache clear error: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Cache clear failed",
                "message": str(e)
            }
        )


@router.post("/cache/optimize")
async def optimize_cache_endpoint(
    cache_manager: IntelligentMemoryCache = Depends(get_cache_manager),
    metrics_collector: IntentMetricsCollector = Depends(get_metrics_manager),  # ✅ UTILISÉ
    validation: Dict[str, Any] = Depends(admin_endpoint_validator)  # ✅ UTILISÉ
):
    """
    ⚡ Endpoint admin : Optimiser le cache
    
    Lance l'optimisation du cache : supprime entrées expirées
    et peu utilisées pour améliorer performance.
    
    Returns:
        Statistiques d'optimisation
    """
    try:
        # Vérification permissions  # ✅ UTILISÉ
        if not validation.get("is_admin", False):
            error_response = create_error_response(
                "INSUFFICIENT_PERMISSIONS",
                "Opération réservée aux administrateurs",
                request_id=validation.get("request_id")
            )
            return JSONResponse(
                status_code=403,
                content=error_response.dict()
            )
        
        optimization_stats = cache_manager.optimize_cache()
        
        # Enregistrement action admin  # ✅ UTILISÉ
        await metrics_collector.record_admin_action(
            action="cache_optimize",
            admin_user=validation.get("user_id", "unknown"),
            details=optimization_stats,
            request_id=validation.get("request_id")
        )
        
        logger.info(
            f"Cache optimized by admin {validation.get('user_id', 'unknown')}: "
            f"{optimization_stats['total_removed']} entries removed"
        )
        
        return {
            "status": "success",
            "message": "Cache optimisé avec succès",
            "optimization_stats": optimization_stats,
            "timestamp": time.time(),
            "admin_user": validation.get("user_id", "unknown")  # ✅ UTILISÉ
        }
        
    except Exception as e:
        logger.error(f"Cache optimization error: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Cache optimization failed",
                "message": str(e)
            }
        )


@router.get("/metrics/export", response_class=PlainTextResponse)
async def export_metrics_csv_endpoint(
    hours: int = Query(24, description="Nombre d'heures à exporter", ge=1, le=168),
    metrics_collector: IntentMetricsCollector = Depends(get_metrics_manager),  # ✅ UTILISÉ
    validation: Dict[str, Any] = Depends(admin_endpoint_validator)  # ✅ UTILISÉ
):
    """
    📤 Endpoint admin : Export métriques CSV
    
    Exporte les métriques détaillées en format CSV pour analyse
    externe ou archivage.
    
    - **hours**: Nombre d'heures d'historique à exporter (1-168)
    
    Returns:
        CSV des métriques avec headers
    """
    try:
        # Vérification permissions admin  # ✅ UTILISÉ
        if not validation.get("is_admin", False):
            return PlainTextResponse(
                content="ERROR: Insufficient permissions for metrics export",
                status_code=403
            )
        
        csv_content = await metrics_collector.export_metrics_csv(hours)  # ✅ UTILISÉ
        
        # Enregistrement export  # ✅ UTILISÉ
        await metrics_collector.record_admin_action(
            action="metrics_export",
            admin_user=validation.get("user_id", "unknown"),
            details={"hours_exported": hours},
            request_id=validation.get("request_id")
        )
        
        logger.info(
            f"Metrics exported by admin {validation.get('user_id', 'unknown')}: "
            f"{hours}h of data"
        )
        
        return PlainTextResponse(
            content=csv_content,
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=intent_metrics_{hours}h.csv",
                "X-Exported-By": validation.get("user_id", "unknown")  # ✅ UTILISÉ
            }
        )
        
    except Exception as e:
        logger.error(f"Metrics export error: {e}")
        return PlainTextResponse(
            content=f"ERROR: Metrics export failed - {str(e)}",
            status_code=500
        )


@router.post("/test-comprehensive")
async def comprehensive_test_endpoint(
    intent_service: OptimizedIntentService = Depends(get_intent_service),
    metrics_collector: IntentMetricsCollector = Depends(get_metrics_manager),  # ✅ UTILISÉ
    validation: Dict[str, Any] = Depends(admin_endpoint_validator)  # ✅ UTILISÉ
):
    """
    🧪 Endpoint test : Test complet avec métriques
    
    Lance une série de tests complets pour valider le fonctionnement
    du service et générer métriques de performance.
    
    Returns:
        Résultats de tests avec statistiques détaillées
    """
    
    # Vérification permissions  # ✅ UTILISÉ
    if not validation.get("is_admin", False):
        error_response = create_error_response(
            "INSUFFICIENT_PERMISSIONS",
            "Tests complets réservés aux administrateurs",
            request_id=validation.get("request_id")
        )
        return JSONResponse(
            status_code=403,
            content=error_response.dict()
        )
    
    # Cases de test (reprises du fichier original)
    test_cases = [
        ("bonjour comment ça va", "GREETING"),
        ("quel est mon solde compte courant", "ACCOUNT_BALANCE"), 
        ("mes dépenses restaurant ce mois", "SEARCH_BY_CATEGORY"),
        ("faire un virement de 100 euros", "TRANSFER"),
        ("au revoir merci", "GOODBYE"),
        ("mes courses chez carrefour en janvier", "SEARCH_BY_CATEGORY"),
        ("virer 250 euros à Marie", "TRANSFER"),
        ("combien j'ai dépensé en transport", "BUDGET_ANALYSIS"),
        ("bloquer ma carte visa", "CARD_MANAGEMENT"),
        ("historique des transactions de décembre", "SEARCH_BY_DATE"),
        ("aide moi", "HELP"),
        ("quelque chose de très complexe et ambigu", "UNKNOWN")
    ]
    
    results = []
    start_test = time.time()
    test_id = f"test_{int(time.time())}"
    
    # Enregistrement début test  # ✅ UTILISÉ
    await metrics_collector.record_test_start(
        test_id=test_id,
        test_type="comprehensive",
        admin_user=validation.get("user_id", "unknown"),
        test_cases_count=len(test_cases)
    )
    
    for i, (query, expected) in enumerate(test_cases):
        request = IntentRequest(query=query, user_id="test_user")
        result = await intent_service.detect_intent(request)
        
        is_correct = result["intent"] == expected or (expected == "UNKNOWN" and result["confidence"] < 0.5)
        
        test_result = {
            "query": query,
            "expected": expected,
            "detected": result["intent"],
            "confidence": result["confidence"],
            "correct": is_correct,
            "latency_ms": result["processing_time_ms"],
            "method": result["method_used"],
            "entities": result["entities"]
        }
        results.append(test_result)
        
        # Enregistrement résultat individuel  # ✅ UTILISÉ
        await metrics_collector.record_test_case_result(
            test_id=test_id,
            case_index=i,
            query=query,
            expected_intent=expected,
            detected_intent=result["intent"],
            confidence=result["confidence"],
            correct=is_correct,
            processing_time_ms=result["processing_time_ms"]
        )
    
    total_test_time = (time.time() - start_test) * 1000
    
    # Statistiques (logique exacte fichier original)
    correct_count = sum(1 for r in results if r["correct"])
    accuracy = correct_count / len(results)
    avg_latency = sum(r["latency_ms"] for r in results) / len(results)
    fast_responses = sum(1 for r in results if r["latency_ms"] < 50)  # Target latency
    
    # Enregistrement fin test  # ✅ UTILISÉ
    await metrics_collector.record_test_completion(
        test_id=test_id,
        total_cases=len(test_cases),
        correct_count=correct_count,
        accuracy=accuracy,
        avg_latency_ms=avg_latency,
        total_test_time_ms=total_test_time,
        admin_user=validation.get("user_id", "unknown")
    )
    
    test_response = {
        "test_results": results,
        "statistics": {
            "total_tests": len(results),
            "correct_predictions": correct_count,
            "accuracy_rate": round(accuracy, 3),
            "avg_latency_ms": round(avg_latency, 2),
            "fast_responses": fast_responses,
            "fast_response_rate": round(fast_responses / len(results), 3),
            "total_test_time_ms": round(total_test_time, 2),
            "meets_targets": {
                "latency": avg_latency <= 50,  # Target from config
                "accuracy": accuracy >= 0.85   # Target from config
            }
        },
        "service_metrics": intent_service.get_metrics(),
        "test_metadata": {
            "test_id": test_id,
            "admin_user": validation.get("user_id", "unknown"),  # ✅ UTILISÉ
            "timestamp": time.time(),
            "validation_time_ms": validation.get("validation_time_ms", 0)  # ✅ UTILISÉ
        }
    }
    
    logger.info(
        f"Comprehensive test completed by admin {validation.get('user_id', 'unknown')}: "
        f"{accuracy:.1%} accuracy, {avg_latency:.1f}ms avg latency [test_id: {test_id}]"
    )
    
    return test_response


# Middleware d'erreur global pour utiliser ErrorResponse
@router.middleware("http")
async def error_handling_middleware(request: Request, call_next):
    """
    Middleware de gestion d'erreurs utilisant ErrorResponse
    """
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        logger.error(f"Unhandled error in {request.url.path}: {e}")
        
        error_response = create_error_response(  # ✅ UTILISÉ
            "UNHANDLED_ERROR",
            "Erreur non gérée dans l'API",
            details={
                "path": str(request.url.path),
                "method": request.method,
                "exception_type": type(e).__name__
            },
            request_id=f"middleware_{int(time.time())}"
        )
        
        return JSONResponse(
            status_code=500,
            content=error_response.dict()
        )


# Endpoint de diagnostic utilisant ErrorResponse
@router.get("/debug/error-test")
async def error_test_endpoint(
    error_type: str = Query("validation", description="Type d'erreur à tester"),
    validation: Dict[str, Any] = Depends(admin_endpoint_validator)  # ✅ UTILISÉ
):
    """
    🔧 Endpoint debug : Test des réponses d'erreur
    
    Endpoint de test pour valider le fonctionnement des ErrorResponse.
    Réservé aux administrateurs pour debugging.
    
    Args:
        error_type: Type d'erreur à simuler
        
    Returns:
        ErrorResponse selon le type demandé
    """
    # Vérification permissions  # ✅ UTILISÉ
    if not validation.get("is_admin", False):
        error_response = create_error_response(
            "INSUFFICIENT_PERMISSIONS",
            "Endpoint de debug réservé aux administrateurs",
            request_id=validation.get("request_id")
        )
        return JSONResponse(
            status_code=403,
            content=error_response.dict()
        )
    
    # Simulation différents types d'erreurs  # ✅ UTILISÉ
    if error_type == "validation":
        error_response = create_error_response(
            "VALIDATION_ERROR",
            "Erreur de validation simulée",
            details={"field": "test_field", "value": "invalid_value"},
            request_id=validation.get("request_id")
        )
        return JSONResponse(status_code=400, content=error_response.dict())
    
    elif error_type == "intent_detection":
        error_response = create_error_response(
            "INTENT_DETECTION_ERROR",
            "Erreur de détection d'intention simulée",
            details={"query": "test query", "attempted_methods": ["rules", "llm"]},
            request_id=validation.get("request_id")
        )
        return JSONResponse(status_code=422, content=error_response.dict())
    
    elif error_type == "system":
        error_response = create_error_response(
            "SYSTEM_ERROR",
            "Erreur système simulée",
            details={"component": "test_component", "exception_type": "TestException"},
            request_id=validation.get("request_id")
        )
        return JSONResponse(status_code=500, content=error_response.dict())
    
    else:
        error_response = create_error_response(
            "INVALID_ERROR_TYPE",
            f"Type d'erreur '{error_type}' non supporté",
            details={
                "supported_types": ["validation", "intent_detection", "system"],
                "received": error_type
            },
            request_id=validation.get("request_id")
        )
        return JSONResponse(status_code=400, content=error_response.dict())


# Export du router
__all__ = ["router"]