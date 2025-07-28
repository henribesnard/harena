"""
🌐 Routes API - Endpoints de Détection d'Intention

Routes FastAPI pour l'API de détection d'intention avec tous les endpoints
nécessaires : détection, métriques, santé, batch processing.
"""

import time
import logging
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from fastapi.responses import PlainTextResponse
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


@router.post("/detect-intent", response_model=IntentResponse)
async def detect_intent_endpoint(
    request: IntentRequest,
    intent_service: OptimizedIntentService = Depends(get_intent_service),
    user_context: Dict[str, Any] = Depends(get_user_context),
    metrics_collector: IntentMetricsCollector = Depends(get_metrics_manager),
    validation: Dict[str, Any] = Depends(public_endpoint_validator)
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
    
    try:
        # Ajout contexte utilisateur à la requête
        if user_context.get("user_id") and not request.user_id:
            request.user_id = user_context["user_id"]
        
        # Détection d'intention
        result = await intent_service.detect_intent(request)
        
        # Création réponse
        response = IntentResponse(**result)
        
        # Enregistrement métriques (background)
        record_intent_request(
            query=request.query,
            result=result,
            user_id=str(request.user_id) if request.user_id else None
        )
        
        # Logging pour audit
        logger.info(
            f"Intent detected: {response.intent} "
            f"(confidence: {response.confidence:.3f}, "
            f"method: {response.method_used}, "
            f"time: {response.processing_time_ms:.1f}ms) "
            f"for user {user_context.get('user_id', 'anonymous')}"
        )
        
        return response
        
    except ValidationError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Validation error",
                "message": str(e),
                "field": getattr(e, 'field_name', None)
            }
        )
    
    except IntentDetectionError as e:
        logger.error(f"Intent detection error: {e}")
        raise HTTPException(
            status_code=422,
            detail={
                "error": "Intent detection failed",
                "message": str(e),
                "query": request.query[:50] + "..." if len(request.query) > 50 else request.query
            }
        )
    
    except Exception as e:
        logger.error(f"Unexpected error in intent detection: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal server error",
                "message": "Une erreur inattendue s'est produite",
                "request_id": user_context.get("request_id")
            }
        )


@router.post("/batch-detect", response_model=BatchIntentResponse)
async def batch_detect_intent_endpoint(
    request: BatchIntentRequest,
    intent_service: OptimizedIntentService = Depends(get_intent_service),
    user_context: Dict[str, Any] = Depends(get_user_context),
    validation: Dict[str, Any] = Depends(public_endpoint_validator)
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
    
    try:
        # Validation taille batch
        if len(request.queries) > 100:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Batch trop grand",
                    "message": "Maximum 100 requêtes par batch",
                    "received": len(request.queries)
                }
            )
        
        logger.info(f"Starting batch processing: {len(request.queries)} queries")
        
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
                    error_response = IntentResponse(
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
                    intent_responses.append(error_response)
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
            "intent_distribution": {}
        }
        
        # Distribution méthodes et intentions
        for response in intent_responses:
            method = response.method_used
            batch_metrics["method_distribution"][method] = batch_metrics["method_distribution"].get(method, 0) + 1
            
            intent = response.intent
            batch_metrics["intent_distribution"][intent] = batch_metrics["intent_distribution"].get(intent, 0) + 1
        
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
            f"in {total_processing_time:.1f}ms"
        )
        
        return batch_response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch processing error: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Batch processing failed", 
                "message": str(e),
                "batch_size": len(request.queries) if request.queries else 0
            }
        )


@router.get("/health", response_model=HealthResponse)
async def health_check_endpoint(
    intent_service: OptimizedIntentService = Depends(get_intent_service),
    cache_manager: IntelligentMemoryCache = Depends(get_cache_manager)
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
            
            # Métriques de base
            total_requests=service_metrics.get("total_requests", 0),
            average_latency_ms=service_metrics.get("avg_latency_ms", 0.0),
            cache_hit_rate=cache_stats["cache_performance"]["hit_rate"],
            
            # Configuration
            deepseek_fallback_enabled=health_status["configuration"]["deepseek_enabled"],
            cache_enabled=health_status["configuration"]["cache_enabled"]
        )
        
        # Log statut santé
        if health_status["status"] != "healthy":
            logger.warning(f"Service health status: {health_status['status']}")
        
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
    metrics_collector: IntentMetricsCollector = Depends(get_metrics_manager),
    validation: Dict[str, Any] = Depends(admin_endpoint_validator)
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
        # Métriques service principal
        service_metrics = intent_service.get_metrics()
        
        if detailed:
            # Métriques détaillées (admin)
            comprehensive_report = metrics_collector.get_comprehensive_report()
            
            response = MetricsResponse(
                **service_metrics,
                **comprehensive_report.get("historical_analysis", {}),
                component_metrics=service_metrics.get("component_metrics", {}),
                detailed_analytics={
                    "intent_analytics": comprehensive_report.get("intent_analytics", {}),
                    "method_analytics": comprehensive_report.get("method_analytics", {}),
                    "user_analytics": comprehensive_report.get("user_analytics", {}),
                    "real_time_performance": comprehensive_report.get("real_time_performance", {})
                }
            )
        else:
            # Métriques basiques (public)
            response = MetricsResponse(**service_metrics)
        
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
    config_data: Dict[str, Any] = Depends(get_configuration)
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
            "deepseek_enabled": config_data["deepseek_enabled"]
        }
    }


@router.post("/cache/clear")
async def clear_cache_endpoint(
    cache_manager: IntelligentMemoryCache = Depends(get_cache_manager),
    validation: Dict[str, Any] = Depends(admin_endpoint_validator)
):
    """
    🗑️ Endpoint admin : Vider le cache
    
    Vide complètement le cache mémoire. Utile pour maintenance
    ou après mise à jour de modèles.
    
    Returns:
        Statistiques de nettoyage
    """
    try:
        entries_removed = cache_manager.clear()
        
        logger.info(f"Cache cleared by admin: {entries_removed} entries removed")
        
        return {
            "status": "success",
            "message": "Cache vidé avec succès",
            "entries_removed": entries_removed,
            "timestamp": time.time()
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
    validation: Dict[str, Any] = Depends(admin_endpoint_validator)
):
    """
    ⚡ Endpoint admin : Optimiser le cache
    
    Lance l'optimisation du cache : supprime entrées expirées
    et peu utilisées pour améliorer performance.
    
    Returns:
        Statistiques d'optimisation
    """
    try:
        optimization_stats = cache_manager.optimize_cache()
        
        logger.info(f"Cache optimized: {optimization_stats['total_removed']} entries removed")
        
        return {
            "status": "success",
            "message": "Cache optimisé avec succès",
            "optimization_stats": optimization_stats,
            "timestamp": time.time()
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
    metrics_collector: IntentMetricsCollector = Depends(get_metrics_manager),
    validation: Dict[str, Any] = Depends(admin_endpoint_validator)
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
        csv_content = metrics_collector.export_metrics_csv(hours)
        
        logger.info(f"Metrics exported: {hours}h of data")
        
        return PlainTextResponse(
            content=csv_content,
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=intent_metrics_{hours}h.csv"
            }
        )
        
    except Exception as e:
        logger.error(f"Metrics export error: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Metrics export failed",
                "message": str(e)
            }
        )


@router.post("/test-comprehensive")
async def comprehensive_test_endpoint(
    intent_service: OptimizedIntentService = Depends(get_intent_service),
    validation: Dict[str, Any] = Depends(admin_endpoint_validator)
):
    """
    🧪 Endpoint test : Test complet avec métriques
    
    Lance une série de tests complets pour valider le fonctionnement
    du service et générer métriques de performance.
    
    Returns:
        Résultats de tests avec statistiques détaillées
    """
    
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
    
    for query, expected in test_cases:
        request = IntentRequest(query=query, user_id="test_user")
        result = await intent_service.detect_intent(request)
        
        is_correct = result["intent"] == expected or (expected == "UNKNOWN" and result["confidence"] < 0.5)
        
        results.append({
            "query": query,
            "expected": expected,
            "detected": result["intent"],
            "confidence": result["confidence"],
            "correct": is_correct,
            "latency_ms": result["processing_time_ms"],
            "method": result["method_used"],
            "entities": result["entities"]
        })
    
    total_test_time = (time.time() - start_test) * 1000
    
    # Statistiques (logique exacte fichier original)
    correct_count = sum(1 for r in results if r["correct"])
    accuracy = correct_count / len(results)
    avg_latency = sum(r["latency_ms"] for r in results) / len(results)
    fast_responses = sum(1 for r in results if r["latency_ms"] < 50)  # Target latency
    
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
        "service_metrics": intent_service.get_metrics()
    }
    
    logger.info(
        f"Comprehensive test completed: {accuracy:.1%} accuracy, "
        f"{avg_latency:.1f}ms avg latency"
    )
    
    return test_response


# Export du router
__all__ = ["router"]