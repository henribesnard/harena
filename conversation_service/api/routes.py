"""
🌐 Endpoints REST pour conversation service - PHASE 1 L0 Pattern Matching

Routes API optimisées pour Pattern Matcher L0 avec latence <10ms
et réponses détaillées pour debug et monitoring.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List

from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.responses import JSONResponse

# Imports Phase 1
from conversation_service.models.conversation_models import (
    ChatRequest, ChatResponse, ServiceHealth, L0PerformanceMetrics,
    create_l0_success_response, create_l0_error_response, create_system_error_response,
    FinancialIntent, PatternMatch
)
from conversation_service.intent_detection.pattern_matcher import PatternMatcher, create_test_queries_phase1
from conversation_service.utils.logging import log_intent_detection, log_performance_metric

logger = logging.getLogger(__name__)

# Router principal Phase 1
router = APIRouter(
    tags=["conversation-phase1"],
    responses={
        404: {"description": "Not found"},
        500: {"description": "Internal server error"},
        503: {"description": "Service unavailable"}
    }
)

# ==========================================
# INSTANCE GLOBALE PATTERN MATCHER L0
# ==========================================

# Instance globale du Pattern Matcher L0
pattern_matcher_l0 = None

def initialize_pattern_matcher(matcher_instance: PatternMatcher):
    """
    ✅ Initialise le Pattern Matcher L0 dans les routes
    
    Cette fonction DOIT être appelée depuis main.py au démarrage
    pour lier le Pattern Matcher aux routes.
    """
    global pattern_matcher_l0
    pattern_matcher_l0 = matcher_instance
    logger.info("✅ Pattern Matcher L0 correctement lié aux routes")

def get_pattern_matcher_direct() -> PatternMatcher:
    """Accès direct au Pattern Matcher pour validation (non-async)"""
    return pattern_matcher_l0

# ==========================================
# DEPENDENCY INJECTION PHASE 1
# ==========================================

async def get_pattern_matcher() -> PatternMatcher:
    """
    ✅ Dependency injection pour Pattern Matcher L0
    
    Vérifie que le Pattern Matcher est disponible et retourne l'instance.
    Utilisée comme Depends() dans les endpoints.
    """
    if pattern_matcher_l0 is None:
        logger.error("❌ Pattern Matcher L0 non initialisé dans les routes")
        raise HTTPException(
            status_code=503,
            detail="Pattern Matcher L0 non disponible"
        )
    
    return pattern_matcher_l0

# ==========================================
# ENDPOINT PRINCIPAL CHAT - PHASE 1 OPTIMISÉ
# ==========================================

@router.post("/chat", response_model=ChatResponse)
async def chat_l0_endpoint(
    request: ChatRequest,
    matcher = Depends(get_pattern_matcher)
) -> ChatResponse:
    """
    🚀 Endpoint principal Phase 1 - Pattern Matching L0 seulement
    
    Optimisations Phase 1 :
    - Latence target <10ms (Pattern Matching pur)
    - Réponses détaillées avec pattern utilisé
    - Cache intelligent intégré
    - Métriques temps réel L0
    
    Target Phase 1: >85% requêtes en <10ms avec >90% précision
    """
    start_time = time.time()
    request_id = f"l0_req_{int(start_time * 1000)}_{request.user_id}"
    
    try:
        # ===== VALIDATION ULTRA-RAPIDE =====
        message_stripped = request.message.strip()
        if not message_stripped:
            return create_l0_error_response(
                request_id=request_id,
                error_type="validation_error",
                message="Message vide",
                processing_time_ms=0.0
            )
        
        # Log entrée
        logger.info(f"🔄 [{request_id}] L0 Processing: '{message_stripped[:40]}...'")
        
        # ===== PATTERN MATCHING L0 PRINCIPAL =====
        try:
            # Pattern matching avec timeout très court (L0 doit être ultra-rapide)
            pattern_match = await asyncio.wait_for(
                matcher.match_intent(message_stripped, str(request.user_id)),
                timeout=0.05  # 50ms max pour L0
            )
        except asyncio.TimeoutError:
            processing_time = (time.time() - start_time) * 1000
            logger.warning(f"⏰ [{request_id}] L0 Timeout après {processing_time:.2f}ms")
            
            return create_l0_error_response(
                request_id=request_id,
                error_type="l0_timeout",
                message="Pattern matching trop long",
                processing_time_ms=processing_time,
                pattern_analysis={"timeout_ms": 50, "message_length": len(message_stripped)}
            )
        
        # ===== TRAITEMENT RÉSULTAT L0 =====
        processing_time = (time.time() - start_time) * 1000
        
        if not pattern_match:
            # Aucun pattern trouvé ou confiance trop faible
            log_intent_detection(
                "l0_no_match",
                level="L0_PATTERN",
                latency_ms=processing_time,
                user_id=str(request.user_id),
                message_preview=message_stripped[:50]
            )
            
            return create_l0_error_response(
                request_id=request_id,
                error_type="l0_no_match",
                message="Aucun pattern L0 ne correspond",
                processing_time_ms=processing_time,
                pattern_analysis={
                    "patterns_tested": matcher.patterns.pattern_count,
                    "message_normalized": message_stripped,
                    "suggestions": [
                        "Utiliser des mots-clés financiers simples",
                        "Essayer : 'solde', 'virement', 'dépenses', 'carte'"
                    ]
                }
            )
        
        # ===== SUCCÈS L0 - Construction réponse détaillée =====
        
        # Détermination intention depuis pattern
        intent_type = None
        for metadata in matcher.patterns.pattern_metadata.values():
            if metadata["name"] == pattern_match.pattern_name:
                intent_type = metadata["intent"]
                break
        
        if not intent_type:
            intent_type = FinancialIntent.UNKNOWN
        
        # Conversion entités pour réponse
        entities_dict = {}
        for entity in pattern_match.entities:
            entities_dict[entity.type] = {
                "value": entity.value,
                "confidence": entity.confidence,
                "normalized": entity.normalized_value
            }
        
        # Construction réponse avec helper spécialisé
        response = create_l0_success_response(
            request_id=request_id,
            intent=intent_type.value,
            confidence=pattern_match.confidence,
            entities=entities_dict,
            processing_time_ms=processing_time,
            pattern_match=pattern_match,
            cache_hit=False,  # À implémenter avec cache détaillé
            confidence_reasoning=f"Pattern L0 '{pattern_match.pattern_name}' matched avec confiance {pattern_match.confidence:.2f}"
        )
        
        # ===== LOGGING ET MÉTRIQUES =====
        log_intent_detection(
            "l0_success",
            level="L0_PATTERN",
            intent=intent_type.value,
            confidence=pattern_match.confidence,
            latency_ms=processing_time,
            cache_hit=False,
            user_id=str(request.user_id),
            matched_text=pattern_match.matched_text,
            pattern_name=pattern_match.pattern_name
        )
        
        log_performance_metric(
            "l0_pattern_latency",
            processing_time,
            unit="ms",
            component="pattern_matcher",
            level="L0",
            threshold=10.0,
            threshold_exceeded=processing_time > 10.0
        )
        
        # Log succès avec détails
        logger.info(f"✅ [{request_id}] L0 Success: {intent_type.value} via '{pattern_match.pattern_name}' - {processing_time:.2f}ms")
        
        return response
        
    except HTTPException:
        # Re-raise des erreurs HTTP (503, etc.)
        raise
        
    except Exception as system_error:
        # Erreur système avec fallback L0
        error_time = (time.time() - start_time) * 1000
        
        logger.error(f"💥 [{request_id}] L0 System error ({error_time:.2f}ms): {system_error}")
        
        log_intent_detection(
            "l0_system_error",
            level="L0_PATTERN",
            latency_ms=error_time,
            user_id=str(request.user_id),
            error=str(system_error)
        )
        
        return create_system_error_response(
            request_id=request_id,
            error=system_error,
            processing_time_ms=error_time
        )

# ==========================================
# HEALTH CHECK PHASE 1
# ==========================================

@router.get("/health")
async def l0_health_check():
    """
    🏥 Health check spécialisé Phase 1 - Pattern Matcher L0
    
    Timeout: 100ms maximum (L0 ultra-rapide)
    Inclut: status patterns, performance L0, targets
    """
    start_time = time.time()
    
    try:
        # Vérification Pattern Matcher disponible
        if pattern_matcher_l0 is None:
            return {
                "status": "unhealthy",
                "phase": "L0_PATTERN_MATCHING",
                "timestamp": int(time.time()),
                "latency_ms": 0.0,
                "error": "Pattern Matcher L0 unavailable"
            }
        
        # Test fonctionnel rapide L0
        try:
            test_match = await asyncio.wait_for(
                pattern_matcher_l0.match_intent("solde", "health_check"),
                timeout=0.05  # 50ms max pour L0
            )
            
            if test_match:
                matcher_status = "functional"
                test_details = {
                    "pattern_name": test_match.pattern_name,
                    "confidence": test_match.confidence,
                    "pattern_type": test_match.pattern_type.value
                }
            else:
                matcher_status = "functional_no_match"
                test_details = {"message": "No pattern matched test query"}
                
        except asyncio.TimeoutError:
            matcher_status = "timeout"
            test_details = {"timeout_ms": 50}
        except Exception as test_error:
            matcher_status = "error"
            test_details = {"error": str(test_error)[:50]}
        
        # Calcul latence health check
        health_latency = (time.time() - start_time) * 1000
        
        # Métriques L0 actuelles
        l0_status = pattern_matcher_l0.get_status()
        l0_metrics = pattern_matcher_l0.get_l0_metrics()
        
        # Détermination status global
        if matcher_status == "functional" and health_latency < 100:
            overall_status = "healthy"
        elif matcher_status in ["functional", "functional_no_match"]:
            overall_status = "degraded"
        else:
            overall_status = "unhealthy"
        
        return {
            "status": overall_status,
            "phase": "L0_PATTERN_MATCHING",
            "timestamp": int(time.time()),
            "latency_ms": round(health_latency, 2),
            "pattern_matcher": {
                "status": matcher_status,
                "test_details": test_details,
                "patterns_loaded": l0_status["patterns_loaded"],
                "cache_size": l0_status["cache_size"]
            },
            "l0_performance": {
                "total_requests": l0_metrics.total_requests,
                "success_rate": round(l0_metrics.l0_success_rate, 3),
                "avg_latency_ms": round(l0_metrics.avg_l0_latency_ms, 2),
                "usage_percent": round(l0_metrics.target_l0_usage_percent, 1),
                "cache_hit_rate": round(l0_metrics.cache_hit_rate, 3)
            },
            "targets_status": {
                "latency_target": "< 10ms",
                "latency_met": l0_metrics.avg_l0_latency_ms < 10.0,
                "success_target": "> 85%",
                "success_met": l0_metrics.l0_success_rate >= 0.85,
                "usage_target": "> 80%",
                "usage_met": l0_metrics.target_l0_usage_percent >= 80.0
            }
        }
        
    except Exception as health_error:
        health_latency = (time.time() - start_time) * 1000
        logger.error(f"❌ L0 Health check error: {health_error}")
        
        return {
            "status": "unhealthy",
            "phase": "L0_PATTERN_MATCHING",
            "timestamp": int(time.time()),
            "latency_ms": round(health_latency, 2),
            "error": str(health_error)[:100]
        }

# ==========================================
# MÉTRIQUES DÉTAILLÉES PHASE 1
# ==========================================

@router.get("/metrics")
async def get_l0_metrics(
    matcher = Depends(get_pattern_matcher)
) -> Dict[str, Any]:
    """
    📊 Métriques détaillées Phase 1 - Pattern Matcher L0
    
    Inclut: performance L0, usage patterns, distribution confiance
    """
    try:
        # Métriques L0 complètes
        l0_metrics = matcher.get_l0_metrics()
        l0_status = matcher.get_status()
        pattern_usage = matcher.get_pattern_usage_report()
        
        return {
            "service": "conversation_service",
            "phase": "L0_PATTERN_MATCHING",
            "timestamp": int(time.time()),
            
            # Performance globale L0
            "l0_performance": {
                "total_requests": l0_metrics.total_requests,
                "successful_requests": l0_metrics.l0_successful_requests,
                "failed_requests": l0_metrics.l0_failed_requests,
                "success_rate": round(l0_metrics.l0_success_rate, 3),
                "avg_latency_ms": round(l0_metrics.avg_l0_latency_ms, 2),
                "usage_percent": round(l0_metrics.target_l0_usage_percent, 1)
            },
            
            # Cache performance
            "cache_performance": {
                "hit_rate": round(l0_metrics.cache_hit_rate, 3),
                "miss_rate": round(l0_metrics.cache_miss_rate, 3),
                "cache_size": l0_status["cache_size"],
                "cache_max_size": l0_status.get("cache_max_size", 1000)
            },
            
            # Patterns usage
            "pattern_analysis": {
                "total_patterns_available": pattern_usage["total_patterns"],
                "patterns_used": pattern_usage["used_patterns"],
                "patterns_unused": pattern_usage["unused_patterns"],
                "top_patterns": pattern_usage["top_patterns"],
                "pattern_performance": pattern_usage["pattern_performance"]
            },
            
            # Distribution confiance
            "confidence_distribution": l0_metrics.confidence_distribution,
            
            # Targets validation
            "targets_validation": {
                "latency_target_ms": 10.0,
                "latency_current_ms": round(l0_metrics.avg_l0_latency_ms, 2),
                "latency_met": l0_metrics.avg_l0_latency_ms < 10.0,
                
                "success_rate_target": 0.85,
                "success_rate_current": round(l0_metrics.l0_success_rate, 3),
                "success_rate_met": l0_metrics.l0_success_rate >= 0.85,
                
                "usage_target_percent": 80.0,
                "usage_current_percent": round(l0_metrics.target_l0_usage_percent, 1),
                "usage_met": l0_metrics.target_l0_usage_percent >= 80.0,
                
                "cache_target_rate": 0.15,
                "cache_current_rate": round(l0_metrics.cache_hit_rate, 3),
                "cache_met": l0_metrics.cache_hit_rate >= 0.15
            },
            
            # Système
            "system_info": {
                "asyncio_tasks": len(asyncio.all_tasks()),
                "phase": "L0_PATTERN_MATCHING",
                "ready_for_l1": all([
                    l0_metrics.avg_l0_latency_ms < 10.0,
                    l0_metrics.l0_success_rate >= 0.85,
                    l0_metrics.target_l0_usage_percent >= 80.0
                ])
            }
        }
        
    except Exception as metrics_error:
        logger.error(f"❌ Erreur métriques L0: {metrics_error}")
        return {
            "error": "l0_metrics_unavailable",
            "message": str(metrics_error),
            "phase": "L0_PATTERN_MATCHING",
            "timestamp": int(time.time())
        }

# ==========================================
# STATUS INFORMATIF PHASE 1
# ==========================================

@router.get("/status")
async def l0_service_status() -> Dict[str, Any]:
    """
    📋 Status détaillé Phase 1 pour monitoring externe
    """
    try:
        # Status Pattern Matcher
        matcher_available = pattern_matcher_l0 is not None
        
        if matcher_available:
            l0_status = pattern_matcher_l0.get_status()
            patterns_info = {
                "loaded": l0_status["patterns_loaded"],
                "cache_size": l0_status["cache_size"],
                "total_requests": l0_status["total_requests"],
                "success_rate": l0_status["success_rate"]
            }
        else:
            patterns_info = {"error": "Pattern Matcher not available"}
        
        return {
            "service": "conversation_service",
            "version": "1.0.0-phase1",
            "phase": "L0_PATTERN_MATCHING",
            "status": "running" if matcher_available else "degraded",
            
            # Architecture Phase 1
            "architecture": {
                "current_phase": "L0_PATTERN_MATCHING",
                "next_phase": "L1_LIGHTWEIGHT_CLASSIFIER",
                "l0_enabled": True,
                "l1_enabled": False,
                "l2_enabled": False
            },
            
            # Pattern Matcher info
            "pattern_matcher": patterns_info,
            
            # Endpoints Phase 1
            "endpoints": {
                "chat": "/api/v1/chat",
                "health": "/api/v1/health",
                "metrics": "/api/v1/metrics",
                "status": "/api/v1/status",
                "debug": "/api/v1/debug/test-patterns"
            },
            
            # Targets Phase 1
            "performance_targets": {
                "l0_latency_ms": "<10",
                "l0_success_rate": ">85%",
                "l0_usage_percent": ">80%",
                "l0_cache_hit_rate": ">15%"
            },
            
            # Next steps
            "roadmap": {
                "current": "Optimiser patterns L0 pour >85% succès",
                "next": "Ajouter L1 TinyBERT pour requêtes complexes",
                "future": "Intégrer L2 DeepSeek pour fallback intelligent"
            },
            
            "timestamp": int(time.time())
        }
        
    except Exception as status_error:
        logger.error(f"❌ Erreur status L0: {status_error}")
        return {
            "service": "conversation_service",
            "phase": "L0_PATTERN_MATCHING", 
            "status": "error",
            "error": str(status_error),
            "timestamp": int(time.time())
        }

# ==========================================
# ENDPOINTS DEBUG PHASE 1
# ==========================================

@router.post("/debug/test-patterns")
async def debug_test_patterns(
    request: ChatRequest,
    expected_intent: Optional[str] = Query(None, description="Intention attendue pour validation"),
    matcher = Depends(get_pattern_matcher)
) -> Dict[str, Any]:
    """
    🔧 Test debug patterns L0 avec analyse détaillée
    
    Permet de tester un pattern spécifique et voir tous les détails
    du matching pour debug et optimisation.
    """
    try:
        # Test pattern avec analyse complète
        test_result = await matcher.test_pattern(request.message, expected_intent)
        
        # Ajout informations requête
        test_result["request_info"] = {
            "user_id": request.user_id,
            "conversation_id": request.conversation_id,
            "debug_mode": request.debug_mode,
            "timestamp": int(time.time())
        }
        
        return test_result
        
    except Exception as debug_error:
        logger.error(f"❌ Erreur debug patterns: {debug_error}")
        return {
            "status": "error",
            "message": str(debug_error),
            "error_type": type(debug_error).__name__,
            "timestamp": int(time.time())
        }

@router.post("/debug/benchmark-l0")
async def debug_benchmark_l0(
    test_queries: Optional[List[str]] = None,
    matcher = Depends(get_pattern_matcher)
) -> Dict[str, Any]:
    """
    🏁 Benchmark performance L0 avec requêtes test
    
    Teste performance sur liste de requêtes et valide targets Phase 1.
    """
    try:
        # Utilise requêtes par défaut si non fournies
        if not test_queries:
            test_queries = create_test_queries_phase1()
        
        # Benchmark complet
        benchmark_results = await matcher.benchmark_l0_performance(test_queries)
        
        return {
            "benchmark_info": {
                "test_queries_count": len(test_queries),
                "phase": "L0_PATTERN_MATCHING",
                "timestamp": int(time.time())
            },
            "results": benchmark_results
        }
        
    except Exception as benchmark_error:
        logger.error(f"❌ Erreur benchmark L0: {benchmark_error}")
        return {
            "status": "error",
            "message": str(benchmark_error),
            "error_type": type(benchmark_error).__name__,
            "timestamp": int(time.time())
        }

@router.get("/debug/patterns-info")
async def debug_patterns_info(
    matcher = Depends(get_pattern_matcher)
) -> Dict[str, Any]:
    """
    📚 Informations détaillées sur tous les patterns chargés
    
    Pour debug et compréhension des patterns disponibles.
    """
    try:
        # Informations patterns
        patterns_metadata = {}
        
        for pattern_id, metadata in matcher.patterns.pattern_metadata.items():
            patterns_metadata[pattern_id] = {
                "name": metadata["name"],
                "intent": metadata["intent"].value,
                "confidence": metadata["confidence"],
                "pattern_type": metadata["pattern_type"].value,
                "priority": metadata.get("priority", 0),
                "regex": metadata["regex"][:100] + "..." if len(metadata["regex"]) > 100 else metadata["regex"],
                "entities_extractable": metadata.get("entities_extractable", [])
            }
        
        # Groupement par intention
        by_intent = {}
        for pattern_id, info in patterns_metadata.items():
            intent = info["intent"]
            if intent not in by_intent:
                by_intent[intent] = []
            by_intent[intent].append({
                "name": info["name"],
                "confidence": info["confidence"],
                "type": info["pattern_type"],
                "priority": info["priority"]
            })
        
        # Statistiques
        pattern_stats = matcher.get_pattern_usage_report()
        
        return {
            "summary": {
                "total_patterns": len(patterns_metadata),
                "by_intent_count": {intent: len(patterns) for intent, patterns in by_intent.items()},
                "pattern_types": list(set(info["pattern_type"] for info in patterns_metadata.values())),
                "confidence_range": {
                    "min": min(info["confidence"] for info in patterns_metadata.values()),
                    "max": max(info["confidence"] for info in patterns_metadata.values())
                }
            },
            "patterns_by_intent": by_intent,
            "usage_statistics": pattern_stats,
            "timestamp": int(time.time())
        }
        
    except Exception as patterns_error:
        logger.error(f"❌ Erreur patterns info: {patterns_error}")
        return {
            "error": "patterns_info_unavailable",
            "message": str(patterns_error),
            "timestamp": int(time.time())
        }

# ==========================================
# GESTION PATTERNS DYNAMIQUES
# ==========================================

@router.post("/admin/add-pattern")
async def admin_add_pattern(
    intent: str,
    pattern_data: Dict[str, Any],
    matcher = Depends(get_pattern_matcher)
) -> Dict[str, Any]:
    """
    ➕ Ajout pattern dynamique (admin seulement)
    
    Permet d'ajouter de nouveaux patterns en runtime pour test.
    """
    try:
        # Conversion intent string vers enum
        financial_intent = None
        for fi in FinancialIntent:
            if fi.value == intent:
                financial_intent = fi
                break
        
        if not financial_intent:
            return {
                "status": "error",
                "message": f"Intent invalide: {intent}",
                "valid_intents": [fi.value for fi in FinancialIntent]
            }
        
        # Ajout pattern
        success = matcher.add_dynamic_pattern(financial_intent, pattern_data)
        
        if success:
            return {
                "status": "success",
                "message": f"Pattern '{pattern_data.get('name', 'unknown')}' ajouté pour {intent}",
                "new_pattern_count": matcher.patterns.pattern_count,
                "timestamp": int(time.time())
            }
        else:
            return {
                "status": "error",
                "message": "Échec ajout pattern - Vérifiez regex et format",
                "timestamp": int(time.time())
            }
            
    except Exception as add_error:
        logger.error(f"❌ Erreur ajout pattern: {add_error}")
        return {
            "status": "error",
            "message": str(add_error),
            "timestamp": int(time.time())
        }
