"""
🌐 Endpoints REST pour test architecture L0→L1→L2

Routes API avec validation Pydantic, formatage réponses standardisées
et gestion erreurs HTTP avec fallbacks gracieux.
"""

import asyncio
import logging
import time

from fastapi import APIRouter, HTTPException, Request, Depends
from fastapi.responses import JSONResponse

from conversation_service.models.conversation_models import ChatRequest, ChatResponse
from conversation_service.intent_detection.models import IntentResult
from conversation_service.utils import record_intent_performance, get_performance_summary
from conversation_service.utils.logging import log_intent_detection

logger = logging.getLogger(__name__)

# Router principal
router = APIRouter()

def get_intent_engine():
    """Dépendance pour obtenir Intent Detection Engine"""
    from conversation_service.main import get_intent_engine
    return get_intent_engine()

# ==========================================
# ENDPOINT PRINCIPAL CLASSIFICATION
# ==========================================

@router.post("/chat", response_model=ChatResponse)
async def classify_intent_endpoint(
    request: ChatRequest,
    intent_engine = Depends(get_intent_engine)
) -> ChatResponse:
    """
    Endpoint principal classification intentions avec pipeline L0→L1→L2
    
    Test architecture hybride:
    - L0: Patterns pré-calculés (<10ms)
    - L1: TinyBERT classification (15-30ms) 
    - L2: DeepSeek fallback (200-500ms)
    """
    start_time = time.time()
    request_id = f"req_{int(start_time * 1000)}"
    
    try:
        # Log requête entrante
        log_intent_detection(
            "request_received",
            user_id=request.user_id,
            message=request.message[:100],  # Tronquer pour logs
            request_id=request_id
        )
        
        # Classification intention via Intent Detection Engine
        intent_result: IntentResult = await intent_engine.detect_intent(
            user_query=request.message,
            user_id=request.user_id
        )
        
        # Calcul temps total
        total_time = (time.time() - start_time) * 1000
        
        # Métriques performance
        await record_intent_performance(
            level=intent_result.level.value,
            latency_ms=total_time,
            user_id=request.user_id,
            success=True
        )
        
        # Construction réponse avec métadonnées détaillées
        response = ChatResponse(
            intent=intent_result.intent_type,
            entities=intent_result.entities or {},
            confidence=intent_result.confidence.score,
            processing_metadata={
                "request_id": request_id,
                "level_used": intent_result.level.value,
                "processing_time_ms": round(total_time, 2),
                "cache_hit": intent_result.from_cache,
                "engine_latency_ms": round(intent_result.latency_ms, 2),
                "timestamp": int(time.time())
            }
        )
        
        # Log succès avec détails performance
        log_intent_detection(
            "classification_success",
            user_id=request.user_id,
            intent=intent_result.intent_type.value,
            level=intent_result.level.value,
            confidence=intent_result.confidence.score,
            latency_ms=total_time,
            cache_hit=intent_result.from_cache,
            request_id=request_id
        )
        
        return response
        
    except ValueError as e:
        # Erreur validation/parsing
        error_time = (time.time() - start_time) * 1000
        
        await record_intent_performance(
            level="error_validation",
            latency_ms=error_time,
            user_id=request.user_id,
            success=False
        )
        
        logger.warning(f"❌ Erreur validation requête {request_id}: {e}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": "validation_error",
                "message": str(e),
                "request_id": request_id
            }
        )
        
    except TimeoutError as e:
        # Timeout dépassé
        error_time = (time.time() - start_time) * 1000
        
        await record_intent_performance(
            level="error_timeout",
            latency_ms=error_time,
            user_id=request.user_id,
            success=False
        )
        
        logger.error(f"⏱️ Timeout requête {request_id}: {e}")
        
        # Fallback avec intention par défaut
        fallback_response = ChatResponse(
            intent="UNKNOWN",
            entities={},
            confidence=0.0,
            processing_metadata={
                "request_id": request_id,
                "level_used": "error_timeout",
                "processing_time_ms": round(error_time, 2),
                "cache_hit": False,
                "error": "timeout_exceeded",
                "timestamp": int(time.time())
            }
        )
        
        return fallback_response
        
    except Exception as e:
        # Erreur système générale
        error_time = (time.time() - start_time) * 1000
        
        await record_intent_performance(
            level="error_system",
            latency_ms=error_time,
            user_id=request.user_id,
            success=False
        )
        
        logger.error(f"💥 Erreur système requête {request_id}: {e}", exc_info=True)
        
        # Fallback gracieux avec intention par défaut
        fallback_response = ChatResponse(
            intent="UNKNOWN",
            entities={},
            confidence=0.0,
            processing_metadata={
                "request_id": request_id,
                "level_used": "error_fallback",
                "processing_time_ms": round(error_time, 2),
                "cache_hit": False,
                "error": "system_error",
                "timestamp": int(time.time())
            }
        )
        
        return fallback_response

# ==========================================
# HEALTH CHECK SPÉCIALISÉ
# ==========================================

@router.get("/health")
async def service_health():
    """Health check service + dépendances avec métriques performance"""
    try:
        from conversation_service.utils import simple_health_check
        
        # Health check général
        health_status = await simple_health_check()
        
        # Ajout métriques performance Intent Detection
        performance_summary = await get_performance_summary()
        health_status["performance"] = performance_summary
        
        # Status global basé sur performance
        if performance_summary.get("avg_latency_ms", 0) > 1000:
            health_status["status"] = "degraded"
            health_status["warning"] = "High latency detected"
        
        return health_status
        
    except Exception as e:
        logger.error(f"❌ Erreur health check: {e}")
        raise HTTPException(
            status_code=503,
            detail={"error": "health_check_failed", "message": str(e)}
        )

# ==========================================
# ENDPOINTS DEBUG ET MONITORING
# ==========================================

@router.get("/debug/performance")
async def get_performance_metrics():
    """Endpoint debug pour métriques performance détaillées"""
    try:
        performance_data = await get_performance_summary()
        
        # Ajout informations système
        performance_data["system"] = {
            "asyncio_tasks": len(asyncio.all_tasks()),
            "timestamp": int(time.time())
        }
        
        return performance_data
        
    except Exception as e:
        logger.error(f"❌ Erreur récupération métriques: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": "metrics_error", "message": str(e)}
        )

@router.post("/debug/test-levels")
async def test_detection_levels(
    request: ChatRequest,
    force_level: str = None,
    intent_engine = Depends(get_intent_engine)
):
    """
    Endpoint debug pour tester spécifiquement chaque niveau L0/L1/L2
    
    Args:
        force_level: "L0", "L1", ou "L2" pour forcer un niveau spécifique
    """
    if force_level and force_level not in ["L0", "L1", "L2"]:
        raise HTTPException(
            status_code=400,
            detail="force_level must be 'L0', 'L1', or 'L2'"
        )
    
    try:
        # Test avec niveau forcé (pour debug)
        if force_level:
            # Implémentation spécifique selon le niveau
            if force_level == "L0":
                result = await intent_engine._try_pattern_matching(request.message)
            elif force_level == "L1":
                result = await intent_engine._try_lightweight_classification(request.message)
            elif force_level == "L2":
                result = await intent_engine._try_llm_fallback(request.message, request.user_id)
        else:
            # Test normal
            result = await intent_engine.detect_intent(request.message, request.user_id)
        
        return {
            "requested_level": force_level,
            "actual_level": result.level.value,
            "intent": result.intent_type.value,
            "confidence": result.confidence.score,
            "latency_ms": result.latency_ms,
            "from_cache": result.from_cache
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur test niveau {force_level}: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": "level_test_failed", "message": str(e)}
        )

# ==========================================
# GESTION ERREURS GLOBALE
# ==========================================

@router.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handler personnalisé pour erreurs HTTP avec logs structurés"""
    
    log_intent_detection(
        "http_error",
        status_code=exc.status_code,
        detail=exc.detail,
        path=str(request.url.path),
        method=request.method
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "status_code": exc.status_code,
            "detail": exc.detail,
            "timestamp": int(time.time()),
            "path": str(request.url.path)
        }
    )