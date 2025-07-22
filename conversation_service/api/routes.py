"""
🌐 Endpoints REST pour conversation service avec imports locaux

Routes API avec validation Pydantic, formatage réponses standardisées
et gestion erreurs HTTP avec imports locaux pour éviter circuits.
"""

import asyncio
import logging
import time
from typing import Dict, Any

from fastapi import APIRouter, HTTPException, Request, Depends
from fastapi.responses import JSONResponse

# ✅ Imports minimaux au niveau module - éviter les circuits
# Pas d'import de ChatRequest/ChatResponse au niveau module

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

@router.post("/chat")
async def classify_intent_endpoint(
    request_data: Dict[str, Any],  # ✅ Dict générique pour éviter import circulaire
    intent_engine = Depends(get_intent_engine)
) -> Dict[str, Any]:
    """
    Endpoint principal classification intentions avec pipeline L0→L1→L2
    
    Format attendu:
    {
        "message": "string",
        "user_id": int,
        "conversation_id": "string" (optionnel)
    }
    
    Test architecture hybride:
    - L0: Patterns pré-calculés (<10ms)
    - L1: TinyBERT classification (15-30ms) 
    - L2: DeepSeek fallback (200-500ms)
    """
    start_time = time.time()
    request_id = f"req_{int(start_time * 1000)}"
    
    try:
        # ✅ Import local des modèles pour éviter circuits
        from conversation_service.models.conversation_models import ChatRequest, ChatResponse
        from conversation_service.intent_detection.models import IntentResult
        from conversation_service.utils import record_intent_performance, get_performance_summary
        from conversation_service.utils.logging import log_intent_detection
        
        # Validation et conversion des données
        try:
            chat_request = ChatRequest(**request_data)
        except Exception as e:
            raise HTTPException(
                status_code=422,
                detail=f"Format de requête invalide: {e}"
            )
        
        # Log requête entrante
        log_intent_detection(
            "request_received",
            user_id=chat_request.user_id,
            message=chat_request.message[:100],  # Tronquer pour logs
            request_id=request_id
        )
        
        # Classification intention via Intent Detection Engine
        intent_result: IntentResult = await intent_engine.detect_intent(
            user_query=chat_request.message,
            user_id=chat_request.user_id
        )
        
        # Calcul temps total
        total_time = (time.time() - start_time) * 1000
        
        # Métriques performance
        await record_intent_performance(
            level=intent_result.level.value,
            latency_ms=total_time,
            user_id=chat_request.user_id,
            success=True
        )
        
        # ✅ Construction réponse directe en dict pour éviter problèmes sérialization
        response_data = {
            "request_id": request_id,
            "intent": intent_result.intent_type.value,
            "confidence": intent_result.confidence.score,
            "entities": intent_result.entities or {},
            "processing_metadata": {
                "request_id": request_id,
                "level_used": intent_result.level.value,
                "processing_time_ms": round(total_time, 2),
                "cache_hit": intent_result.from_cache,
                "engine_latency_ms": round(intent_result.latency_ms, 2),
                "timestamp": int(time.time())
            },
            "success": True
        }
        
        # Log succès avec détails performance
        log_intent_detection(
            "classification_success",
            user_id=chat_request.user_id,
            intent=intent_result.intent_type.value,
            level=intent_result.level.value,
            confidence=intent_result.confidence.score,
            latency_ms=total_time,
            cache_hit=intent_result.from_cache,
            request_id=request_id
        )
        
        return response_data
        
    except HTTPException:
        # Re-raise HTTPException as-is
        raise
        
    except ValueError as e:
        # Erreur validation/parsing
        error_time = (time.time() - start_time) * 1000
        
        try:
            from conversation_service.utils import record_intent_performance
            await record_intent_performance(
                level="error_validation",
                latency_ms=error_time,
                user_id=request_data.get("user_id", 0),
                success=False
            )
        except:
            pass  # Ignore si utils pas disponible
        
        logger.warning(f"❌ Erreur validation requête {request_id}: {e}")
        return {
            "request_id": request_id,
            "success": False,
            "error": "validation_error",
            "message": str(e),
            "processing_time_ms": round(error_time, 2)
        }
        
    except TimeoutError as e:
        # Timeout dépassé
        error_time = (time.time() - start_time) * 1000
        
        try:
            from conversation_service.utils import record_intent_performance
            await record_intent_performance(
                level="error_timeout",
                latency_ms=error_time,
                user_id=request_data.get("user_id", 0),
                success=False
            )
        except:
            pass
        
        logger.error(f"⏱️ Timeout requête {request_id}: {e}")
        
        # Fallback avec intention par défaut
        return {
            "request_id": request_id,
            "intent": "UNKNOWN",
            "confidence": 0.0,
            "entities": {},
            "processing_metadata": {
                "request_id": request_id,
                "level_used": "error_timeout",
                "processing_time_ms": round(error_time, 2),
                "cache_hit": False,
                "error": "timeout_exceeded",
                "timestamp": int(time.time())
            },
            "success": False,
            "error": "timeout_error"
        }
        
    except Exception as e:
        # Erreur système générale
        error_time = (time.time() - start_time) * 1000
        
        try:
            from conversation_service.utils import record_intent_performance
            await record_intent_performance(
                level="error_system",
                latency_ms=error_time,
                user_id=request_data.get("user_id", 0),
                success=False
            )
        except:
            pass
        
        logger.error(f"💥 Erreur système requête {request_id}: {e}", exc_info=True)
        
        # Fallback gracieux avec intention par défaut
        return {
            "request_id": request_id,
            "intent": "UNKNOWN",
            "confidence": 0.0,
            "entities": {},
            "processing_metadata": {
                "request_id": request_id,
                "level_used": "error_fallback",
                "processing_time_ms": round(error_time, 2),
                "cache_hit": False,
                "error": "system_error",
                "timestamp": int(time.time())
            },
            "success": False,
            "error": "system_error",
            "message": str(e)
        }

# ==========================================
# HEALTH CHECK SPÉCIALISÉ
# ==========================================

@router.get("/health")
async def service_health():
    """Health check service + dépendances avec métriques performance"""
    try:
        # Import local pour éviter circuits
        from conversation_service.utils import simple_health_check, get_performance_summary
        
        # Health check général
        health_status = await simple_health_check()
        
        # Ajout métriques performance Intent Detection
        try:
            performance_summary = await get_performance_summary()
            health_status["performance"] = performance_summary
            
            # Status global basé sur performance
            if performance_summary.get("avg_latency_ms", 0) > 1000:
                health_status["status"] = "degraded"
                health_status["warning"] = "High latency detected"
        except Exception as e:
            logger.warning(f"⚠️ Impossible de récupérer les métriques: {e}")
            health_status["performance"] = {"error": "metrics_unavailable"}
        
        return health_status
        
    except Exception as e:
        logger.error(f"❌ Erreur health check: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "service": "conversation_service",
            "timestamp": int(time.time())
        }

# ==========================================
# ENDPOINTS DEBUG ET MONITORING
# ==========================================

@router.get("/debug/performance")
async def get_performance_metrics():
    """Endpoint debug pour métriques performance détaillées"""
    try:
        # Import local pour éviter circuits
        from conversation_service.utils import get_performance_summary
        
        performance_data = await get_performance_summary()
        
        # Ajout informations système
        performance_data["system"] = {
            "asyncio_tasks": len(asyncio.all_tasks()),
            "timestamp": int(time.time())
        }
        
        return performance_data
        
    except Exception as e:
        logger.error(f"❌ Erreur récupération métriques: {e}")
        return {
            "error": "metrics_error",
            "message": str(e),
            "timestamp": int(time.time())
        }

@router.post("/debug/test-levels")
async def test_detection_levels(
    request_data: Dict[str, Any],  # ✅ Dict générique
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
        # Import local pour éviter circuits
        from conversation_service.models.conversation_models import ChatRequest
        
        # Validation données
        chat_request = ChatRequest(**request_data)
        
        # Test avec niveau forcé (pour debug)
        if force_level:
            # Implémentation spécifique selon le niveau
            if force_level == "L0":
                result = await intent_engine._try_pattern_matching(chat_request.message)
            elif force_level == "L1":
                result = await intent_engine._try_lightweight_classification(chat_request.message)
            elif force_level == "L2":
                result = await intent_engine._try_llm_fallback(chat_request.message, chat_request.user_id)
        else:
            # Test normal
            result = await intent_engine.detect_intent(chat_request.message, chat_request.user_id)
        
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
        return {
            "error": "level_test_failed",
            "message": str(e),
            "requested_level": force_level
        }

# ==========================================
# ENDPOINT MÉTRIQUES SIMPLE
# ==========================================

@router.get("/metrics")
async def get_simple_metrics(intent_engine = Depends(get_intent_engine)):
    """Métriques simplifiées du service"""
    try:
        # Import local pour éviter circuits
        from conversation_service.agents.intent_classifier import IntentClassifier
        
        # Obtenir métriques depuis l'engine si possible
        try:
            health_status = await intent_engine.get_health_status()
            return {
                "service": "conversation_service",
                "health": health_status,
                "timestamp": int(time.time())
            }
        except AttributeError:
            # Fallback si get_health_status n'existe pas
            return {
                "service": "conversation_service",
                "status": "running",
                "timestamp": int(time.time()),
                "note": "basic_metrics_only"
            }
        
    except Exception as e:
        logger.error(f"❌ Erreur métriques: {e}")
        return {
            "service": "conversation_service",
            "error": str(e),
            "timestamp": int(time.time())
        }

# ==========================================
# GESTION ERREURS GLOBALE - SUPPRIMÉE
# ==========================================

# Note: L'exception handler au niveau router ne fonctionne pas
# La gestion d'erreur se fait directement dans les endpoints