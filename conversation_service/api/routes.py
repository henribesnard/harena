"""
🌐 Endpoints REST pour conversation service

Routes API standardisées avec validation Pydantic, gestion d'erreurs robuste
et pattern d'initialisation identique aux autres services (search_service).
"""

import asyncio
import logging
import time
import inspect
from typing import Dict, Any, Optional

from fastapi import APIRouter, HTTPException, Request, Depends, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Router principal avec configuration
router = APIRouter(
    tags=["conversation"],
    responses={
        404: {"description": "Not found"},
        500: {"description": "Internal server error"},
        503: {"description": "Service unavailable"}
    }
)

# ==========================================
# INSTANCE GLOBALE - PATTERN STANDARDISÉ
# ==========================================

# Instance globale du moteur (comme search_service)
intent_engine = None

def initialize_intent_engine(classifier):
    """
    Initialise le moteur d'intention avec le classifier fourni
    ✅ Pattern identique à search_service.initialize_search_engine()
    """
    global intent_engine
    intent_engine = classifier
    logger.info("✅ Intent engine initialized in routes")

# ==========================================
# MODÈLES PYDANTIC LOCAUX
# ==========================================

class ChatRequest(BaseModel):
    """Modèle de requête de chat standardisé"""
    message: str = Field(..., min_length=1, max_length=1000, description="Message utilisateur")
    user_id: int = Field(..., gt=0, description="ID utilisateur")
    conversation_id: Optional[str] = Field(None, description="ID conversation optionnel")

class ChatResponse(BaseModel):
    """Modèle de réponse de chat standardisé"""
    request_id: str
    intent: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    entities: Dict[str, Any] = Field(default_factory=dict)
    processing_metadata: Dict[str, Any] = Field(default_factory=dict)
    success: bool = True
    error: Optional[str] = None
    message: Optional[str] = None

class HealthResponse(BaseModel):
    """Modèle de réponse health check"""
    status: str = Field(..., description="healthy, degraded, ou unhealthy")
    service: str = "conversation_service"
    timestamp: int
    performance: Optional[Dict[str, Any]] = None
    warning: Optional[str] = None
    error: Optional[str] = None

# ==========================================
# FONCTION DE DÉPENDANCE STANDARDISÉE
# ==========================================

async def get_intent_engine():
    """
    Dependency pour obtenir le moteur d'intention initialisé
    ✅ Pattern identique à search_service.get_search_engine()
    """
    if not intent_engine:
        raise HTTPException(
            status_code=503, 
            detail="Service non disponible - Intent engine non initialisé"
        )
    
    return intent_engine

# ==========================================
# HELPER POUR DETECTION SIGNATURE
# ==========================================

async def call_classify_intent_adaptively(engine, message: str, user_id: str):
    """
    Appelle classify_intent avec détection automatique de la signature
    """
    try:
        # Inspecter la signature de la méthode
        method = getattr(engine, 'classify_intent')
        sig = inspect.signature(method)
        param_names = list(sig.parameters.keys())
        
        logger.info(f"🔍 Signature détectée: {param_names}")
        
        # Essayer différentes signatures courantes
        if 'query' in param_names and 'user_id' in param_names:
            # Signature avec query nommé
            return await engine.classify_intent(query=message, user_id=user_id)
        elif 'user_query' in param_names and 'user_id' in param_names:
            # Signature avec user_query nommé
            return await engine.classify_intent(user_query=message, user_id=user_id)
        elif 'message' in param_names and 'user_id' in param_names:
            # Signature avec message nommé
            return await engine.classify_intent(message=message, user_id=user_id)
        elif len(param_names) >= 3:  # self + 2 params
            # Signature positionnelle
            return await engine.classify_intent(message, user_id)
        elif len(param_names) >= 2:  # self + 1 param
            # Signature avec seulement le message
            return await engine.classify_intent(message)
        else:
            raise ValueError(f"Signature non supportée: {param_names}")
            
    except Exception as e:
        logger.error(f"❌ Erreur détection signature: {e}")
        # Fallback: essayer la signature la plus simple
        try:
            return await engine.classify_intent(message, user_id)
        except:
            return await engine.classify_intent(message)

# ==========================================
# ENDPOINT PRINCIPAL CLASSIFICATION
# ==========================================

@router.post("/chat", response_model=ChatResponse)
async def classify_intent_endpoint(
    request: ChatRequest,
    engine = Depends(get_intent_engine)
) -> ChatResponse:
    """
    Endpoint principal classification intentions avec pipeline L0→L1→L2
    
    **Architecture hybride:**
    - L0: Patterns pré-calculés (<10ms)
    - L1: TinyBERT classification (15-30ms) 
    - L2: DeepSeek fallback (200-500ms)
    
    **Format de réponse:**
    - intent: Type d'intention détectée
    - confidence: Score de confiance [0.0-1.0]
    - entities: Entités extraites du message
    - processing_metadata: Métadonnées de traitement
    """
    start_time = time.time()
    request_id = f"req_{int(start_time * 1000)}"
    
    try:
        # Import local des dépendances pour éviter circuits
        try:
            from conversation_service.utils.logging import log_intent_detection
            from conversation_service.utils import record_intent_performance
        except ImportError as e:
            logger.error(f"❌ Import error modules conversation: {e}")
            raise HTTPException(
                status_code=503,
                detail=f"Service modules unavailable: {str(e)}"
            )
        
        # Log requête entrante
        try:
            log_intent_detection(
                "request_received",
                user_id=request.user_id,
                message_preview=request.message[:50],
                request_id=request_id
            )
        except Exception as e:
            logger.warning(f"⚠️ Erreur logging entrée: {e}")
        
        # ✅ Classification intention avec détection automatique de signature
        try:
            logger.info(f"🔍 Appel classify_intent avec message: {request.message[:50]}")
            intent_result = await call_classify_intent_adaptively(
                engine, 
                request.message, 
                str(request.user_id)
            )
            logger.info(f"✅ Classification réussie: {intent_result}")
        except Exception as e:
            logger.error(f"❌ Erreur classification: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Intent classification failed: {str(e)}"
            )
        
        # Calcul temps total
        total_time = (time.time() - start_time) * 1000
        
        # Métriques performance (non-blocking)
        try:
            await record_intent_performance(
                level="L1_LIGHTWEIGHT",  # Fallback level 
                latency_ms=total_time,
                user_id=request.user_id,
                success=True
            )
        except Exception as e:
            logger.warning(f"⚠️ Erreur enregistrement métriques: {e}")
        
        # ✅ Adapter la réponse selon le format du classifier
        # Le classifier retourne probablement un format différent de IntentResult
        if hasattr(intent_result, 'intent_type'):
            # Format IntentResult standard
            intent_value = intent_result.intent_type.value if hasattr(intent_result.intent_type, 'value') else str(intent_result.intent_type)
            confidence_score = intent_result.confidence.score if hasattr(intent_result.confidence, 'score') else float(intent_result.confidence)
            entities = intent_result.entities or {}
            level_used = intent_result.level.value if hasattr(intent_result, 'level') and hasattr(intent_result.level, 'value') else "L1_LIGHTWEIGHT"
            cache_hit = getattr(intent_result, 'from_cache', False)
            engine_latency = getattr(intent_result, 'latency_ms', total_time)
        elif isinstance(intent_result, dict):
            # Format dict
            intent_value = str(intent_result.get('intent', 'UNKNOWN'))
            confidence_score = float(intent_result.get('confidence', 0.5))
            entities = intent_result.get('entities', {})
            level_used = intent_result.get('level', 'L1_LIGHTWEIGHT')
            cache_hit = intent_result.get('from_cache', False)
            engine_latency = intent_result.get('latency_ms', total_time)
        else:
            # Format simple string ou autre
            intent_value = str(intent_result)
            confidence_score = 0.5
            entities = {}
            level_used = "L1_LIGHTWEIGHT"
            cache_hit = False
            engine_latency = total_time
        
        # Construction réponse standardisée
        response = ChatResponse(
            request_id=request_id,
            intent=intent_value,
            confidence=confidence_score,
            entities=entities,
            processing_metadata={
                "request_id": request_id,
                "level_used": level_used,
                "processing_time_ms": round(total_time, 2),
                "cache_hit": cache_hit,
                "engine_latency_ms": round(engine_latency, 2),
                "timestamp": int(time.time())
            },
            success=True
        )
        
        # Log succès avec détails performance
        try:
            log_intent_detection(
                "classification_success",
                user_id=request.user_id,
                intent=intent_value,
                level=level_used,
                confidence=confidence_score,
                latency_ms=total_time,
                cache_hit=cache_hit,
                request_id=request_id
            )
        except Exception as e:
            logger.warning(f"⚠️ Erreur logging succès: {e}")
        
        return response
        
    except HTTPException:
        # Re-raise HTTPException as-is
        raise
        
    except Exception as e:
        # Erreur système générale avec fallback gracieux
        error_time = (time.time() - start_time) * 1000
        
        # Métriques d'erreur (non-blocking)
        try:
            from conversation_service.utils import record_intent_performance
            await record_intent_performance(
                level="error_system",
                latency_ms=error_time,
                user_id=request.user_id,
                success=False
            )
        except:
            pass
        
        logger.error(f"💥 Erreur système requête {request_id}: {e}", exc_info=True)
        
        # Fallback gracieux avec intention par défaut
        return ChatResponse(
            request_id=request_id,
            intent="UNKNOWN",
            confidence=0.0,
            entities={},
            processing_metadata={
                "request_id": request_id,
                "level_used": "error_fallback",
                "processing_time_ms": round(error_time, 2),
                "cache_hit": False,
                "error": "system_error",
                "timestamp": int(time.time())
            },
            success=False,
            error="system_error",
            message=str(e)
        )

# ==========================================
# HEALTH CHECK STANDARDISÉ
# ==========================================

@router.get("/health", response_model=HealthResponse)
async def conversation_health_check() -> HealthResponse:
    """
    Health check spécifique conversation service avec métriques performance
    
    **Status possibles:**
    - healthy: Service fonctionnel, performance normale
    - degraded: Service fonctionnel, performance dégradée
    - unhealthy: Service non fonctionnel
    """
    try:
        # Import local pour éviter circuits
        try:
            from conversation_service.utils import simple_health_check, get_performance_summary
        except ImportError as e:
            logger.error(f"❌ Import error utils: {e}")
            return HealthResponse(
                status="unhealthy",
                timestamp=int(time.time()),
                error=f"Utils unavailable: {str(e)}"
            )
        
        # Health check général
        try:
            health_status = await simple_health_check()
        except Exception as e:
            logger.error(f"❌ Health check failed: {e}")
            return HealthResponse(
                status="unhealthy",
                timestamp=int(time.time()),
                error=f"Health check failed: {str(e)}"
            )
        
        # Ajout métriques performance Intent Detection
        performance_data = None
        warning_msg = None
        
        try:
            performance_summary = await get_performance_summary()
            performance_data = performance_summary
            
            # Déterminer status basé sur performance
            avg_latency = performance_summary.get("avg_latency_ms", 0)
            if avg_latency > 1000:
                warning_msg = f"High latency detected: {avg_latency}ms"
                health_status["status"] = "degraded"
                
        except Exception as e:
            logger.warning(f"⚠️ Impossible de récupérer les métriques: {e}")
            performance_data = {"error": "metrics_unavailable"}
        
        return HealthResponse(
            status=health_status.get("status", "healthy"),
            timestamp=int(time.time()),
            performance=performance_data,
            warning=warning_msg
        )
        
    except Exception as e:
        logger.error(f"❌ Erreur health check: {e}")
        return HealthResponse(
            status="unhealthy",
            timestamp=int(time.time()),
            error=str(e)
        )

# ==========================================
# ENDPOINT MÉTRIQUES
# ==========================================

@router.get("/metrics")
async def get_conversation_metrics(
    engine = Depends(get_intent_engine)
) -> Dict[str, Any]:
    """
    Métriques détaillées du service de conversation
    
    **Inclut:**
    - Statistiques de performance
    - Métriques Intent Detection Engine  
    - Informations système
    """
    try:
        # Import local pour éviter circuits
        try:
            from conversation_service.utils import get_performance_summary
        except ImportError as e:
            return {
                "error": "utils_unavailable",
                "message": str(e),
                "timestamp": int(time.time())
            }
        
        # Métriques de performance
        try:
            performance_data = await get_performance_summary()
        except Exception as e:
            performance_data = {"error": f"performance_unavailable: {str(e)}"}
        
        # Métriques Intent Engine
        engine_metrics = {}
        try:
            if hasattr(engine, 'get_agent_metrics'):
                agent_metrics = engine.get_agent_metrics()
                engine_metrics = {
                    "status": "running",
                    "total_classifications": agent_metrics.get('total_classifications', 0),
                    "avg_confidence": agent_metrics.get('avg_confidence', 0.0),
                    "successful_classifications": agent_metrics.get('successful_classifications', 0)
                }
            else:
                engine_metrics = {"status": "running", "note": "basic_metrics_only"}
        except Exception as e:
            engine_metrics = {"error": f"engine_metrics_unavailable: {str(e)}"}
        
        # Informations système
        system_info = {
            "asyncio_tasks": len(asyncio.all_tasks()),
            "service": "conversation_service",
            "timestamp": int(time.time())
        }
        
        return {
            "performance": performance_data,
            "intent_engine": engine_metrics,
            "system": system_info
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur récupération métriques: {e}")
        return {
            "error": "metrics_error",
            "message": str(e),
            "timestamp": int(time.time())
        }

# ==========================================
# ENDPOINT STATUS
# ==========================================

@router.get("/status")
async def conversation_status() -> Dict[str, Any]:
    """
    Status détaillé du service de conversation
    
    **Informations fournies:**
    - Version et architecture du service
    - Endpoints disponibles
    - Configuration actuelle
    """
    return {
        "service": "conversation_service",
        "version": "1.0.0",
        "architecture": "mvp_intent_classifier",
        "model": "deepseek-chat",
        "status": "running",
        "endpoints": {
            "chat": "/api/v1/conversation/chat",
            "health": "/api/v1/conversation/health",
            "metrics": "/api/v1/conversation/metrics",
            "status": "/api/v1/conversation/status",
            "debug": "/api/v1/conversation/debug/test-levels"
        },
        "timestamp": int(time.time())
    }

# ==========================================
# ENDPOINT DEBUG
# ==========================================

@router.post("/debug/test-levels")
async def test_detection_levels(
    request: ChatRequest,
    force_level: Optional[str] = Query(None, regex="^(L0|L1|L2)$", description="Force un niveau spécifique"),
    engine = Depends(get_intent_engine)
) -> Dict[str, Any]:
    """
    Endpoint debug pour tester spécifiquement chaque niveau L0/L1/L2
    
    **Paramètres:**
    - force_level: "L0", "L1", ou "L2" pour forcer un niveau spécifique
    
    **Utilisation:**
    - Tests unitaires des différents niveaux
    - Debug des performances par niveau
    - Validation de la logique de fallback
    """
    try:
        start_time = time.time()
        
        # Test avec niveau forcé (pour debug)
        if force_level:
            logger.info(f"🔧 Test debug niveau {force_level} pour: {request.message[:50]}")
            
            # Pour l'instant, on utilise toujours la méthode standard
            # Les niveaux spécifiques seront implémentés plus tard
            result = await call_classify_intent_adaptively(engine, request.message, str(request.user_id))
        else:
            # Test normal sans forçage
            result = await call_classify_intent_adaptively(engine, request.message, str(request.user_id))
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "requested_level": force_level,
            "actual_level": "L1_LIGHTWEIGHT",  # Par défaut pour le moment
            "intent": str(result.get('intent', result) if isinstance(result, dict) else result),
            "confidence": float(result.get('confidence', 0.5) if isinstance(result, dict) else 0.5),
            "latency_ms": round(processing_time, 2),
            "from_cache": False,
            "message_preview": request.message[:50],
            "timestamp": int(time.time())
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur test niveau {force_level}: {e}")
        return {
            "error": "level_test_failed",
            "message": str(e),
            "requested_level": force_level,
            "timestamp": int(time.time())
        }