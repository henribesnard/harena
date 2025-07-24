"""
🚀 Point d'entrée principal - FastAPI avec Pattern Matcher L0 Phase 1

Configuration application FastAPI ultra-simplifiée avec SEULEMENT Pattern Matcher L0
pour performances <10ms sur 85%+ des requêtes financières.

Version Phase 1 : L0 Pattern Matching seulement
"""

import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

# Imports Phase 1 - Pattern Matcher seulement
from conversation_service.intent_detection.pattern_matcher import PatternMatcher, validate_l0_phase1_performance
from conversation_service.models.conversation_models import ServiceHealth, L0PerformanceMetrics
from conversation_service.utils.logging import setup_logging, log_intent_detection

# Configuration logging structuré
setup_logging()
logger = logging.getLogger(__name__)

# Instance globale Pattern Matcher L0
pattern_matcher: PatternMatcher = None
_service_initialized = False
_service_start_time = None

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Gestion lifecycle Phase 1 - Pattern Matcher L0 seulement"""
    
    global pattern_matcher, _service_initialized, _service_start_time
    
    # ==========================================
    # STARTUP - Initialisation Pattern Matcher L0
    # ==========================================
    logger.info("🚀 Démarrage Conversation Service - PHASE 1 (L0 Pattern Matching)")
    _service_start_time = asyncio.get_event_loop().time()
    
    try:
        # ===== ÉTAPE 1: Pattern Matcher L0 =====
        logger.info("⚡ Initialisation Pattern Matcher L0...")
        pattern_matcher = PatternMatcher(cache_manager=None)  # Pas de cache externe en Phase 1
        await pattern_matcher.initialize()
        logger.info("✅ Pattern Matcher L0 initialisé")
        
        # ===== ÉTAPE 2: Liaison avec routes =====
        logger.info("🔗 Liaison Pattern Matcher avec les routes...")
        from conversation_service.api.routes import initialize_pattern_matcher
        initialize_pattern_matcher(pattern_matcher)
        logger.info("✅ Pattern Matcher lié aux routes avec succès")
        
        # ===== ÉTAPE 3: Validation liaison =====
        logger.info("🧪 Validation liaison pattern_matcher↔routes...")
        from conversation_service.api.routes import get_pattern_matcher_direct
        
        try:
            linked_matcher = get_pattern_matcher_direct()
            if linked_matcher is None:
                raise RuntimeError("Pattern Matcher non accessible dans routes")
                
            # Test fonctionnel ultra-rapide L0
            test_match = await linked_matcher.match_intent("solde", "startup_test")
            if test_match:
                logger.info(f"✅ Test L0 réussi - Pattern: {test_match.pattern_name}, Confiance: {test_match.confidence:.2f}")
            else:
                logger.warning("⚠️ Test L0 sans match - Patterns peuvent nécessiter optimisation")
                
        except Exception as validation_error:
            logger.error(f"❌ Validation liaison échouée: {validation_error}")
            raise
        
        # ===== ÉTAPE 4: Validation performance Phase 1 =====
        logger.info("🎯 Validation performance Phase 1...")
        try:
            validation_results = await validate_l0_phase1_performance(pattern_matcher)
            
            if validation_results["overall_status"] == "READY_FOR_L1":
                logger.info("🎉 Performance Phase 1 validée - Prêt pour L1")
            else:
                logger.warning(f"⚠️ Performance Phase 1 nécessite optimisation: {validation_results['recommendations']}")
                # Continue quand même en mode dégradé
        except Exception as validation_error:
            logger.warning(f"⚠️ Validation performance échouée: {validation_error}")
        
        # ===== FINALISATION =====
        _service_initialized = True
        app.state.pattern_matcher = pattern_matcher
        app.state.service_initialized = True
        app.state.service_phase = "L0_PATTERN_MATCHING"
        
        # Log métriques de démarrage
        log_intent_detection(
            "service_startup_complete",
            level="L0_PATTERN",
            message=f"Service Phase 1 démarré avec {pattern_matcher.patterns.pattern_count} patterns"
        )
        
        # Informations de configuration Phase 1
        debug_mode = os.environ.get("CONVERSATION_SERVICE_DEBUG", "false").lower() == "true"
        
        logger.info(f"🔧 Phase: L0_PATTERN_MATCHING")
        logger.info(f"🔧 Mode: {'DEBUG' if debug_mode else 'PRODUCTION'}")
        logger.info(f"📊 Patterns chargés: {pattern_matcher.patterns.pattern_count}")
        logger.info("🎉 Conversation Service Phase 1 complètement initialisé!")
        
        yield  # Application running
        
    except Exception as startup_error:
        logger.error(f"❌ Erreur initialisation Phase 1: {startup_error}")
        _service_initialized = False
        raise
    
    # ==========================================  
    # SHUTDOWN - Nettoyage ressources
    # ==========================================
    logger.info("🛑 Arrêt Conversation Service Phase 1...")
    
    try:
        # Arrêt Pattern Matcher
        if pattern_matcher:
            try:
                await pattern_matcher.shutdown()
                logger.info("✅ Pattern Matcher L0 arrêté")
            except Exception as shutdown_error:
                logger.warning(f"⚠️ Erreur arrêt pattern matcher: {shutdown_error}")
        
        # Log métriques finales
        if pattern_matcher:
            final_status = pattern_matcher.get_status()
            logger.info(f"📊 Métriques finales Phase 1:")
            logger.info(f"   - Requêtes traitées: {final_status['total_requests']}")
            logger.info(f"   - Taux succès L0: {final_status['success_rate']:.1%}")
            logger.info(f"   - Latence moyenne: {final_status['avg_latency_ms']:.1f}ms")
        
        logger.info("✅ Conversation Service Phase 1 arrêté proprement")
        
    except Exception as shutdown_error:
        logger.error(f"❌ Erreur arrêt: {shutdown_error}")

# ===== Configuration FastAPI Phase 1 =====
try:
    api_title = "Conversation Service - Phase 1"
    api_version = "1.0.0-phase1"
    api_description = "Service de conversation financière - Phase 1 : L0 Pattern Matching (<10ms)"
    debug_mode = os.environ.get("CONVERSATION_SERVICE_DEBUG", "false").lower() == "true"
except Exception as config_error:
    logger.warning(f"⚠️ Configuration par défaut utilisée: {config_error}")
    api_title = "Conversation Service - Phase 1"
    api_version = "1.0.0-phase1"
    api_description = "Service de conversation financière - Phase 1"
    debug_mode = False

app = FastAPI(
    title=api_title,
    version=api_version,
    description=api_description,
    debug=debug_mode,
    lifespan=lifespan
)

# ==========================================
# MIDDLEWARE CONFIGURATION
# ==========================================

# CORS pour développement
if debug_mode:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Security middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"] if debug_mode else ["localhost", "127.0.0.1"]
)

# ==========================================
# ROUTES REGISTRATION
# ==========================================

# Import et registration des routes Phase 1
from conversation_service.api.routes import router
app.include_router(router, prefix="/api/v1", tags=["conversation-phase1"])

# ==========================================
# HEALTH CHECKS PHASE 1
# ==========================================

@app.get("/health")
async def global_health_check():
    """Health check global Phase 1 - Pattern Matcher L0"""
    try:
        if not _service_initialized:
            raise HTTPException(status_code=503, detail="Service Phase 1 initializing")
        
        if not pattern_matcher:
            raise HTTPException(status_code=503, detail="Pattern Matcher L0 unavailable")
        
        # Test fonctionnel rapide L0
        health_status = {
            "status": "healthy",
            "service": "conversation_service",
            "phase": "L0_PATTERN_MATCHING",
            "version": "1.0.0-phase1",
            "timestamp": int(asyncio.get_event_loop().time()),
            "uptime_seconds": int(asyncio.get_event_loop().time() - _service_start_time) if _service_start_time else 0
        }
        
        # Test Pattern Matcher avec timeout
        try:
            test_match = await asyncio.wait_for(
                pattern_matcher.match_intent("solde", "health_check"),
                timeout=0.1  # Très rapide pour L0
            )
            
            if test_match:
                health_status["pattern_matcher"] = {
                    "status": "functional",
                    "test_pattern": test_match.pattern_name,
                    "test_confidence": test_match.confidence
                }
            else:
                health_status["pattern_matcher"] = {
                    "status": "functional_no_match",
                    "message": "No pattern matched test query"
                }
                
        except asyncio.TimeoutError:
            health_status["pattern_matcher"] = {"status": "timeout"}
            health_status["status"] = "degraded"
        except Exception as test_error:
            health_status["pattern_matcher"] = {"status": "error", "error": str(test_error)[:50]}
            health_status["status"] = "degraded"
        
        # Métriques L0
        try:
            l0_status = pattern_matcher.get_status()
            health_status["l0_metrics"] = {
                "patterns_loaded": l0_status["patterns_loaded"],
                "total_requests": l0_status["total_requests"],
                "success_rate": l0_status["success_rate"],
                "avg_latency_ms": l0_status["avg_latency_ms"],
                "targets_met": l0_status["targets_met"]
            }
        except Exception as metrics_error:
            health_status["l0_metrics"] = {"error": str(metrics_error)[:50]}
        
        return health_status
        
    except HTTPException:
        raise
    except Exception as health_error:
        logger.error(f"❌ Health check error: {health_error}")
        raise HTTPException(status_code=503, detail=f"Health check failed: {str(health_error)}")

@app.get("/health/ready")
async def readiness_check():
    """Readiness check Phase 1"""
    if not _service_initialized:
        raise HTTPException(status_code=503, detail="Service Phase 1 not ready")
    
    if not pattern_matcher:
        raise HTTPException(status_code=503, detail="Pattern Matcher L0 not ready")
    
    return {
        "status": "ready",
        "phase": "L0_PATTERN_MATCHING",
        "timestamp": int(asyncio.get_event_loop().time())
    }

@app.get("/health/live")
async def liveness_check():
    """Liveness check ultra-basique"""
    return {
        "status": "alive",
        "service": "conversation_service",
        "phase": "L0_PATTERN_MATCHING"
    }

@app.get("/")
async def root():
    """Endpoint racine Phase 1"""
    return {
        "service": api_title,
        "version": api_version,
        "description": api_description,
        "phase": "L0_PATTERN_MATCHING",
        "status": "running" if _service_initialized else "initializing",
        "patterns_loaded": pattern_matcher.patterns.pattern_count if pattern_matcher else 0,
        "endpoints": {
            "chat": "/api/v1/chat",
            "health": "/health",
            "metrics": "/api/v1/metrics",
            "docs": "/docs"
        },
        "targets": {
            "latency_ms": "<10",
            "success_rate": ">85%",
            "l0_usage": ">80%"
        }
    }

# ==========================================
# FONCTIONS D'ACCÈS GLOBAL PHASE 1
# ==========================================

def get_pattern_matcher() -> PatternMatcher:
    """Retourne l'instance globale Pattern Matcher L0"""
    if pattern_matcher is None:
        raise RuntimeError("Pattern Matcher L0 not initialized")
    return pattern_matcher

def is_service_ready() -> bool:
    """Vérifie si le service Phase 1 est complètement initialisé"""
    return _service_initialized and pattern_matcher is not None

def get_service_phase() -> str:
    """Retourne la phase actuelle du service"""
    return "L0_PATTERN_MATCHING"

# Export pour utilisation dans les routes et local_app.py
__all__ = ["app", "get_pattern_matcher", "is_service_ready", "get_service_phase"]

# ==========================================
# MAIN POUR DÉVELOPPEMENT LOCAL
# ==========================================

if __name__ == "__main__":
    import uvicorn
    
    # Configuration serveur Phase 1
    host = "localhost"
    port = 8001
    log_level = "INFO"
    debug_mode = os.environ.get("CONVERSATION_SERVICE_DEBUG", "false").lower() == "true"
    
    logger.info(f"🚀 Démarrage serveur Phase 1 sur {host}:{port}")
    
    # Configuration uvicorn optimisée Phase 1
    uvicorn.run(
        "conversation_service.main:app",
        host=host,
        port=port,
        log_level=log_level.lower(),
        access_log=debug_mode,
        reload=debug_mode,
        workers=1,  # Single worker
        loop="asyncio",
        timeout_keep_alive=30,
        timeout_graceful_shutdown=5
    )
