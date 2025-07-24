"""
🚀 Point d'entrée principal - FastAPI avec Pattern Matcher L0 Phase 1

Configuration application FastAPI ultra-simplifiée avec SEULEMENT Pattern Matcher L0
pour performances <10ms sur 85%+ des requêtes financières.

Version Phase 1 : L0 Pattern Matching seulement

✅ CORRECTIONS APPLIQUÉES:
- Suppression des imports inutilisés (ServiceHealth, L0PerformanceMetrics)
- Utilisation des helpers du models pour la cohérence
- Nettoyage des variables globales non utilisées
- Optimisation des health checks avec les nouveaux modèles
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

# ✅ IMPORTS CORRIGÉS - Suppression des classes non utilisées
from conversation_service.intent_detection.pattern_matcher import PatternMatcher, validate_l0_phase1_performance
from conversation_service.utils.logging import setup_logging, log_intent_detection

# Configuration logging structuré
setup_logging()
logger = logging.getLogger(__name__)

# ==========================================
# VARIABLES GLOBALES PHASE 1 - SIMPLIFIÉES
# ==========================================

# Instance globale Pattern Matcher L0
pattern_matcher: PatternMatcher = None
_service_initialized = False
_service_start_time = None

# ==========================================
# LIFECYCLE MANAGEMENT
# ==========================================

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

# ==========================================
# CONFIGURATION FASTAPI PHASE 1
# ==========================================

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
# HEALTH CHECKS PHASE 1 - ✅ SIMPLIFIÉS
# ==========================================

@app.get("/health")
async def global_health_check():
    """
    🏥 Health check global Phase 1 - Pattern Matcher L0
    
    ✅ SIMPLIFIÉ: Utilise les status du pattern matcher directement
    sans duplicata avec les routes spécialisées
    """
    try:
        if not _service_initialized:
            raise HTTPException(status_code=503, detail="Service Phase 1 initializing")
        
        if not pattern_matcher:
            raise HTTPException(status_code=503, detail="Pattern Matcher L0 unavailable")
        
        # Status basique pour health check global
        uptime = int(asyncio.get_event_loop().time() - _service_start_time) if _service_start_time else 0
        
        health_status = {
            "status": "healthy",
            "service": "conversation_service",
            "phase": "L0_PATTERN_MATCHING",
            "version": api_version,
            "timestamp": int(asyncio.get_event_loop().time()),
            "uptime_seconds": uptime
        }
        
        # Test Pattern Matcher ultra-rapide
        try:
            test_match = await asyncio.wait_for(
                pattern_matcher.match_intent("solde", "global_health_check"),
                timeout=0.05  # 50ms max
            )
            
            if test_match:
                health_status["pattern_matcher"] = {
                    "status": "functional",
                    "test_success": True,
                    "test_pattern": test_match.pattern_name
                }
            else:
                health_status["pattern_matcher"] = {
                    "status": "functional_no_match",
                    "test_success": False,
                    "message": "No pattern matched test query"
                }
                
        except asyncio.TimeoutError:
            health_status["pattern_matcher"] = {
                "status": "timeout", 
                "test_success": False,
                "timeout_ms": 50
            }
            health_status["status"] = "degraded"
        except Exception as test_error:
            health_status["pattern_matcher"] = {
                "status": "error", 
                "test_success": False,
                "error": str(test_error)[:50]
            }
            health_status["status"] = "degraded"
        
        # ✅ Status simple du pattern matcher
        try:
            l0_status = pattern_matcher.get_status()
            health_status["quick_metrics"] = {
                "patterns_loaded": l0_status["patterns_loaded"],
                "total_requests": l0_status["total_requests"],
                "success_rate": round(l0_status.get("success_rate", 0.0), 2)
            }
        except Exception as metrics_error:
            health_status["quick_metrics"] = {
                "error": str(metrics_error)[:50]
            }
        
        # Note: Pour des métriques détaillées, utiliser /api/v1/health
        health_status["detailed_health"] = "/api/v1/health"
        health_status["detailed_metrics"] = "/api/v1/metrics"
        
        return health_status
        
    except HTTPException:
        raise
    except Exception as health_error:
        logger.error(f"❌ Global health check error: {health_error}")
        raise HTTPException(status_code=503, detail=f"Health check failed: {str(health_error)}")

@app.get("/health/ready")
async def readiness_check():
    """✅ Readiness check Phase 1 - Ultra-simple"""
    if not _service_initialized:
        raise HTTPException(status_code=503, detail="Service Phase 1 not ready")
    
    if not pattern_matcher:
        raise HTTPException(status_code=503, detail="Pattern Matcher L0 not ready")
    
    return {
        "status": "ready",
        "phase": "L0_PATTERN_MATCHING",
        "service": "conversation_service",
        "timestamp": int(asyncio.get_event_loop().time())
    }

@app.get("/health/live")
async def liveness_check():
    """✅ Liveness check ultra-basique - Toujours disponible"""
    return {
        "status": "alive",
        "service": "conversation_service", 
        "phase": "L0_PATTERN_MATCHING",
        "timestamp": int(asyncio.get_event_loop().time())
    }

# ==========================================
# ENDPOINT RACINE INFORMATIF
# ==========================================

@app.get("/")
async def root():
    """
    📋 Endpoint racine Phase 1 - Informations service
    
    ✅ AMÉLIORÉ: Plus d'informations utiles pour découverte API
    """
    try:
        # Informations de base
        base_info = {
            "service": api_title,
            "version": api_version,
            "description": api_description,
            "phase": "L0_PATTERN_MATCHING",
            "status": "running" if _service_initialized else "initializing",
            "timestamp": int(asyncio.get_event_loop().time())
        }
        
        # Informations pattern matcher si disponible
        if pattern_matcher:
            try:
                matcher_status = pattern_matcher.get_status()
                base_info["pattern_matcher"] = {
                    "patterns_loaded": matcher_status["patterns_loaded"],
                    "total_requests": matcher_status["total_requests"],
                    "cache_size": matcher_status.get("cache_size", 0)
                }
            except Exception as status_error:
                base_info["pattern_matcher"] = {
                    "status": "error",
                    "error": str(status_error)[:50]
                }
        else:
            base_info["pattern_matcher"] = {
                "status": "not_initialized"
            }
        
        # Endpoints disponibles
        base_info["endpoints"] = {
            "main": {
                "chat": "/api/v1/chat",
                "health": "/api/v1/health", 
                "metrics": "/api/v1/metrics",
                "status": "/api/v1/status"
            },
            "debug": {
                "test_patterns": "/api/v1/debug/test-patterns",
                "benchmark": "/api/v1/debug/benchmark-l0",
                "patterns_info": "/api/v1/debug/patterns-info"
            },
            "validation": {
                "phase1_ready": "/api/v1/validate-phase1"
            },
            "system": {
                "global_health": "/health",
                "readiness": "/health/ready",
                "liveness": "/health/live",
                "docs": "/docs",
                "openapi": "/openapi.json"
            }
        }
        
        # Targets Phase 1
        base_info["performance_targets"] = {
            "latency_ms": "<10",
            "success_rate": ">85%",
            "l0_usage": ">80%",
            "cache_hit_rate": ">15%"
        }
        
        # Informations Phase
        base_info["phase_info"] = {
            "current": "L0_PATTERN_MATCHING",
            "description": "Pattern matching ultra-rapide pour requêtes financières courantes",
            "next_phase": "L1_LIGHTWEIGHT_CLASSIFIER",
            "capabilities": [
                "Consultation soldes instantanée",
                "Virements simples",
                "Gestion carte basique", 
                "Analyse dépenses courantes"
            ],
            "limitations": [
                "Pas de requêtes complexes multi-étapes",
                "Pas d'analyse contextuelle avancée",
                "Pas de conversations multi-tours"
            ]
        }
        
        return base_info
        
    except Exception as root_error:
        logger.error(f"❌ Root endpoint error: {root_error}")
        # Retourne info minimale en cas d'erreur
        return {
            "service": "conversation_service",
            "phase": "L0_PATTERN_MATCHING", 
            "status": "error",
            "error": str(root_error)[:100],
            "basic_endpoints": {
                "health": "/health",
                "docs": "/docs"
            }
        }

# ==========================================
# FONCTIONS D'ACCÈS GLOBAL PHASE 1 - ✅ UTILISÉES
# ==========================================

def get_pattern_matcher() -> PatternMatcher:
    """✅ Retourne l'instance globale Pattern Matcher L0"""
    if pattern_matcher is None:
        raise RuntimeError("Pattern Matcher L0 not initialized")
    return pattern_matcher

def is_service_ready() -> bool:
    """✅ Vérifie si le service Phase 1 est complètement initialisé"""
    return _service_initialized and pattern_matcher is not None

def get_service_phase() -> str:
    """✅ Retourne la phase actuelle du service"""
    return "L0_PATTERN_MATCHING"

def get_service_uptime() -> int:
    """✅ Retourne l'uptime du service en secondes"""
    if not _service_start_time:
        return 0
    return int(asyncio.get_event_loop().time() - _service_start_time)

def get_service_info() -> dict:
    """✅ Informations complètes du service pour monitoring"""
    return {
        "service": api_title,
        "version": api_version,
        "phase": "L0_PATTERN_MATCHING",
        "initialized": _service_initialized,
        "uptime_seconds": get_service_uptime(),
        "pattern_matcher_available": pattern_matcher is not None,
        "debug_mode": debug_mode
    }

# ==========================================
# EXPORTS POUR UTILISATION EXTERNE
# ==========================================

__all__ = [
    "app", 
    "get_pattern_matcher", 
    "is_service_ready", 
    "get_service_phase",
    "get_service_uptime",
    "get_service_info"
]

# ==========================================
# MAIN POUR DÉVELOPPEMENT LOCAL
# ==========================================

if __name__ == "__main__":
    import uvicorn
    
    # Configuration serveur Phase 1
    host = os.environ.get("CONVERSATION_SERVICE_HOST", "localhost")
    port = int(os.environ.get("CONVERSATION_SERVICE_PORT", "8001"))
    log_level = os.environ.get("CONVERSATION_SERVICE_LOG_LEVEL", "INFO")
    workers = int(os.environ.get("CONVERSATION_SERVICE_WORKERS", "1"))
    
    logger.info(f"🚀 Démarrage serveur Phase 1")
    logger.info(f"   Host: {host}")
    logger.info(f"   Port: {port}")
    logger.info(f"   Debug: {debug_mode}")
    logger.info(f"   Log Level: {log_level}")
    logger.info(f"   Workers: {workers}")
    
    # Configuration uvicorn optimisée Phase 1
    uvicorn_config = {
        "app": "conversation_service.main:app",
        "host": host,
        "port": port,
        "log_level": log_level.lower(),
        "access_log": debug_mode,
        "reload": debug_mode,
        "workers": workers,
        "loop": "asyncio",
        "timeout_keep_alive": 30,
        "timeout_graceful_shutdown": 10
    }
    
    # Ajout SSL si configuré (production)
    ssl_keyfile = os.environ.get("CONVERSATION_SERVICE_SSL_KEYFILE")
    ssl_certfile = os.environ.get("CONVERSATION_SERVICE_SSL_CERTFILE")
    
    if ssl_keyfile and ssl_certfile:
        uvicorn_config.update({
            "ssl_keyfile": ssl_keyfile,
            "ssl_certfile": ssl_certfile
        })
        logger.info("🔒 SSL configuré")
    
    try:
        uvicorn.run(**uvicorn_config)
    except KeyboardInterrupt:
        logger.info("🛑 Arrêt serveur demandé par utilisateur")
    except Exception as server_error:
        logger.error(f"❌ Erreur serveur: {server_error}")
        sys.exit(1)