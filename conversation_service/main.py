"""
🚀 Point d'entrée principal - FastAPI avec Intent Detection Engine

Configuration application FastAPI avec initialisation Intent Detection Engine,
middleware basique, health checks et gestion lifecycle.
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

from config_service.config import settings
from conversation_service.intent_detection.engine import IntentDetectionEngine
from conversation_service.utils.logging import setup_logging
from conversation_service.utils import record_intent_performance
from conversation_service.api.routes import router

# Configuration logging structuré
setup_logging()
logger = logging.getLogger(__name__)

# Instance globale Intent Detection Engine
intent_engine: IntentDetectionEngine = None

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Gestion lifecycle (startup/shutdown) avec initialisation services"""
    
    # ==========================================
    # STARTUP - Initialisation services
    # ==========================================
    logger.info("🚀 Démarrage Conversation Service...")
    
    try:
        # Initialisation Intent Detection Engine
        logger.info("⚙️ Initialisation Intent Detection Engine...")
        global intent_engine
        intent_engine = IntentDetectionEngine()
        await intent_engine.initialize()
        
        # Validation configuration au démarrage
        validation_result = settings.validate_configuration()
        if not validation_result["valid"]:
            logger.error(f"❌ Configuration invalide: {validation_result['errors']}")
            raise RuntimeError(f"Configuration invalide: {validation_result['errors']}")
        
        if validation_result["warnings"]:
            logger.warning(f"⚠️ Avertissements configuration: {validation_result['warnings']}")
        
        # Test santé services critiques
        from conversation_service.utils import simple_health_check
        health_status = await simple_health_check()
        if not health_status.get("healthy", False):
            logger.error(f"❌ Services critiques indisponibles: {health_status}")
            # Note: On continue quand même pour permettre le démarrage en mode dégradé
        
        # Métriques de démarrage
        await record_intent_performance("startup", 0, "system", success=True)
        
        logger.info("✅ Conversation Service initialisé avec succès")
        
        # ✅ Utiliser les variables d'environnement directement au lieu de settings.DEBUG
        debug_mode = os.environ.get("CONVERSATION_SERVICE_DEBUG", "false").lower() == "true"
        port = getattr(settings, "CONVERSATION_SERVICE_PORT", 8001)
        timeout = getattr(settings, "REQUEST_TIMEOUT", 30)
        confidence = getattr(settings, "MIN_CONFIDENCE_THRESHOLD", 0.7)
        redis_enabled = getattr(settings, "REDIS_CACHE_ENABLED", True)
        
        logger.info(f"🔧 Mode: {'DEBUG' if debug_mode else 'PRODUCTION'}")
        logger.info(f"🌐 Port: {port}")
        logger.info(f"⏱️ Timeout: {timeout}s")
        logger.info(f"🎯 Confidence: {confidence}")
        logger.info(f"💾 Cache Redis: {'Activé' if redis_enabled else 'Désactivé'}")
        
    except Exception as e:
        logger.error(f"❌ Erreur initialisation: {e}")
        raise
    
    yield  # Application running
    
    # ==========================================  
    # SHUTDOWN - Nettoyage ressources
    # ==========================================
    logger.info("🛑 Arrêt Conversation Service...")
    
    try:
        # Arrêt Intent Detection Engine
        if intent_engine:
            await intent_engine.shutdown()
            logger.info("✅ Intent Detection Engine arrêté")
        
        # Métriques de fermeture
        await record_intent_performance("shutdown", 0, "system", success=True)
        
        logger.info("✅ Conversation Service arrêté proprement")
        
    except Exception as e:
        logger.error(f"❌ Erreur arrêt: {e}")

# ✅ Configuration FastAPI avec gestion des attributs manquants
try:
    # Essayer d'utiliser les attributs standards
    api_title = getattr(settings, "API_TITLE", "Conversation Service")
    api_version = getattr(settings, "API_VERSION", "1.0.0")
    api_description = getattr(settings, "API_DESCRIPTION", "Service de conversation avec IA")
    debug_mode = os.environ.get("CONVERSATION_SERVICE_DEBUG", "false").lower() == "true"
except AttributeError as e:
    # Fallback avec valeurs par défaut
    logger.warning(f"⚠️ Attribut manquant dans settings: {e}")
    api_title = "Conversation Service"
    api_version = "1.0.0"
    api_description = "Service de conversation avec IA"
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

# Routes principales
app.include_router(router, prefix="/api/v1")

# ==========================================
# HEALTH CHECKS GLOBAUX
# ==========================================

@app.get("/health")
async def global_health_check():
    """Health check global service + dépendances"""
    try:
        from conversation_service.utils import simple_health_check
        health_status = await simple_health_check()
        
        # Ajout informations Intent Detection Engine
        if intent_engine:
            engine_health = await intent_engine.get_health_status()
            health_status.update({"intent_engine": engine_health})
        
        return health_status
        
    except Exception as e:
        logger.error(f"❌ Erreur health check: {e}")
        raise HTTPException(status_code=503, detail=f"Health check failed: {e}")

@app.get("/health/ready")
async def readiness_check():
    """Readiness check pour déploiement"""
    if not intent_engine:
        raise HTTPException(status_code=503, detail="Intent Detection Engine not initialized")
    
    return {"status": "ready", "timestamp": asyncio.get_event_loop().time()}

@app.get("/health/live")
async def liveness_check():
    """Liveness check basique"""
    return {"status": "alive", "service": "conversation_service"}

@app.get("/")
async def root():
    """Endpoint racine avec informations service"""
    return {
        "service": api_title,
        "version": api_version,
        "description": api_description,
        "status": "running",
        "endpoints": {
            "chat": "/api/v1/chat",
            "health": "/health",
            "docs": "/docs"
        }
    }

# ==========================================
# FONCTION HELPER POUR ACCÈS À L'ENGINE
# ==========================================

def get_intent_engine() -> IntentDetectionEngine:
    """
    Retourne l'instance globale Intent Detection Engine
    ✅ Version corrigée sans référence à settings.DEBUG
    """
    if intent_engine is None:
        raise RuntimeError("Intent Detection Engine not initialized")
    return intent_engine

# Export pour utilisation dans les routes
__all__ = ["app", "get_intent_engine"]

if __name__ == "__main__":
    import uvicorn
    
    # ✅ Configuration avec gestion d'erreurs pour les attributs manquants
    try:
        host = getattr(settings, "HOST", "localhost")
        port = getattr(settings, "CONVERSATION_SERVICE_PORT", 8001)
        log_level = getattr(settings, "CONVERSATION_SERVICE_LOG_LEVEL", "INFO")
        debug_mode = os.environ.get("CONVERSATION_SERVICE_DEBUG", "false").lower() == "true"
    except AttributeError as e:
        logger.warning(f"⚠️ Configuration par défaut utilisée: {e}")
        host = "localhost"
        port = 8001
        log_level = "INFO"
        debug_mode = False
    
    # Configuration optimisée pour production
    uvicorn.run(
        "conversation_service.main:app",
        host=host,
        port=port,
        log_level=log_level.lower(),
        access_log=debug_mode,
        reload=debug_mode,
        workers=1,  # Single worker pour développement
        loop="asyncio"
    )