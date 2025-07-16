# conversation_service/main.py
import logging
import sys
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from datetime import datetime
import time

from .config.settings import settings
from .api.routes import router
from .clients.deepseek_client import deepseek_client
from .agents.intent_classifier import intent_classifier

# Configuration du logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger("conversation_service")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gestionnaire du cycle de vie de l'application.
    Initialise les ressources au démarrage et les libère à l'arrêt.
    """
    logger.info("🚀 Démarrage du Conversation Service...")
    
    try:
        # Validation de la configuration
        validation = settings.validate_configuration()
        if not validation["valid"]:
            logger.error(f"Configuration invalide: {validation['errors']}")
            raise RuntimeError(f"Configuration invalide: {validation['errors']}")
        
        if validation["warnings"]:
            logger.warning(f"Avertissements configuration: {validation['warnings']}")
        
        # Test de connexion DeepSeek
        logger.info("🔍 Test de connexion DeepSeek...")
        health_check = await deepseek_client.health_check()
        
        if health_check["status"] != "healthy":
            logger.error(f"❌ DeepSeek non disponible: {health_check.get('error', 'Unknown error')}")
            raise RuntimeError("DeepSeek API non disponible")
        
        logger.info(f"✅ DeepSeek connecté - Temps de réponse: {health_check['response_time']:.2f}s")
        
        # Initialisation de l'agent
        logger.info("🤖 Initialisation de l'agent de classification...")
        agent_metrics = intent_classifier.get_metrics()
        logger.info(f"✅ Agent initialisé - Seuil confiance: {settings.MIN_CONFIDENCE_THRESHOLD}")
        
        # Service prêt
        logger.info(f"🎉 Conversation Service démarré avec succès!")
        logger.info(f"📊 Configuration: Port {settings.PORT}, Debug: {settings.DEBUG}")
        logger.info(f"🎯 Modèle: {settings.DEEPSEEK_CHAT_MODEL}")
        
        yield
        
    except Exception as e:
        logger.error(f"❌ Erreur lors du démarrage: {str(e)}")
        raise
    
    finally:
        # Nettoyage lors de l'arrêt
        logger.info("🛑 Arrêt du Conversation Service...")
        
        # Affichage des métriques finales
        try:
            agent_metrics = intent_classifier.get_metrics()
            deepseek_metrics = deepseek_client.get_metrics()
            
            logger.info("📊 Métriques finales:")
            logger.info(f"  - Classifications totales: {agent_metrics['total_classifications']}")
            logger.info(f"  - Taux de succès: {agent_metrics['success_rate']:.1%}")
            logger.info(f"  - Confiance moyenne: {agent_metrics['avg_confidence']:.2f}")
            logger.info(f"  - Temps moyen: {agent_metrics['avg_processing_time']:.2f}s")
            logger.info(f"  - Cache hit rate: {deepseek_metrics['cache_hit_rate']:.1%}")
            
        except Exception as e:
            logger.error(f"Erreur lors de l'affichage des métriques: {str(e)}")
        
        logger.info("👋 Service arrêté proprement")

# Création de l'application FastAPI
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description=settings.API_DESCRIPTION,
    debug=settings.DEBUG,
    lifespan=lifespan
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En production, spécifier les domaines autorisés
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Middleware de logging des requêtes
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Middleware pour logger toutes les requêtes"""
    start_time = time.time()
    
    # Log de la requête entrante
    logger.info(f"📥 {request.method} {request.url.path} - {request.client.host}")
    
    try:
        response = await call_next(request)
        
        # Calcul du temps de traitement
        process_time = time.time() - start_time
        
        # Log de la réponse
        logger.info(f"📤 {request.method} {request.url.path} - {response.status_code} - {process_time:.3f}s")
        
        # Ajout du header de temps de traitement
        response.headers["X-Process-Time"] = str(process_time)
        
        return response
        
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(f"❌ {request.method} {request.url.path} - Error: {str(e)} - {process_time:.3f}s")
        raise

# Gestionnaire d'exceptions global
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Gestionnaire personnalisé pour les exceptions HTTP"""
    
    logger.warning(f"HTTP Exception: {exc.status_code} - {exc.detail} - {request.url.path}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "status_code": exc.status_code,
            "message": exc.detail,
            "path": str(request.url.path),
            "timestamp": datetime.utcnow().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Gestionnaire pour les exceptions non gérées"""
    
    logger.error(f"Unhandled exception: {str(exc)} - {request.url.path}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "status_code": 500,
            "message": "Internal server error" if not settings.DEBUG else str(exc),
            "path": str(request.url.path),
            "timestamp": datetime.utcnow().isoformat()
        }
    )

# Inclusion des routes
app.include_router(router)

# Endpoint racine
@app.get("/")
async def root():
    """Endpoint racine du service"""
    return {
        "service": settings.API_TITLE,
        "version": settings.API_VERSION,
        "status": "running",
        "timestamp": datetime.utcnow().isoformat(),
        "endpoints": {
            "health": "/health",
            "metrics": "/metrics",
            "conversation": "/api/v1/conversation/chat",
            "config": "/api/v1/conversation/config"
        }
    }

# Point d'entrée principal pour exécution directe
if __name__ == "__main__":
    logger.info(f"🚀 Démarrage du serveur sur {settings.HOST}:{settings.PORT}")
    
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
        access_log=True
    )