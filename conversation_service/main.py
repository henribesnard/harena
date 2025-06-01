"""
Module principal du service de conversation.

Ce module initialise et configure le service de conversation de Harena,
responsable de la détection d'intention et de la génération de réponses
contextuelles via DeepSeek.
"""
import logging
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from conversation_service.api.routes import router
from conversation_service.api.websocket import websocket_router
from conversation_service.core.deepseek_client import deepseek_client
from conversation_service.core.intent_detection import intent_detector
from conversation_service.storage.conversation_store import conversation_store
from conversation_service.utils.token_counter import token_counter
from config_service.config import settings

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(message)s',
)
logger = logging.getLogger("conversation_service")

# Variables globales
startup_time = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestionnaire du cycle de vie du service de conversation."""
    global startup_time
    
    # Initialisation
    logger.info("Démarrage du service de conversation")
    startup_time = time.time()
    
    # Vérification des configurations critiques
    if not settings.DEEPSEEK_API_KEY:
        logger.warning("DEEPSEEK_API_KEY non définie. La génération de réponses ne fonctionnera pas.")
    
    if not settings.SQLALCHEMY_DATABASE_URI:
        logger.warning("DATABASE_URI non définie. Le stockage des conversations ne fonctionnera pas.")
    
    # Initialisation des services
    try:
        # Client DeepSeek
        await deepseek_client.initialize()
        logger.info("Client DeepSeek initialisé avec succès")
        
        # Détecteur d'intention
        await intent_detector.initialize()
        logger.info("Détecteur d'intention initialisé avec succès")
        
        # Store de conversation
        await conversation_store.initialize()
        logger.info("Store de conversation initialisé avec succès")
        
        # Compteur de tokens
        token_counter.initialize()
        logger.info("Compteur de tokens initialisé avec succès")
        
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation des services: {e}")
    
    # Injecter les instances dans les modules
    import conversation_service.api.routes as routes
    import conversation_service.api.websocket as ws
    
    routes.deepseek_client = deepseek_client
    routes.intent_detector = intent_detector
    routes.conversation_store = conversation_store
    routes.token_counter = token_counter
    
    ws.deepseek_client = deepseek_client
    ws.intent_detector = intent_detector
    ws.conversation_store = conversation_store
    ws.token_counter = token_counter
    
    logger.info("Service de conversation prêt")
    
    yield  # L'application s'exécute ici
    
    # Nettoyage
    logger.info("Arrêt du service de conversation")
    
    if deepseek_client:
        await deepseek_client.close()
    if conversation_store:
        await conversation_store.close()
    
    logger.info("Service de conversation arrêté")


def create_app() -> FastAPI:
    """Crée et configure l'application FastAPI du service de conversation."""
    app = FastAPI(
        title="Harena Conversation Service",
        description="Service de conversation intelligente avec détection d'intention et génération de réponses",
        version="1.0.0",
        lifespan=lifespan
    )
    
    # Configuration CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS.split(","),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Middleware de logging des requêtes
    @app.middleware("http")
    async def log_requests(request, call_next):
        """Log les requêtes pour le monitoring."""
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        
        # Logger les requêtes lentes
        if process_time > 5.0:  # Plus de 5 secondes
            logger.warning(
                f"Requête lente détectée: {request.method} {request.url.path} "
                f"- {process_time:.2f}s"
            )
        
        response.headers["X-Process-Time"] = str(process_time)
        return response
    
    # Enregistrement des routeurs
    app.include_router(router, prefix="/api/v1", tags=["conversation"])
    app.include_router(websocket_router, tags=["websocket"])
    
    # Endpoint racine
    @app.get("/")
    def root():
        """Point d'entrée racine."""
        return {
            "service": "conversation_service",
            "version": "1.0.0",
            "status": "running",
            "description": "Service de conversation intelligente Harena"
        }
    
    # Endpoint de santé
    @app.get("/health")
    async def health_check():
        """Vérification de l'état de santé du service."""
        # Vérifier l'état des composants
        deepseek_ok = deepseek_client is not None and deepseek_client.is_initialized()
        database_ok = conversation_store is not None and await conversation_store.is_healthy()
        
        # État global
        if deepseek_ok and database_ok:
            status = "healthy"
        elif deepseek_ok or database_ok:
            status = "degraded"
        else:
            status = "unhealthy"
        
        # Calcul de l'uptime
        uptime = time.time() - startup_time if startup_time else 0
        
        # Stats des tokens si disponible
        token_stats = token_counter.get_stats() if token_counter else None
        
        return {
            "status": status,
            "deepseek_status": deepseek_ok,
            "database_status": database_ok,
            "version": "1.0.0",
            "uptime_seconds": uptime,
            "token_stats": token_stats,
            "timestamp": time.time()
        }
    
    return app


# Création de l'application
app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8005, reload=True)