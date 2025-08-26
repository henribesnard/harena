"""
Point d'entrée principal conversation service - Phase 1
Compatible avec architecture ServiceLoader Harena
"""
import logging
import asyncio
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Configuration path pour imports
current_dir = Path(__file__).parent.absolute()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Imports conversation service
from conversation_service.clients.deepseek_client import DeepSeekClient
from conversation_service.core.cache_manager import CacheManager
from conversation_service.api.routes.conversation import router as conversation_router
from conversation_service.api.middleware.auth_middleware import JWTAuthMiddleware
from conversation_service.utils.metrics_collector import metrics_collector
from config_service.config import settings

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger("conversation_service")

class ConversationServiceLoader:
    """
    Service loader compatible avec architecture Harena existante
    Inspiré du pattern utilisé dans heroku_app.py
    """
    
    def __init__(self):
        self.deepseek_client: DeepSeekClient = None
        self.cache_manager: CacheManager = None
        self.service_healthy = False
        self.initialization_error = None
    
    async def initialize_conversation_service(self, app: FastAPI) -> bool:
        """
        Initialise conversation service selon pattern ServiceLoader Harena
        Compatible avec heroku_app.py
        """
        try:
            logger.info("🚀 Initialisation Conversation Service Phase 1...")
            
            # Vérification configuration
            if not settings.CONVERSATION_SERVICE_ENABLED:
                logger.info("❌ Conversation Service désactivé par configuration")
                return False
            
            # Validation configuration critique
            if not self._validate_configuration():
                return False
            
            # Initialisation clients externes
            success = await self._initialize_external_clients()
            if not success:
                return False
            
            # Health check initial complet
            health_ok = await self._comprehensive_health_check()
            if not health_ok:
                logger.error("❌ Health check initial échoué")
                return False
            
            # Injection services dans app state (pattern Harena)
            self._inject_services_into_app_state(app)
            
            # Configuration middleware et routes
            self._configure_app_middleware_and_routes(app)
            
            self.service_healthy = True
            logger.info("✅ Conversation Service Phase 1 initialisé avec succès")
            logger.info(f"📊 Fonctionnalités: Classification intentions (35 types supportés)")
            logger.info(f"🔐 Authentification: JWT obligatoire")
            logger.info(f"💾 Cache: Redis sémantique activé")
            
            return True
            
        except Exception as e:
            self.initialization_error = str(e)
            logger.error(f"❌ Erreur critique initialisation: {str(e)}", exc_info=True)
            return False
    
    def _validate_configuration(self) -> bool:
        """Validation configuration requise"""
        try:
            # DeepSeek API Key
            if not settings.DEEPSEEK_API_KEY:
                logger.error("❌ DEEPSEEK_API_KEY manquant")
                return False
            
            # JWT Secret
            if not settings.JWT_SECRET_KEY:
                logger.error("❌ JWT_SECRET_KEY manquant")
                return False
            
            if len(settings.JWT_SECRET_KEY) < 32:
                logger.error("❌ JWT_SECRET_KEY trop court (minimum 32 caractères)")
                return False
            
            logger.info("✅ Configuration validée")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur validation configuration: {str(e)}")
            return False
    
    async def _initialize_external_clients(self) -> bool:
        """Initialisation clients externes avec retry"""
        try:
            # Initialisation DeepSeek client
            logger.info("🤖 Initialisation client DeepSeek...")
            self.deepseek_client = DeepSeekClient()
            await self.deepseek_client.initialize()
            
            # Test connexion DeepSeek
            deepseek_healthy = await self.deepseek_client.health_check()
            if not deepseek_healthy:
                logger.error("❌ DeepSeek API non accessible")
                return False
            
            logger.info("✅ DeepSeek client opérationnel")
            
            # Initialisation Cache Manager
            logger.info("💾 Initialisation cache Redis...")
            self.cache_manager = CacheManager()
            await self.cache_manager.initialize()
            
            # Test connexion Redis (non bloquant)
            cache_healthy = await self.cache_manager.health_check()
            if cache_healthy:
                logger.info("✅ Redis cache opérationnel")
            else:
                logger.warning("⚠️ Redis indisponible - cache désactivé")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur initialisation clients: {str(e)}")
            return False
    
    async def _comprehensive_health_check(self) -> bool:
        """Health check complet tous services"""
        try:
            health_status = {
                "deepseek": False,
                "cache": False
            }
            
            # Test DeepSeek
            if self.deepseek_client:
                health_status["deepseek"] = await self.deepseek_client.health_check()
            
            # Test Cache (non critique)
            if self.cache_manager:
                health_status["cache"] = await self.cache_manager.health_check()
            
            # DeepSeek obligatoire, cache optionnel
            service_operational = health_status["deepseek"]
            
            logger.info(f"🏥 Health check: DeepSeek={health_status['deepseek']}, Cache={health_status['cache']}")
            
            return service_operational
            
        except Exception as e:
            logger.error(f"❌ Erreur health check: {str(e)}")
            return False
    
    def _inject_services_into_app_state(self, app: FastAPI) -> None:
        """Injection services dans app state (pattern Harena)"""
        app.state.conversation_service = self
        app.state.deepseek_client = self.deepseek_client
        app.state.cache_manager = self.cache_manager
        
        # Métriques globales
        app.state.metrics_collector = metrics_collector
        
        logger.info("✅ Services injectés dans app state")
    
    def _configure_app_middleware_and_routes(self, app: FastAPI) -> None:
        """Configuration middleware et routes"""
        
        # Middleware authentification JWT
        app.add_middleware(JWTAuthMiddleware)
        logger.info("🔐 Middleware JWT configuré")
        
        # Routes conversation
        app.include_router(conversation_router, prefix="/api/v1")
        logger.info("📡 Routes conversation configurées")
    
    async def cleanup(self) -> None:
        """Nettoyage ressources"""
        try:
            if self.deepseek_client:
                await self.deepseek_client.close()
            
            if self.cache_manager:
                await self.cache_manager.close()
            
            logger.info("✅ Ressources conversation service nettoyées")
            
        except Exception as e:
            logger.error(f"❌ Erreur nettoyage: {str(e)}")

# Instance globale service loader
conversation_service_loader = ConversationServiceLoader()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestionnaire cycle de vie application"""
    # Startup
    logger.info("🚀 Démarrage application conversation service")
    
    try:
        # Initialisation service
        initialization_success = await conversation_service_loader.initialize_conversation_service(app)
        
        if not initialization_success:
            logger.error("❌ Échec initialisation - service non disponible")
            # App démarre quand même pour exposer health check
        
        yield
        
    except Exception as e:
        logger.error(f"❌ Erreur critique startup: {str(e)}", exc_info=True)
        yield
    
    finally:
        # Shutdown
        logger.info("🔄 Arrêt application conversation service")
        await conversation_service_loader.cleanup()

# Application FastAPI
app = FastAPI(
    title="Harena Conversation Service",
    description="Service IA conversationnelle financière - Phase 1: Classification d'intentions",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.ENVIRONMENT != "production" else None,
    redoc_url="/redoc" if settings.ENVIRONMENT != "production" else None
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.ENVIRONMENT != "production" else ["https://app.harena.fr", "https://api.harena.fr"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Health check global (compatible pattern Harena)
@app.get("/health")
async def global_health():
    """Health check global compatible avec monitoring Harena"""
    try:
        if conversation_service_loader.service_healthy:
            health_metrics = metrics_collector.get_health_metrics()
            
            return JSONResponse(
                status_code=200,
                content={
                    "status": "healthy",
                    "service": "conversation_service",
                    "phase": 1,
                    "timestamp": health_metrics.get("timestamp", ""),
                    "version": "1.0.0",
                    "uptime_seconds": health_metrics.get("uptime_seconds", 0),
                    "components": {
                        "deepseek_api": "operational",
                        "redis_cache": "operational" if conversation_service_loader.cache_manager else "disabled",
                        "intent_classification": "operational"
                    }
                }
            )
        else:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unhealthy",
                    "service": "conversation_service",
                    "phase": 1,
                    "error": conversation_service_loader.initialization_error or "Service non initialisé",
                    "components": {
                        "deepseek_api": "unknown",
                        "redis_cache": "unknown", 
                        "intent_classification": "unavailable"
                    }
                }
            )
            
    except Exception as e:
        logger.error(f"❌ Erreur health check global: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "service": "conversation_service", 
                "error": str(e)
            }
        )

# Endpoint métrique compatible Prometheus
@app.get("/metrics")
async def metrics_endpoint():
    """Métriques Prometheus pour monitoring"""
    try:
        if not conversation_service_loader.service_healthy:
            raise HTTPException(status_code=503, detail="Service non opérationnel")
        
        metrics_data = metrics_collector.get_all_metrics()
        
        # Format compatible monitoring Harena
        return {
            "service": "conversation_service",
            "timestamp": metrics_data["timestamp"],
            "metrics": metrics_data,
            "labels": {
                "service": "conversation_service",
                "phase": "1",
                "version": "1.0.0"
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur export métriques: {str(e)}")
        raise HTTPException(status_code=500, detail="Erreur métriques")

# Handler erreurs globales
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Handler erreurs globales avec logging"""
    logger.error(f"❌ Erreur non gérée: {str(exc)}", exc_info=True)
    
    metrics_collector.increment_counter("conversation.errors.unhandled")
    
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Erreur interne du service",
            "service": "conversation_service",
            "request_path": str(request.url.path) if hasattr(request, 'url') else "unknown"
        }
    )

# Point d'entrée pour uvicorn
if __name__ == "__main__":
    import uvicorn
    
    logger.info("🚀 Démarrage direct conversation service")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.ENVIRONMENT == "development",
        log_level="info"
    )