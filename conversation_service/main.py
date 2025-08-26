"""
Point d'entrée principal conversation service optimisé - Phase 1 JSON Output
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
from datetime import datetime, timezone

# Configuration path pour imports
current_dir = Path(__file__).parent.absolute()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Imports conversation service
from conversation_service.clients.deepseek_client import DeepSeekClient, DeepSeekError
from conversation_service.core.cache_manager import CacheManager
from conversation_service.api.routes.conversation import router as conversation_router
from conversation_service.api.middleware.auth_middleware import JWTAuthMiddleware
from conversation_service.utils.metrics_collector import metrics_collector
from config_service.config import settings

# Configuration logging optimisée
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout,
    force=True  # Force reconfiguration
)
logger = logging.getLogger("conversation_service")


class ConversationServiceLoader:
    """
    Service loader optimisé compatible avec architecture Harena existante
    Inspiré du pattern utilisé dans heroku_app.py avec améliorations
    """
    
    def __init__(self):
        self.deepseek_client: DeepSeekClient = None
        self.cache_manager: CacheManager = None
        self.service_healthy = False
        self.initialization_error = None
        self.service_start_time = datetime.now(timezone.utc)
        
        # Configuration service
        self.service_config = {
            "phase": 1,
            "version": "1.0.0",
            "features": ["intent_classification", "json_output", "cache", "auth", "metrics"],
            "json_output_enforced": True,
            "deepseek_model": getattr(settings, 'DEEPSEEK_CHAT_MODEL', 'deepseek-chat')
        }
        
        logger.info("ConversationServiceLoader initialisé")
    
    async def initialize_conversation_service(self, app: FastAPI) -> bool:
        """
        Initialise conversation service selon pattern ServiceLoader Harena optimisé
        
        Returns:
            bool: True si initialisation réussie, False sinon
        """
        try:
            logger.info("🚀 Initialisation Conversation Service Phase 1 JSON Output...")
            
            # Vérification configuration service
            if not getattr(settings, 'CONVERSATION_SERVICE_ENABLED', True):
                logger.info("⚠️ Conversation Service désactivé par configuration")
                return False
            
            # Validation configuration critique avec diagnostic
            validation_success = await self._validate_comprehensive_configuration()
            if not validation_success:
                return False
            
            # Initialisation clients externes avec retry
            clients_success = await self._initialize_external_clients_with_retry()
            if not clients_success:
                return False
            
            # Validation JSON Output fonctionnelle
            json_validation = await self._validate_json_output_functionality()
            if not json_validation:
                logger.error("❌ Validation JSON Output échouée")
                return False
            
            # Health check initial complet multi-niveaux
            health_ok = await self._comprehensive_health_check()
            if not health_ok:
                logger.error("❌ Health check initial échoué")
                return False
            
            # Injection services dans app state (pattern Harena)
            self._inject_services_into_app_state(app)
            
            # Configuration middleware et routes avec validation
            self._configure_app_middleware_and_routes(app)
            
            # Warm-up optionnel du cache
            await self._optional_cache_warmup()
            
            # Finalisation
            self.service_healthy = True
            uptime = (datetime.now(timezone.utc) - self.service_start_time).total_seconds()
            
            logger.info("✅ Conversation Service Phase 1 initialisé avec succès")
            logger.info(f"📊 Configuration: {len(self.service_config['features'])} fonctionnalités actives")
            logger.info(f"🤖 DeepSeek: {self.service_config['deepseek_model']} avec JSON Output forcé")
            logger.info(f"🔐 Authentification: JWT obligatoire")
            logger.info(f"💾 Cache: Redis sémantique {"activé" if self.cache_manager else "désactivé"}")
            logger.info(f"⏱️ Temps initialisation: {uptime:.2f}s")
            
            return True
            
        except Exception as e:
            self.initialization_error = str(e)
            self.service_healthy = False
            logger.error(f"❌ Erreur critique initialisation: {str(e)}", exc_info=True)
            return False
    
    async def _validate_comprehensive_configuration(self) -> bool:
        """Validation configuration complète avec diagnostic détaillé"""
        try:
            validation_errors = []
            validation_warnings = []
            
            # DeepSeek API Key
            if not getattr(settings, 'DEEPSEEK_API_KEY', None):
                validation_errors.append("DEEPSEEK_API_KEY manquant")
            else:
                api_key = settings.DEEPSEEK_API_KEY
                if len(api_key) < 20:
                    validation_errors.append("DEEPSEEK_API_KEY semble invalide (trop court)")
                if not api_key.startswith(('sk-', 'test-')):
                    validation_warnings.append("DEEPSEEK_API_KEY format inhabituel")
            
            # Secret key used for bearer token verification across services
            if not getattr(settings, 'SECRET_KEY', None):
                validation_errors.append("SECRET_KEY manquant")
            else:
                jwt_secret = settings.SECRET_KEY
                if len(jwt_secret) < 32:
                    validation_errors.append("SECRET_KEY trop court (minimum 32 caractères)")
                if jwt_secret in ['changeme', 'secret', 'test']:
            # Secret Key
            if not getattr(settings, 'SECRET_KEY', None):
                validation_errors.append("SECRET_KEY manquant")
            else:
                secret = settings.SECRET_KEY
                if len(secret) < 32:
                    validation_errors.append("SECRET_KEY trop court (minimum 32 caractères)")
                if secret in ['changeme', 'secret', 'test']:
                    validation_errors.append("SECRET_KEY trop simple (sécurité faible)")
            
            # Configuration DeepSeek
            deepseek_url = getattr(settings, 'DEEPSEEK_BASE_URL', 'https://api.deepseek.com')
            if not deepseek_url.startswith('https://'):
                validation_warnings.append("DEEPSEEK_BASE_URL n'utilise pas HTTPS")
            
            # Configuration Cache (optionnelle)
            redis_config = getattr(settings, 'REDIS_URL', None)
            if not redis_config:
                validation_warnings.append("REDIS_URL non configuré - cache désactivé")
            
            # Configuration métriques
            if not getattr(settings, 'METRICS_ENABLED', True):
                validation_warnings.append("Métriques désactivées")
            
            # Log résultats validation
            if validation_errors:
                logger.error(f"❌ Erreurs configuration: {', '.join(validation_errors)}")
                return False
            
            if validation_warnings:
                for warning in validation_warnings:
                    logger.warning(f"⚠️ Configuration: {warning}")
            
            logger.info("✅ Configuration validée avec succès")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur validation configuration: {str(e)}")
            return False
    
    async def _initialize_external_clients_with_retry(self) -> bool:
        """Initialisation clients externes avec retry intelligent"""
        try:
            # Initialisation DeepSeek client avec retry
            logger.info("🤖 Initialisation client DeepSeek...")
            
            for attempt in range(3):
                try:
                    self.deepseek_client = DeepSeekClient()
                    await self.deepseek_client.initialize()
                    break
                except DeepSeekError as e:
                    logger.warning(f"Tentative {attempt + 1} DeepSeek échouée: {str(e)}")
                    if attempt == 2:
                        raise
                    await asyncio.sleep(2 ** attempt)
            
            # Test connexion DeepSeek
            deepseek_healthy = await self.deepseek_client.health_check()
            if not deepseek_healthy:
                logger.error("❌ DeepSeek API non accessible")
                return False
            
            logger.info("✅ DeepSeek client opérationnel")
            
            # Initialisation Cache Manager (non critique)
            logger.info("💾 Initialisation cache Redis...")
            try:
                self.cache_manager = CacheManager()
                await self.cache_manager.initialize()
                
                # Test connexion Redis (non bloquant)
                cache_healthy = await self.cache_manager.health_check()
                if cache_healthy:
                    logger.info("✅ Redis cache opérationnel")
                else:
                    logger.warning("⚠️ Redis indisponible - cache désactivé")
                    self.cache_manager = None
                    
            except Exception as e:
                logger.warning(f"⚠️ Cache Redis non disponible: {str(e)}")
                self.cache_manager = None
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur initialisation clients: {str(e)}")
            return False
    
    async def _validate_json_output_functionality(self) -> bool:
        """Validation fonctionnelle JSON Output DeepSeek"""
        try:
            logger.info("🔍 Validation JSON Output fonctionnalité...")
            
            if not self.deepseek_client:
                logger.error("❌ DeepSeek client non disponible pour validation JSON")
                return False
            
            # Test JSON Output avec prompt simple
            test_response = await self.deepseek_client.chat_completion(
                messages=[{
                    "role": "system", 
                    "content": "Tu réponds uniquement en JSON valide."
                }, {
                    "role": "user", 
                    "content": "Teste JSON Output avec cette structure: {\"test\": true, \"message\": \"validation\"}"
                }],
                max_tokens=100,
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            
            # Validation réponse
            if not test_response or "choices" not in test_response:
                logger.error("❌ Test JSON Output: réponse invalide")
                return False
            
            content = test_response["choices"][0]["message"]["content"]
            
            # Validation JSON parsing
            import json
            try:
                parsed_json = json.loads(content)
                if not isinstance(parsed_json, dict):
                    logger.error("❌ Test JSON Output: format non-objet")
                    return False
                    
                logger.info(f"✅ JSON Output fonctionnel: {parsed_json}")
                return True
                
            except json.JSONDecodeError as e:
                logger.error(f"❌ Test JSON Output: parsing échoué - {str(e)}")
                logger.error(f"Contenu reçu: {content}")
                return False
            
        except Exception as e:
            logger.error(f"❌ Erreur validation JSON Output: {str(e)}")
            return False
    
    async def _comprehensive_health_check(self) -> bool:
        """Health check complet multi-services avec diagnostic"""
        try:
            logger.info("🏥 Health check complet...")
            
            health_results = {
                "deepseek": {"status": False, "details": ""},
                "cache": {"status": False, "details": ""},
                "json_output": {"status": False, "details": ""},
                "configuration": {"status": True, "details": "OK"}
            }
            
            # Test DeepSeek
            if self.deepseek_client:
                try:
                    deepseek_health = await self.deepseek_client.health_check()
                    health_results["deepseek"]["status"] = deepseek_health
                    health_results["deepseek"]["details"] = "Opérationnel" if deepseek_health else "Inaccessible"
                except Exception as e:
                    health_results["deepseek"]["details"] = f"Erreur: {str(e)}"
            
            # Test Cache (non critique)
            if self.cache_manager:
                try:
                    cache_health = await self.cache_manager.health_check()
                    health_results["cache"]["status"] = cache_health
                    health_results["cache"]["details"] = "Opérationnel" if cache_health else "Indisponible"
                except Exception as e:
                    health_results["cache"]["details"] = f"Erreur: {str(e)}"
            else:
                health_results["cache"]["details"] = "Désactivé"
            
            # Validation JSON Output déjà effectuée
            health_results["json_output"]["status"] = True
            health_results["json_output"]["details"] = "Validé"
            
            # Évaluation globale
            critical_services = ["deepseek", "json_output", "configuration"]
            critical_ok = all(health_results[service]["status"] for service in critical_services)
            
            # Log détaillé
            for service, result in health_results.items():
                status_icon = "✅" if result["status"] else "❌"
                logger.info(f"{status_icon} {service.title()}: {result['details']}")
            
            if critical_ok:
                logger.info("🏥 Health check global: ✅ Services critiques opérationnels")
            else:
                logger.error("🏥 Health check global: ❌ Services critiques défaillants")
            
            return critical_ok
            
        except Exception as e:
            logger.error(f"❌ Erreur health check: {str(e)}")
            return False
    
    def _inject_services_into_app_state(self, app: FastAPI) -> None:
        """Injection services dans app state (pattern Harena amélioré)"""
        # Services principaux
        app.state.conversation_service = self
        app.state.deepseek_client = self.deepseek_client
        app.state.cache_manager = self.cache_manager
        
        # Configuration service
        app.state.service_config = self.service_config
        app.state.service_start_time = self.service_start_time
        
        # Métriques globales
        app.state.metrics_collector = metrics_collector
        
        # Metadata pour debugging
        app.state.service_metadata = {
            "initialization_time": datetime.now(timezone.utc),
            "python_version": sys.version,
            "service_loader_version": "1.0.0"
        }
        
        logger.info("✅ Services injectés dans app state")
    
    def _configure_app_middleware_and_routes(self, app: FastAPI) -> None:
        """Configuration middleware et routes avec validation"""
        
        # Middleware authentification JWT (ordre important)
        app.add_middleware(JWTAuthMiddleware)
        logger.info("🔐 Middleware JWT configuré")
        
        # Routes conversation avec préfixe API
        app.include_router(conversation_router, prefix="/api/v1")
        logger.info("🔗 Routes conversation configurées")
        
        # Routes de santé globales
        self._add_global_health_routes(app)
        logger.info("🏥 Routes santé configurées")
    
    def _add_global_health_routes(self, app: FastAPI) -> None:
        """Ajout routes de santé globales"""
        
        @app.get("/health/live")
        async def liveness_probe():
            """Probe liveness pour Kubernetes/Docker"""
            return {"status": "alive", "timestamp": datetime.now(timezone.utc).isoformat()}
        
        @app.get("/health/ready") 
        async def readiness_probe():
            """Probe readiness pour Kubernetes/Docker"""
            is_ready = (
                self.service_healthy and 
                self.deepseek_client and
                await self.deepseek_client.health_check()
            )
            
            status_code = 200 if is_ready else 503
            return JSONResponse(
                status_code=status_code,
                content={
                    "status": "ready" if is_ready else "not_ready",
                    "service_healthy": self.service_healthy,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            )
    
    async def _optional_cache_warmup(self) -> None:
        """Warm-up optionnel du cache avec données communes"""
        if not self.cache_manager:
            return
        
        try:
            # Exemples de données pour warm-up
            warmup_data = [
                {
                    "key": "common_greeting",
                    "data": {"intent": "GREETING", "confidence": 0.99},
                    "type": "intent"
                },
                {
                    "key": "common_balance",
                    "data": {"intent": "BALANCE_INQUIRY", "confidence": 0.98},
                    "type": "intent"
                }
            ]
            
            warmed = await self.cache_manager.warm_up_cache(warmup_data)
            if warmed > 0:
                logger.info(f"💾 Cache warm-up: {warmed} entrées préchargées")
                
        except Exception as e:
            logger.debug(f"Cache warm-up optionnel échoué: {str(e)}")
    
    async def cleanup(self) -> None:
        """Nettoyage ressources avec métriques finales"""
        try:
            cleanup_start = datetime.now(timezone.utc)
            
            # Métriques finales
            if self.service_healthy:
                uptime = (cleanup_start - self.service_start_time).total_seconds()
                final_metrics = metrics_collector.get_all_metrics()
                
                logger.info(f"📊 Métriques finales - Uptime: {uptime:.1f}s")
                logger.info(f"📊 Requêtes totales: {final_metrics.get('counters', {}).get('conversation.requests.total', 0)}")
                logger.info(f"📊 Taux succès: {100 - (final_metrics.get('counters', {}).get('conversation.errors.technical', 0) / max(final_metrics.get('counters', {}).get('conversation.requests.total', 1), 1) * 100):.1f}%")
            
            # Fermeture clients
            if self.deepseek_client:
                await self.deepseek_client.close()
                logger.info("🤖 DeepSeek client fermé")
            
            if self.cache_manager:
                await self.cache_manager.close()
                logger.info("💾 Cache manager fermé")
            
            cleanup_time = (datetime.now(timezone.utc) - cleanup_start).total_seconds()
            logger.info(f"✅ Nettoyage terminé en {cleanup_time:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ Erreur nettoyage: {str(e)}")


# Instance globale service loader
conversation_service_loader = ConversationServiceLoader()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestionnaire cycle de vie application optimisé"""
    startup_start = datetime.now(timezone.utc)
    
    # Startup
    logger.info("🚀 Démarrage application conversation service")
    
    try:
        # Initialisation service avec timeout
        initialization_success = await asyncio.wait_for(
            conversation_service_loader.initialize_conversation_service(app),
            timeout=60.0  # 60s timeout initialisation
        )
        
        startup_time = (datetime.now(timezone.utc) - startup_start).total_seconds()
        
        if initialization_success:
            logger.info(f"🎉 Service démarré avec succès en {startup_time:.2f}s")
        else:
            logger.error(f"❌ Échec initialisation en {startup_time:.2f}s - service dégradé")
            # App démarre quand même pour exposer health checks
        
        yield  # Application running
        
    except asyncio.TimeoutError:
        logger.error("❌ Timeout initialisation service (60s)")
        yield  # App démarre en mode dégradé
        
    except Exception as e:
        logger.error(f"❌ Erreur critique startup: {str(e)}", exc_info=True)
        yield  # App démarre en mode dégradé
    
    finally:
        # Shutdown
        shutdown_start = datetime.now(timezone.utc)
        logger.info("🔄 Arrêt application conversation service")
        
        try:
            await conversation_service_loader.cleanup()
            shutdown_time = (datetime.now(timezone.utc) - shutdown_start).total_seconds()
            logger.info(f"✅ Arrêt propre terminé en {shutdown_time:.2f}s")
        except Exception as e:
            logger.error(f"❌ Erreur arrêt: {str(e)}")

# Application FastAPI avec configuration optimisée
app = FastAPI(
    title="Harena Conversation Service",
    description="Service IA conversationnelle financière - Phase 1: Classification d'intentions avec JSON Output forcé",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if getattr(settings, 'ENVIRONMENT', 'production') != "production" else None,
    redoc_url="/redoc" if getattr(settings, 'ENVIRONMENT', 'production') != "production" else None,
    openapi_url="/openapi.json" if getattr(settings, 'ENVIRONMENT', 'production') != "production" else None
)

# Configuration CORS sécurisée
cors_origins = ["*"] if getattr(settings, 'ENVIRONMENT', 'production') != "production" else [
    "https://app.harena.fr",
    "https://api.harena.fr",
    "https://harenabackend-ab1b255e55c6.herokuapp.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    max_age=3600  # Cache preflight 1h
)

# Health check global principal (compatible pattern Harena)
@app.get("/health")
async def global_health():
    """Health check global compatible avec monitoring Harena"""
    try:
        if conversation_service_loader.service_healthy:
            health_metrics = metrics_collector.get_health_metrics()
            service_uptime = (datetime.now(timezone.utc) - conversation_service_loader.service_start_time).total_seconds()
            
            return JSONResponse(
                status_code=200,
                content={
                    "status": "healthy",
                    "service": "conversation_service",
                    "phase": conversation_service_loader.service_config["phase"],
                    "version": conversation_service_loader.service_config["version"],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "uptime_seconds": service_uptime,
                    "health_details": {
                        "total_requests": health_metrics.get("total_requests", 0),
                        "error_rate_percent": health_metrics.get("error_rate_percent", 0),
                        "avg_latency_ms": health_metrics.get("latency_p95_ms", 0)
                    },
                    "components": {
                        "deepseek_api": "operational",
                        "redis_cache": "operational" if conversation_service_loader.cache_manager else "disabled",
                        "intent_classification": "operational",
                        "json_output": "enforced",
                        "jwt_auth": "active"
                    },
                    "features": conversation_service_loader.service_config["features"]
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
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "components": {
                        "deepseek_api": "unknown",
                        "redis_cache": "unknown", 
                        "intent_classification": "unavailable",
                        "json_output": "unknown"
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
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )

# Endpoint métriques compatible Prometheus
@app.get("/metrics")
async def metrics_endpoint():
    """Métriques Prometheus pour monitoring"""
    try:
        if not conversation_service_loader.service_healthy:
            raise HTTPException(status_code=503, detail="Service non opérationnel")
        
        metrics_data = metrics_collector.get_all_metrics()
        service_uptime = (datetime.now(timezone.utc) - conversation_service_loader.service_start_time).total_seconds()
        
        # Format compatible monitoring Harena
        return {
            "service": "conversation_service",
            "timestamp": metrics_data.get("timestamp", datetime.now(timezone.utc).isoformat()),
            "uptime_seconds": service_uptime,
            "metrics": metrics_data,
            "service_info": {
                "phase": conversation_service_loader.service_config["phase"],
                "version": conversation_service_loader.service_config["version"],
                "features": conversation_service_loader.service_config["features"],
                "json_output_enforced": conversation_service_loader.service_config["json_output_enforced"]
            },
            "labels": {
                "service": "conversation_service",
                "phase": str(conversation_service_loader.service_config["phase"]),
                "version": conversation_service_loader.service_config["version"],
                "environment": getattr(settings, 'ENVIRONMENT', 'production')
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur export métriques: {str(e)}")
        raise HTTPException(status_code=500, detail="Erreur métriques")

# Handler erreurs globales optimisé
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Handler erreurs globales avec logging détaillé"""
    error_id = f"err_{int(datetime.now(timezone.utc).timestamp())}"
    
    logger.error(
        f"❌ [{error_id}] Erreur non gérée: {exc.__class__.__name__}: {str(exc)} | "
        f"Path: {getattr(request, 'url', {}).path if hasattr(request, 'url') else 'unknown'} | "
        f"Method: {getattr(request, 'method', 'unknown')}",
        exc_info=True
    )
    
    metrics_collector.increment_counter("conversation.errors.unhandled")
    
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Erreur interne du service",
            "service": "conversation_service",
            "error_id": error_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_path": str(request.url.path) if hasattr(request, 'url') else "unknown"
        }
    )

# Point d'entrée pour uvicorn avec configuration optimisée
if __name__ == "__main__":
    import uvicorn
    
    logger.info("🚀 Démarrage direct conversation service")
    
    # Configuration uvicorn
    uvicorn_config = {
        "app": "main:app",
        "host": "0.0.0.0",
        "port": int(getattr(settings, 'PORT', 8000)),
        "reload": getattr(settings, 'ENVIRONMENT', 'production') == "development",
        "log_level": "info",
        "access_log": True,
        "use_colors": True,
        "server_header": False,  # Sécurité
        "date_header": False,    # Sécurité
    }
    
    logger.info(f"⚙️ Configuration: {uvicorn_config}")
    uvicorn.run(**uvicorn_config)