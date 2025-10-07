"""
Application Harena pour développement local.
Version avec conversation_service.

OK SERVICES DISPONIBLES:
- User Service: Gestion utilisateurs
- Sync Service: Synchronisation Bridge API
- Enrichment Service: Elasticsearch uniquement (v2.0)
- Search Service: Recherche lexicale simplifiée
- Conversation Service: IA conversationnelle (phase 1)
- Metric Service: Métriques et analytics financiers (Prophet ML)
"""

import logging
import os
import sys
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from contextlib import asynccontextmanager

# Charger le fichier .env en priorité (avant l'import des settings)
load_dotenv()

from config_service.config import settings
from conversation_service.api.middleware.auth_middleware import JWTAuthMiddleware

# Configuration du logging robuste pour le dev local
# - force=True garantit la réinitialisation même si Uvicorn a déjà configuré le logging
# - niveau pris des settings (fallback INFO)
# - sortie sur stdout et (optionnel) fichier si LOG_TO_FILE=true
def _configure_logging():
    level_name = getattr(settings, "LOG_LEVEL", "INFO") or "INFO"
    level = getattr(logging, str(level_name).upper(), logging.INFO)

    handlers = [logging.StreamHandler(sys.stdout)]

    # Toujours écrire un fichier local de dev pour simplifier le debug
    try:
        default_log_path = str(Path(__file__).with_name("harena_local.log"))
        default_fh = logging.FileHandler(default_log_path, encoding="utf-8")
        default_fh.setLevel(level)
        default_fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        handlers.append(default_fh)
    except Exception:
        # Ne jamais casser le démarrage à cause du fichier de log
        pass

    # Optionnellement écrire vers le fichier configuré par les settings
    try:
        if getattr(settings, "LOG_TO_FILE", False):
            configured_path = getattr(settings, "LOG_FILE", None)
            if configured_path:
                alt_fh = logging.FileHandler(configured_path, encoding="utf-8")
                alt_fh.setLevel(level)
                alt_fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
                handlers.append(alt_fh)
    except Exception:
        pass

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers,
        force=True,
    )

# S'assurer que le logger applicatif principal a son propre FileHandler dédié
def _ensure_named_logger_filehandler():
    try:
        logger_name = "harena_local"
        target_path = str(Path(__file__).with_name("harena_local.log"))
        lg = logging.getLogger(logger_name)
        # Éviter les doublons
        for h in lg.handlers:
            if hasattr(h, 'baseFilename') and getattr(h, 'baseFilename', None) == os.path.abspath(target_path):
                return
        fh = logging.FileHandler(target_path, encoding="utf-8")
        fh.setLevel(logging.getLogger().level)
        fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        lg.addHandler(fh)
        # S'assurer que ce logger écrit même si root est reconfiguré par uvicorn
        lg.propagate = True
    except Exception:
        pass

_configure_logging()
_ensure_named_logger_filehandler()
logger = logging.getLogger("harena_local")

# Attacher explicitement les loggers Uvicorn d'accès/erreurs au même fichier pour visibilité
def _attach_uvicorn_loggers():
    try:
        target_path = os.path.abspath(str(Path(__file__).with_name("harena_local.log")))
        fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        for lname in ("uvicorn.access", "uvicorn.error"):
            lg = logging.getLogger(lname)
            # éviter les doublons
            has = False
            for h in lg.handlers:
                if hasattr(h, 'baseFilename') and getattr(h, 'baseFilename', None) == target_path:
                    has = True
                    break
            if not has:
                fh = logging.FileHandler(target_path, encoding="utf-8")
                fh.setLevel(logging.getLogger().level)
                fh.setFormatter(fmt)
                lg.addHandler(fh)
    except Exception:
        pass

_attach_uvicorn_loggers()

# Fix DATABASE_URL pour développement local
DATABASE_URL = settings.DATABASE_URL
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    os.environ["DATABASE_URL"] = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Ajouter le répertoire courant au path
current_dir = Path(__file__).parent.absolute()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

class ServiceLoader:
    """Chargeur de services"""

    def __init__(self):
        self.services_status = {}
        self.search_service_initialized = False
        self.search_service_error = None
        self.conversation_service_initialized = False
        self.conversation_service_error = None
        self.metric_service_initialized = False
        self.metric_service_error = None

    def load_service_router(self, app: FastAPI, service_name: str, router_path: str, prefix: str):
        """Charge et enregistre un router de service"""
        try:
            # Import dynamique du router
            module = __import__(router_path, fromlist=["router"])
            router = getattr(module, "router", None)
            
            if router:
                # Enregistrer le router
                app.include_router(router, prefix=prefix, tags=[service_name])
                routes_count = len(router.routes) if hasattr(router, 'routes') else 0
                
                # Log et statut
                logger.info(f"OK {service_name}: {routes_count} routes sur {prefix}")
                self.services_status[service_name] = {"status": "ok", "routes": routes_count, "prefix": prefix}
                return True
            else:
                logger.error(f"❌ {service_name}: Pas de router trouvé")
                self.services_status[service_name] = {"status": "error", "error": "Pas de router"}
                return False
                
        except Exception as e:
            logger.error(f"❌ {service_name}: {str(e)}")
            self.services_status[service_name] = {"status": "error", "error": str(e)}
            return False
    
    def check_service_health(self, service_name: str, module_path: str):
        """Vérifie rapidement la santé d'un service"""
        try:
            # Pour db_service, pas de main.py à vérifier - juste tester la connexion
            if service_name == "db_service":
                try:
                    from db_service.session import engine
                    from sqlalchemy import text
                    with engine.connect() as conn:
                        conn.execute(text("SELECT 1"))
                    logger.info(f"OK {service_name}: Connexion DB OK")
                    return True
                except Exception as e:
                    logger.error(f"❌ {service_name}: Connexion DB échouée - {str(e)}")
                    return False
            
            # Pour les autres services, essayer d'importer le main
            try:
                main_module = __import__(f"{module_path}.main", fromlist=["app"])
                
                # Vérifier l'existence de l'app
                if hasattr(main_module, "app") or hasattr(main_module, "create_app"):
                    logger.info(f"OK {service_name}: Module principal OK")
                    return True
                else:
                    logger.warning(f"⚠️ {service_name}: Pas d'app FastAPI trouvée")
                    return False
            except ImportError:
                # Si pas de main.py, c'est pas forcément un problème (comme db_service)
                logger.info(f"ℹ️ {service_name}: Pas de main.py (normal pour certains services)")
                return True
                
        except Exception as e:
            logger.error(f"❌ {service_name}: Échec vérification - {str(e)}")
            return False

def create_app():
    """Créer l'application FastAPI principale"""

    loader = ServiceLoader()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("BOOT Demarrage Harena Finance Platform - LOCAL DEV")
        
        # Test DB critique
        try:
            from db_service.session import engine
            from sqlalchemy import text
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("OK Base de données connectée")
        except Exception as e:
            logger.error(f"❌ DB critique: {e}")
            raise RuntimeError("Database connection failed")
        
        # Vérifier santé des services existants
        services_health = [
            ("user_service", "user_service"),
            ("db_service", "db_service"),
            ("sync_service", "sync_service"),
            ("enrichment_service", "enrichment_service"),
            ("conversation_service", "conversation_service"),
            ("metric_service", "metric_service"),
        ]
        
        for service_name, module_path in services_health:
            loader.check_service_health(service_name, module_path)
        
        # Note: L'inclusion des routers a été déplacée vers la création de l'app
        # pour garantir leur présence dans l'OpenAPI schema

        # 3. Enrichment Service - VERSION ELASTICSEARCH UNIQUEMENT (INCHANGÉ)
        logger.info("INFO Chargement et initialisation enrichment_service (Elasticsearch uniquement)...")
        try:
            # Vérifier BONSAI_URL pour enrichment_service
            bonsai_url = settings.BONSAI_URL
            if not bonsai_url:
                logger.warning("⚠️ BONSAI_URL non configurée - enrichment_service sera en mode dégradé")
                enrichment_elasticsearch_available = False
                enrichment_init_success = False
            else:
                logger.info(f"CONF BONSAI_URL configurée pour enrichment: {bonsai_url[:50]}...")
                enrichment_elasticsearch_available = True
                
                # Initialiser les composants enrichment_service
                try:
                    logger.info("INFO Initialisation des composants enrichment_service...")
                    from enrichment_service.storage.elasticsearch_client import ElasticsearchClient
                    from enrichment_service.core.processor import ElasticsearchTransactionProcessor
                    
                    # Créer et initialiser le client Elasticsearch pour enrichment
                    enrichment_elasticsearch_client = ElasticsearchClient()
                    await enrichment_elasticsearch_client.initialize()
                    logger.info("OK Enrichment Elasticsearch client initialisé")
                    
                    # Créer le processeur
                    enrichment_processor = ElasticsearchTransactionProcessor(enrichment_elasticsearch_client)
                    logger.info("OK Enrichment processor créé")
                    
                    # Injecter dans les routes enrichment_service
                    import enrichment_service.api.routes as enrichment_routes
                    enrichment_routes.elasticsearch_client = enrichment_elasticsearch_client
                    enrichment_routes.elasticsearch_processor = enrichment_processor
                    logger.info("OK Instances injectées dans enrichment_service routes")
                    
                    enrichment_init_success = True
                    
                except Exception as e:
                    logger.error(f"❌ Erreur initialisation composants enrichment: {e}")
                    enrichment_init_success = False
            
            # Note: Router enrichment_service inclus dans create_app()
            routes_count = 8  # Nombre approximatif pour les statuts
            
            if enrichment_elasticsearch_available and enrichment_init_success:
                logger.info(f"OK enrichment_service: {routes_count} routes sur /api/v1/enrichment (AVEC initialisation)")
                loader.services_status["enrichment_service"] = {
                    "status": "ok", 
                    "routes": routes_count, 
                    "prefix": "/api/v1/enrichment",
                    "architecture": "elasticsearch_only",
                    "version": "2.0.0-elasticsearch",
                    "elasticsearch_available": True,
                    "initialized": True
                }
            else:
                logger.warning(f"⚠️ enrichment_service: {routes_count} routes chargées en mode dégradé")
                loader.services_status["enrichment_service"] = {
                    "status": "degraded", 
                    "routes": routes_count, 
                    "prefix": "/api/v1/enrichment",
                    "architecture": "elasticsearch_only",
                    "version": "2.0.0-elasticsearch",
                    "elasticsearch_available": enrichment_elasticsearch_available,
                    "initialized": enrichment_init_success,
                    "error": "BONSAI_URL not configured" if not enrichment_elasticsearch_available else "Initialization failed"
                }
                
        except Exception as e:
            logger.error(f"❌ Enrichment Service: {e}")
            loader.services_status["enrichment_service"] = {
                "status": "error", 
                "error": str(e),
                "architecture": "elasticsearch_only",
                "version": "2.0.0-elasticsearch"
            }

        # 4. Search Service - VERSION FINALE CORRIGÉE
        logger.info("INFO Chargement et initialisation du search_service...")
        try:
            # Vérifier BONSAI_URL
            bonsai_url = settings.BONSAI_URL
            if not bonsai_url:
                raise ValueError("BONSAI_URL n'est pas configurée")
            
            logger.info(f"CONF BONSAI_URL configurée: {bonsai_url[:50]}...")
            
            # OK CORRECTION CRITIQUE : Import correct de mes classes corrigées
            from search_service.core.elasticsearch_client import ElasticsearchClient  # OK CORRIGÉ
            from search_service.core.search_engine import SearchEngine
            from search_service.api.routes import router as search_router, initialize_search_engine
            
            # Initialiser le client Elasticsearch directement
            logger.info("CONF Initialisation du client Elasticsearch...")
            elasticsearch_client = ElasticsearchClient()
            await elasticsearch_client.initialize()
            logger.info("OK Client Elasticsearch initialisé")
            
            # Test de connexion
            health = await elasticsearch_client.health_check()
            if health.get("status") != "healthy":
                logger.warning(f"⚠️ Elasticsearch health: {health}")
            else:
                logger.info("OK Test de connexion Elasticsearch réussi")
            
            # OK CORRECTION FINALE : Initialisation directe du moteur avec mes corrections
            search_engine = SearchEngine(
                elasticsearch_client=elasticsearch_client,
                cache_enabled=True
            )
            logger.info("INIT SearchEngine initialisé avec TOUTES mes corrections!")
            
            # Injecter dans les routes
            initialize_search_engine(elasticsearch_client)
            
            # Note: Router search_service inclus dans create_app()
            routes_count = 5  # Nombre approximatif pour les statuts
            logger.info(f"OK search_service: {routes_count} routes sur /api/v1/search")
            
            # Mettre dans app.state
            app.state.service_initialized = True
            app.state.elasticsearch_client = elasticsearch_client
            app.state.search_engine = search_engine
            app.state.initialization_error = None
            
            loader.search_service_initialized = True
            loader.search_service_error = None
            
            loader.services_status["search_service"] = {
                "status": "ok", 
                "routes": routes_count, 
                "prefix": "/api/v1/search",
                "initialized": True,
                "architecture": "corrected_final_v2"  # OK Version marquée
            }
            
            logger.info("🎉 search_service: Complètement initialisé avec corrections FINALES!")
            
        except Exception as e:
            error_msg = f"Erreur initialisation search_service: {str(e)}"
            logger.error(f"❌ {error_msg}")
            logger.error(f"❌ Stacktrace: ", exc_info=True)
            
            # Marquer l'échec
            app.state.service_initialized = False
            app.state.elasticsearch_client = None
            app.state.initialization_error = error_msg
            
            loader.search_service_initialized = False
            loader.search_service_error = error_msg

            loader.services_status["search_service"] = {
                "status": "error",
                "error": error_msg,
                "architecture": "corrected_final_v2"
            }

        # 5. Conversation Service v2.0
        logger.info("💬 Chargement et initialisation du conversation_service v2.0...")
        try:
            # Import de la nouvelle architecture v2.0
            from conversation_service.api.routes.conversation import router as conversation_router
            from conversation_service.api.dependencies import app_state, get_application_lifespan

            # Initialisation du pipeline v2.0
            conversation_initialized = await app_state.initialize()

            # Note: Router conversation_service inclus dans create_app()
            routes_count = 9  # Nombre approximatif pour les statuts

            loader.conversation_service_initialized = conversation_initialized
            loader.conversation_service_error = app_state.initialization_error

            status = "ok" if conversation_initialized else "error"
            loader.services_status["conversation_service"] = {
                "status": status,
                "routes": routes_count,
                "prefix": "/api/v1/conversation",
                "initialized": conversation_initialized,
                "architecture": "v2.0",
                "pipeline_stages": 5,
                "error": app_state.initialization_error,
            }

            if conversation_initialized:
                logger.info("OK conversation_service v2.0: Pipeline complet initialise (5 stages)")
            else:
                logger.warning(f"WARNING conversation_service v2.0: Echec initialisation - {app_state.initialization_error}")

        except Exception as e:
            logger.error(f"ERROR conversation_service v2.0: {e}")
            loader.conversation_service_initialized = False
            loader.conversation_service_error = str(e)
            loader.services_status["conversation_service"] = {
                "status": "error",
                "error": str(e),
                "architecture": "v2.0"
            }

        # 6. Metric Service
        logger.info("📊 Chargement et initialisation du metric_service...")
        try:
            # Initialiser Redis cache pour metric_service
            from metric_service.core.cache import cache_manager

            # Connecter Redis
            await cache_manager.connect()
            logger.info("OK Metric Service: Redis cache connecté")

            # Note: Router metric_service inclus dans create_app()
            routes_count = 9  # Trends (2) + Health (4) + Patterns (1) + Forecasts (2)

            loader.metric_service_initialized = True
            loader.metric_service_error = None

            loader.services_status["metric_service"] = {
                "status": "ok",
                "routes": routes_count,
                "prefix": "/api/v1/metrics",
                "initialized": True,
                "architecture": "prophet_ml",
                "version": "1.0.0",
                "features": [
                    "MoM/YoY Trends",
                    "Savings Rate",
                    "Expense Ratios",
                    "Burn Rate & Runway",
                    "Balance Forecast (Prophet)",
                    "Recurring Expenses Detection"
                ]
            }

            logger.info("OK metric_service: Initialisé avec Prophet ML forecasting")

        except Exception as e:
            error_msg = f"Erreur initialisation metric_service: {str(e)}"
            logger.error(f"❌ {error_msg}")
            logger.error(f"❌ Stacktrace: ", exc_info=True)

            loader.metric_service_initialized = False
            loader.metric_service_error = error_msg

            loader.services_status["metric_service"] = {
                "status": "error",
                "error": error_msg,
                "architecture": "prophet_ml"
            }

        # Compter les services réussis (INCHANGÉ)
        successful_services = len([s for s in loader.services_status.values() if s.get("status") in ["ok", "degraded"]])
        logger.info(f"OK Démarrage terminé: {successful_services} services chargés")
        
        # Rapport final détaillé
        ok_services = [name for name, status in loader.services_status.items() if status.get("status") == "ok"]
        degraded_services = [name for name, status in loader.services_status.items() if status.get("status") == "degraded"]
        failed_services = [name for name, status in loader.services_status.items() if status.get("status") == "error"]
        
        logger.info(f"📊 Services OK: {', '.join(ok_services)}")
        if degraded_services:
            logger.warning(f"⚠️ Services dégradés mais fonctionnels: {', '.join(degraded_services)}")
        if failed_services:
            logger.error(f"❌ Services en erreur d'initialisation: {', '.join(failed_services)}")

        logger.info("🎉 Plateforme Harena complètement déployée!")

        try:
            yield
        finally:
            logger.info("🛑 Arrêt de Harena")

    app = FastAPI(
        title="Harena Finance Platform - Local Dev",
        description="Plateforme de gestion financière - Version développement avec conversation_service",
        version="1.0.0-dev",
        lifespan=lifespan
    )

    # CORS (INCHANGÉ)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.add_middleware(JWTAuthMiddleware)

    # Middleware HTTP fonctionnel (plus fiable que BaseHTTPMiddleware pour le logging)
    access_logger = logging.getLogger("uvicorn.access")

    @app.middleware("http")
    async def request_logging_middleware(request, call_next):
        import time as _time
        start = _time.time()
        path = request.url.path
        method = request.method
        client_ip = getattr(request.client, 'host', 'unknown') if request.client else 'unknown'
        try:
            response = await call_next(request)
            duration_ms = int((_time.time() - start) * 1000)
            user_id = getattr(getattr(request, 'state', object()), 'user_id', None)
            access_logger.info(f"{client_ip} - \"{method} {path}\" {response.status_code} {duration_ms}ms" + (f" user_id={user_id}" if user_id else ""))
            return response
        except Exception as e:
            duration_ms = int((_time.time() - start) * 1000)
            access_logger.error(f"{client_ip} - \"{method} {path}\" 500 {duration_ms}ms err={e}", exc_info=True)
            raise

    # Route de debug pour lister les routes disponibles
    @app.get("/__routes")
    async def list_routes():
        return sorted([f"{getattr(r, 'path', '')} [{','.join(getattr(r, 'methods', []) or [])}]" for r in app.router.routes])

    # ========================================
    # INCLUSION DES ROUTERS (déplacé du lifespan)
    # ========================================
    
    # 1. User Service
    try:
        from user_service.api.endpoints.users import router as user_router
        app.include_router(user_router, prefix="/api/v1/users", tags=["users"])
        logger.info("OK user_service router included")
    except Exception as e:
        logger.error(f"❌ User Service router: {e}")

    # 2. Sync Service - modules principaux
    sync_modules = [
        ("sync_service.api.endpoints.sync", "/api/v1/sync", "sync"),
        ("sync_service.api.endpoints.transactions", "/api/v1/transactions", "transactions"),
        ("sync_service.api.endpoints.accounts", "/api/v1/accounts", "accounts"),
        ("sync_service.api.endpoints.categories", "/api/v1/categories", "categories"),
        ("sync_service.api.endpoints.items", "/api/v1/items", "items"),
        ("sync_service.api.endpoints.webhooks", "/webhooks", "webhooks"),
    ]

    for module_path, prefix, tag in sync_modules:
        try:
            module = __import__(module_path, fromlist=["router"])
            router = getattr(module, "router")
            app.include_router(router, prefix=prefix, tags=[tag])
            logger.info(f"OK {module_path.split('.')[-1]} router included")
        except Exception as e:
            logger.error(f"❌ {module_path}: {e}")

    # 3. Enrichment Service
    try:
        from enrichment_service.api.routes import router as enrichment_router
        app.include_router(enrichment_router, prefix="/api/v1/enrichment", tags=["enrichment"])
        logger.info("OK enrichment_service router included")
    except Exception as e:
        logger.error(f"❌ Enrichment Service router: {e}")

    # 4. Search Service
    try:
        from search_service.api.routes import router as search_router
        app.include_router(search_router, prefix="/api/v1/search", tags=["search"])
        logger.info("OK search_service router included")
    except Exception as e:
        logger.error(f"❌ Search Service router: {e}")

    # 5. Conversation Service v2.0
    try:
        from conversation_service.api.routes.conversation import router as conversation_router
        app.include_router(conversation_router, tags=["conversation"])
        logger.info("OK conversation_service v2.0 router included (Architecture Complete)")
    except Exception as e:
        logger.error(f"ERROR Conversation Service v2.0 router: {e}")

    # 6. Metric Service - Tous les routers (anciens + nouveaux)
    metric_modules = [
        # Anciens routers (deprecated)
        ("metric_service.api.routes.trends", "/api/v1/metrics/trends", "metrics-trends"),
        ("metric_service.api.routes.health", "/api/v1/metrics/health", "metrics-health"),
        ("metric_service.api.routes.patterns", "/api/v1/metrics/patterns", "metrics-patterns"),
        # Nouveaux routers (5 métriques essentielles - Specs conformes)
        ("metric_service.api.routes.expenses", "/api/v1/metrics/expenses", "metrics-expenses"),
        ("metric_service.api.routes.income", "/api/v1/metrics/income", "metrics-income"),
        ("metric_service.api.routes.coverage", "/api/v1/metrics/coverage", "metrics-coverage"),
    ]

    for module_path, prefix, tag in metric_modules:
        try:
            module = __import__(module_path, fromlist=["router"])
            router = getattr(module, "router")
            routes_count = len(router.routes) if hasattr(router, 'routes') else 0
            app.include_router(router, prefix=prefix, tags=[tag])
            logger.info(f"OK {module_path.split('.')[-1]} router included - {routes_count} routes on {prefix}")
        except Exception as e:
            logger.error(f"❌ {module_path}: {e}")
            import traceback
            logger.error(traceback.format_exc())

    @app.get("/health")
    async def health():
        """Health check global (INCHANGÉ)"""
        ok_services = [name for name, status in loader.services_status.items() 
                      if status.get("status") == "ok"]
        degraded_services = [name for name, status in loader.services_status.items() 
                           if status.get("status") == "degraded"]
        
        # Détails spéciaux pour search_service, enrichment_service, conversation_service et metric_service
        search_status = loader.services_status.get("search_service", {})
        enrichment_status = loader.services_status.get("enrichment_service", {})
        conversation_status = loader.services_status.get("conversation_service", {})
        metric_status = loader.services_status.get("metric_service", {})
        
        return {
            "status": "healthy" if ok_services else ("degraded" if degraded_services else "unhealthy"),
            "services_ok": len(ok_services),
            "services_degraded": len(degraded_services),
            "total_services": len(loader.services_status),
            "services": {
                "ok": list(ok_services),
                "degraded": list(degraded_services)
            },
            "search_service": {
                "status": search_status.get("status"),
                "initialized": search_status.get("initialized", False),
                "error": search_status.get("error"),
                "architecture": search_status.get("architecture")
            },
            "enrichment_service": {
                "status": enrichment_status.get("status"),
                "architecture": enrichment_status.get("architecture"),
                "version": enrichment_status.get("version"),
                "elasticsearch_available": enrichment_status.get("elasticsearch_available", False),
                "initialized": enrichment_status.get("initialized", False),
                "error": enrichment_status.get("error")
            },
            "conversation_service": {
                "status": conversation_status.get("status"),
                "initialized": conversation_status.get("initialized", False),
                "error": conversation_status.get("error"),
            },
            "metric_service": {
                "status": metric_status.get("status"),
                "initialized": metric_status.get("initialized", False),
                "error": metric_status.get("error"),
                "architecture": metric_status.get("architecture")
            }
        }

    @app.get("/status")
    async def status():
        """Statut détaillé (INCHANGÉ)"""
        return {
            "platform": "Harena Finance",
            "services": loader.services_status,
            "environment": settings.ENVIRONMENT,
            "search_service_details": {
                "initialized": loader.search_service_initialized,
                "error": loader.search_service_error,
                "architecture": "corrected_final_v2"
            },
            "enrichment_service_details": {
                "architecture": "elasticsearch_only",
                "version": "2.0.0-elasticsearch",
                "features": [
                    "Transaction structuring",
                    "Elasticsearch indexing",
                    "Batch processing",
                    "User sync operations"
                ]
            },
            "conversation_service_details": {
                "initialized": loader.conversation_service_initialized,
                "error": loader.conversation_service_error,
            },
            "metric_service_details": {
                "initialized": loader.metric_service_initialized,
                "error": loader.metric_service_error,
                "architecture": "prophet_ml",
                "version": "1.0.0",
                "features": loader.services_status.get("metric_service", {}).get("features", [])
            }
        }

    @app.get("/")
    async def root():
        """Page d'accueil (INCHANGÉ)"""
        return {
            "message": "🏦 Harena Finance Platform - LOCAL DEVELOPMENT (Core Services)",
            "version": "1.0.0-dev-core",
            "services_available": [
                "user_service - Gestion utilisateurs",
                "sync_service - Synchronisation Bridge API",
                "enrichment_service - Enrichissement Elasticsearch (v2.0)",
                "search_service - Recherche lexicale (Architecture finale corrigée)",
                "conversation_service - IA conversationnelle",
                "metric_service - Métriques et analytics financiers (Prophet ML)",
            ],
            "endpoints": {
                "/health": "Contrôle santé",
                "/status": "Statut des services",
                "/docs": "Documentation interactive",
            },
        }

    return app

# Créer l'app
app = create_app()

if __name__ == "__main__":
    import uvicorn
    logger.info("Lancement du serveur de developpement avec hot reload")
    logger.info("Acces: http://localhost:8000")
    logger.info("Docs: http://localhost:8000/docs")
    logger.info("Status: http://localhost:8000/status")
    logger.info("Services Core: User, Sync, Enrichment, Search, Conversation, Metrics")
    logger.info("Architecture allegee pour developpement core avec securite JWT")

    # Configuration explicite de logging pour Uvicorn (console + fichier)
    log_file_path = str(Path(__file__).with_name("harena_local.log"))
    uvicorn_log_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"},
            "access": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"},
        },
        "handlers": {
            "console": {"class": "logging.StreamHandler", "formatter": "default", "stream": "ext://sys.stdout"},
            "file": {"class": "logging.FileHandler", "formatter": "default", "filename": log_file_path, "encoding": "utf-8"},
            "file_access": {"class": "logging.FileHandler", "formatter": "access", "filename": log_file_path, "encoding": "utf-8"},
        },
        "loggers": {
            "uvicorn": {"handlers": ["console", "file"], "level": "INFO"},
            "uvicorn.error": {"handlers": ["console", "file"], "level": "INFO", "propagate": False},
            "uvicorn.access": {"handlers": ["console", "file_access"], "level": "INFO", "propagate": False},
            "harena_local": {"handlers": ["console", "file"], "level": "INFO", "propagate": True},
        },
    }

    uvicorn.run(
        "local_app:app", 
        host="::",  # écoute IPv4/IPv6 pour couvrir localhost
        port=8000,
        reload=True,
        log_level="info",
        access_log=True,
        log_config=uvicorn_log_config
    )
