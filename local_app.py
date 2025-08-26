"""
Application Harena pour développement local.
Version avec conversation_service.

✅ SERVICES DISPONIBLES:
- User Service: Gestion utilisateurs
- Sync Service: Synchronisation Bridge API
- Enrichment Service: Elasticsearch uniquement (v2.0)
- Search Service: Recherche lexicale simplifiée
- Conversation Service: IA conversationnelle (phase 1)
"""

import logging
import os
import sys
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from contextlib import asynccontextmanager

from config_service.config import settings

# Charger le fichier .env en priorité
load_dotenv()

# Configuration du logging simple
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger("harena_local")

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
                logger.info(f"✅ {service_name}: {routes_count} routes sur {prefix}")
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
                    logger.info(f"✅ {service_name}: Connexion DB OK")
                    return True
                except Exception as e:
                    logger.error(f"❌ {service_name}: Connexion DB échouée - {str(e)}")
                    return False
            
            # Pour les autres services, essayer d'importer le main
            try:
                main_module = __import__(f"{module_path}.main", fromlist=["app"])
                
                # Vérifier l'existence de l'app
                if hasattr(main_module, "app") or hasattr(main_module, "create_app"):
                    logger.info(f"✅ {service_name}: Module principal OK")
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
        logger.info("🚀 Démarrage Harena Finance Platform - LOCAL DEV")
        
        # Test DB critique
        try:
            from db_service.session import engine
            from sqlalchemy import text
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("✅ Base de données connectée")
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
        ]
        
        for service_name, module_path in services_health:
            loader.check_service_health(service_name, module_path)
        
        # Charger les routers des services
        logger.info("📋 Chargement des routes des services...")
        
        # 1. User Service
        try:
            from user_service.api.endpoints.users import router as user_router
            app.include_router(user_router, prefix="/api/v1/users", tags=["users"])
            routes_count = len(user_router.routes) if hasattr(user_router, 'routes') else 0
            logger.info(f"✅ user_service: {routes_count} routes sur /api/v1/users")
            loader.services_status["user_service"] = {"status": "ok", "routes": routes_count, "prefix": "/api/v1/users"}
        except Exception as e:
            logger.error(f"❌ User Service: {e}")
            loader.services_status["user_service"] = {"status": "error", "error": str(e)}

        # 2. Sync Service - modules principaux
        sync_modules = [
            ("sync_service.api.endpoints.sync", "/api/v1/sync", "Synchronisation"),
            ("sync_service.api.endpoints.transactions", "/api/v1/transactions", "Transactions"),
            ("sync_service.api.endpoints.accounts", "/api/v1/accounts", "Comptes"),
            ("sync_service.api.endpoints.categories", "/api/v1/categories", "Catégories"),
            ("sync_service.api.endpoints.items", "/api/v1/items", "Items Bridge"),
            ("sync_service.api.endpoints.webhooks", "/webhooks", "Webhooks"),
        ]

        sync_successful = 0
        for module_path, prefix, description in sync_modules:
            try:
                module = __import__(module_path, fromlist=["router"])
                router = getattr(module, "router")
                service_name = f"sync_{module_path.split('.')[-1]}"
                app.include_router(router, prefix=prefix, tags=[module_path.split('.')[-1]])
                routes_count = len(router.routes) if hasattr(router, 'routes') else 0
                logger.info(f"✅ {service_name}: {routes_count} routes sur {prefix}")
                loader.services_status[service_name] = {"status": "ok", "routes": routes_count, "prefix": prefix}
                sync_successful += 1
            except Exception as e:
                logger.error(f"❌ {module_path}: {e}")
                loader.services_status[f"sync_{module_path.split('.')[-1]}"] = {"status": "error", "error": str(e)}

        # 3. Enrichment Service - VERSION ELASTICSEARCH UNIQUEMENT (INCHANGÉ)
        logger.info("🔍 Chargement et initialisation enrichment_service (Elasticsearch uniquement)...")
        try:
            # Vérifier BONSAI_URL pour enrichment_service
            bonsai_url = settings.BONSAI_URL
            if not bonsai_url:
                logger.warning("⚠️ BONSAI_URL non configurée - enrichment_service sera en mode dégradé")
                enrichment_elasticsearch_available = False
                enrichment_init_success = False
            else:
                logger.info(f"📡 BONSAI_URL configurée pour enrichment: {bonsai_url[:50]}...")
                enrichment_elasticsearch_available = True
                
                # Initialiser les composants enrichment_service
                try:
                    logger.info("🔍 Initialisation des composants enrichment_service...")
                    from enrichment_service.storage.elasticsearch_client import ElasticsearchClient
                    from enrichment_service.core.processor import ElasticsearchTransactionProcessor
                    
                    # Créer et initialiser le client Elasticsearch pour enrichment
                    enrichment_elasticsearch_client = ElasticsearchClient()
                    await enrichment_elasticsearch_client.initialize()
                    logger.info("✅ Enrichment Elasticsearch client initialisé")
                    
                    # Créer le processeur
                    enrichment_processor = ElasticsearchTransactionProcessor(enrichment_elasticsearch_client)
                    logger.info("✅ Enrichment processor créé")
                    
                    # Injecter dans les routes enrichment_service
                    import enrichment_service.api.routes as enrichment_routes
                    enrichment_routes.elasticsearch_client = enrichment_elasticsearch_client
                    enrichment_routes.elasticsearch_processor = enrichment_processor
                    logger.info("✅ Instances injectées dans enrichment_service routes")
                    
                    enrichment_init_success = True
                    
                except Exception as e:
                    logger.error(f"❌ Erreur initialisation composants enrichment: {e}")
                    enrichment_init_success = False
            
            # Charger les routes enrichment_service
            from enrichment_service.api.routes import router as enrichment_router
            app.include_router(enrichment_router, prefix="/api/v1/enrichment", tags=["enrichment"])
            routes_count = len(enrichment_router.routes) if hasattr(enrichment_router, 'routes') else 0
            
            if enrichment_elasticsearch_available and enrichment_init_success:
                logger.info(f"✅ enrichment_service: {routes_count} routes sur /api/v1/enrichment (AVEC initialisation)")
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
        logger.info("🔍 Chargement et initialisation du search_service...")
        try:
            # Vérifier BONSAI_URL
            bonsai_url = settings.BONSAI_URL
            if not bonsai_url:
                raise ValueError("BONSAI_URL n'est pas configurée")
            
            logger.info(f"📡 BONSAI_URL configurée: {bonsai_url[:50]}...")
            
            # ✅ CORRECTION CRITIQUE : Import correct de mes classes corrigées
            from search_service.core.elasticsearch_client import ElasticsearchClient  # ✅ CORRIGÉ
            from search_service.core.search_engine import SearchEngine
            from search_service.api.routes import router as search_router, initialize_search_engine
            
            # Initialiser le client Elasticsearch directement
            logger.info("📡 Initialisation du client Elasticsearch...")
            elasticsearch_client = ElasticsearchClient()
            await elasticsearch_client.initialize()
            logger.info("✅ Client Elasticsearch initialisé")
            
            # Test de connexion
            health = await elasticsearch_client.health_check()
            if health.get("status") != "healthy":
                logger.warning(f"⚠️ Elasticsearch health: {health}")
            else:
                logger.info("✅ Test de connexion Elasticsearch réussi")
            
            # ✅ CORRECTION FINALE : Initialisation directe du moteur avec mes corrections
            search_engine = SearchEngine(
                elasticsearch_client=elasticsearch_client,
                cache_enabled=True
            )
            logger.info("🔥 SearchEngine initialisé avec TOUTES mes corrections!")
            
            # Injecter dans les routes
            initialize_search_engine(elasticsearch_client)
            
            # Charger les routes
            app.include_router(search_router, prefix="/api/v1/search", tags=["search"])
            routes_count = len(search_router.routes) if hasattr(search_router, 'routes') else 0
            logger.info(f"✅ search_service: {routes_count} routes sur /api/v1/search")
            
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
                "architecture": "corrected_final_v2"  # ✅ Version marquée
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

        # 5. Conversation Service
        logger.info("💬 Chargement et initialisation du conversation_service...")
        try:
            from conversation_service.main import ConversationServiceLoader
            from conversation_service.api.routes.conversation import router as conversation_router

            conversation_loader = ConversationServiceLoader()
            conversation_initialized = await conversation_loader.initialize_conversation_service(app)

            app.include_router(conversation_router, prefix="/api/v1/conversation", tags=["conversation"])
            routes_count = len(conversation_router.routes) if hasattr(conversation_router, 'routes') else 0

            loader.conversation_service_initialized = conversation_initialized
            loader.conversation_service_error = conversation_loader.initialization_error

            status = "ok" if conversation_initialized else "error"
            loader.services_status["conversation_service"] = {
                "status": status,
                "routes": routes_count,
                "prefix": "/api/v1/conversation",
                "initialized": conversation_initialized,
                "error": conversation_loader.initialization_error,
            }

            logger.info("✅ conversation_service: %s", "initialisé" if conversation_initialized else "non initialisé")
        except Exception as e:
            logger.error(f"❌ conversation_service: {e}")
            loader.conversation_service_initialized = False
            loader.conversation_service_error = str(e)
            loader.services_status["conversation_service"] = {"status": "error", "error": str(e)}

        # Compter les services réussis (INCHANGÉ)
        successful_services = len([s for s in loader.services_status.values() if s.get("status") in ["ok", "degraded"]])
        logger.info(f"✅ Démarrage terminé: {successful_services} services chargés")
        
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

    @app.get("/health")
    async def health():
        """Health check global (INCHANGÉ)"""
        ok_services = [name for name, status in loader.services_status.items() 
                      if status.get("status") == "ok"]
        degraded_services = [name for name, status in loader.services_status.items() 
                           if status.get("status") == "degraded"]
        
        # Détails spéciaux pour search_service, enrichment_service et conversation_service
        search_status = loader.services_status.get("search_service", {})
        enrichment_status = loader.services_status.get("enrichment_service", {})
        conversation_status = loader.services_status.get("conversation_service", {})
        
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
    logger.info("🔥 Lancement du serveur de développement avec hot reload")
    logger.info("📡 Accès: http://localhost:8000")
    logger.info("📚 Docs: http://localhost:8000/docs")
    logger.info("🔍 Status: http://localhost:8000/status")
    logger.info("🏦 Services Core: User, Sync, Enrichment, Search, Conversation")
    logger.info("✅ Architecture allégée pour développement core")
    
    uvicorn.run(
        "local_app:app", 
        host="0.0.0.0", 
        port=8000,
        reload=True,
        log_level="info"
    )