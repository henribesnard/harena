"""
Application Harena simplifiée pour Heroku.
Inspiré de la version complexe qui fonctionnait bien.
"""


import logging
import os
import sys
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Configuration du logging simple
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger("harena")

# Fix Heroku DATABASE_URL
DATABASE_URL = os.environ.get("DATABASE_URL")
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    os.environ["DATABASE_URL"] = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Ajouter le répertoire courant au path
current_dir = Path(__file__).parent.absolute()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

class ServiceLoader:
    """Chargeur de services simplifié inspiré de la version complexe."""
    
    def __init__(self):
        self.services_status = {}
        self.search_service_initialized = False
        self.search_service_error = None
    
    async def initialize_search_service(self, app: FastAPI):
        """Initialise le search_service avec ses composants Elasticsearch."""
        logger.info("🔍 Initialisation du search_service...")
        
        try:
            # Vérifier BONSAI_URL
            bonsai_url = os.environ.get("BONSAI_URL")
            if not bonsai_url:
                raise ValueError("BONSAI_URL n'est pas configurée")
            
            logger.info(f"📡 BONSAI_URL configurée: {bonsai_url[:50]}...")
            
            # Import des modules search_service
            from search_service.clients.elasticsearch_client import initialize_default_client
            from search_service.core import core_manager
            
            # Initialiser le client Elasticsearch
            logger.info("📡 Initialisation du client Elasticsearch...")
            elasticsearch_client = await initialize_default_client()
            logger.info("✅ Client Elasticsearch initialisé")
            
            # Test de connexion
            connection_test = await elasticsearch_client.test_connection()
            if not connection_test:
                raise RuntimeError("Test de connexion Elasticsearch échoué")
            logger.info("✅ Test de connexion Elasticsearch réussi")
            
            # Initialiser le core manager
            logger.info("🧠 Initialisation du core manager...")
            await core_manager.initialize(elasticsearch_client)
            
            if not core_manager.is_initialized():
                raise RuntimeError("Core manager non initialisé")
            logger.info("✅ Core manager initialisé")
            
            # Mettre les composants dans app.state pour les routes
            app.state.service_initialized = True
            app.state.elasticsearch_client = elasticsearch_client
            app.state.core_manager = core_manager
            app.state.initialization_error = None
            
            self.search_service_initialized = True
            self.search_service_error = None
            
            logger.info("🎉 Search Service complètement initialisé!")
            return True
            
        except Exception as e:
            error_msg = f"Erreur initialisation search_service: {str(e)}"
            logger.error(f"❌ {error_msg}")
            
            # Marquer l'échec dans app.state
            app.state.service_initialized = False
            app.state.elasticsearch_client = None
            app.state.core_manager = None
            app.state.initialization_error = error_msg
            
            self.search_service_initialized = False
            self.search_service_error = error_msg
            return False
    
    def load_service_router(self, app: FastAPI, service_name: str, router_path: str, prefix: str):
        """Charge et enregistre un router de service."""
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
        """Vérifie rapidement la santé d'un service - version inspirée de la complexe."""
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
    """Créer l'application FastAPI principale."""
    
    app = FastAPI(
        title="Harena Finance Platform",
        description="Plateforme de gestion financière avec IA",
        version="1.0.0"
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    loader = ServiceLoader()

    @app.on_event("startup")
    async def startup():
        logger.info("🚀 Démarrage Harena Finance Platform")
        
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

        # 2. Sync Service - modules principaux (inspiré de la version complexe)
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

        # 3. Enrichment Service
        try:
            from enrichment_service.api.routes import router as enrichment_router
            app.include_router(enrichment_router, prefix="/api/v1/enrichment", tags=["enrichment"])
            routes_count = len(enrichment_router.routes) if hasattr(enrichment_router, 'routes') else 0
            logger.info(f"✅ enrichment_service: {routes_count} routes sur /api/v1/enrichment")
            loader.services_status["enrichment_service"] = {"status": "ok", "routes": routes_count, "prefix": "/api/v1/enrichment"}
        except Exception as e:
            logger.error(f"❌ Enrichment Service: {e}")
            loader.services_status["enrichment_service"] = {"status": "error", "error": str(e)}

        # 4. ✅ Search Service - AVEC INITIALISATION CORRECTE
        logger.info("🔍 Chargement et initialisation du search_service...")
        try:
            # D'abord initialiser les composants Elasticsearch
            search_init_success = await loader.initialize_search_service(app)
            
            # Ensuite charger les routes (même si init échoue pour debug)
            try:
                from search_service.api.routes import router as search_router
                app.include_router(search_router, prefix="/api/v1/search", tags=["search"])
                routes_count = len(search_router.routes) if hasattr(search_router, 'routes') else 0
                
                if search_init_success:
                    logger.info(f"✅ search_service: {routes_count} routes sur /api/v1/search (AVEC initialisation)")
                    loader.services_status["search_service"] = {
                        "status": "ok", 
                        "routes": routes_count, 
                        "prefix": "/api/v1/search",
                        "initialized": True
                    }
                else:
                    logger.warning(f"⚠️ search_service: {routes_count} routes chargées SANS initialisation")
                    loader.services_status["search_service"] = {
                        "status": "degraded", 
                        "routes": routes_count, 
                        "prefix": "/api/v1/search",
                        "initialized": False,
                        "error": loader.search_service_error
                    }
                    
            except ImportError as e1:
                logger.warning(f"⚠️ Import direct échoué: {e1}")
                
                # Fallback: Via api_manager
                try:
                    from search_service.api import api_manager
                    
                    if hasattr(api_manager, 'router') and api_manager.router:
                        app.include_router(api_manager.router, prefix="/api/v1/search", tags=["search"])
                        routes_count = len(api_manager.router.routes) if hasattr(api_manager.router, 'routes') else 0
                        
                        status = "ok" if search_init_success else "degraded"
                        logger.info(f"✅ search_service (api_manager): {routes_count} routes sur /api/v1/search")
                        loader.services_status["search_service"] = {
                            "status": status, 
                            "routes": routes_count, 
                            "prefix": "/api/v1/search",
                            "initialized": search_init_success,
                            "error": loader.search_service_error if not search_init_success else None
                        }
                    else:
                        raise ImportError("api_manager.router non disponible")
                        
                except ImportError as e2:
                    logger.error(f"❌ search_service: Tous les imports ont échoué")
                    logger.error(f"   - Direct: {e1}")
                    logger.error(f"   - API Manager: {e2}")
                    loader.services_status["search_service"] = {"status": "error", "error": f"Import failed: {str(e2)}"}
                    
        except Exception as e:
            logger.error(f"❌ search_service: Erreur générale - {str(e)}")
            loader.services_status["search_service"] = {"status": "error", "error": str(e)}

        # Compter les services réussis
        successful_services = len([s for s in loader.services_status.values() if s.get("status") in ["ok", "degraded"]])
        logger.info(f"✅ Démarrage terminé: {successful_services} services chargés")
        
        # Rapport final détaillé
        ok_services = [name for name, status in loader.services_status.items() if status.get("status") == "ok"]
        degraded_services = [name for name, status in loader.services_status.items() if status.get("status") == "degraded"]
        failed_services = [name for name, status in loader.services_status.items() if status.get("status") == "error"]
        
        logger.info(f"📊 Services OK: {', '.join(ok_services)}")
        if degraded_services:
            logger.warning(f"📊 Services dégradés: {', '.join(degraded_services)}")
        if failed_services:
            logger.warning(f"📊 Services en erreur: {', '.join(failed_services)}")
        
        logger.info("🔮 À venir: conversation_service avec AutoGen + DeepSeek")

    @app.get("/health")
    async def health():
        """Health check global avec détails search_service."""
        ok_services = [name for name, status in loader.services_status.items() 
                      if status.get("status") == "ok"]
        degraded_services = [name for name, status in loader.services_status.items() 
                           if status.get("status") == "degraded"]
        
        # Détail spécial pour search_service
        search_status = loader.services_status.get("search_service", {})
        
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
                "error": search_status.get("error")
            }
        }

    @app.get("/status")
    async def status():
        """Statut détaillé."""
        return {
            "platform": "Harena Finance",
            "services": loader.services_status,
            "environment": os.environ.get("ENVIRONMENT", "production"),
            "search_service_details": {
                "initialized": loader.search_service_initialized,
                "error": loader.search_service_error
            }
        }

    @app.get("/")
    async def root():
        """Page d'accueil."""
        return {
            "message": "🏦 Harena Finance Platform",
            "version": "1.0.0",
            "services_available": [
                "user_service - Gestion utilisateurs",
                "sync_service - Synchronisation Bridge API", 
                "enrichment_service - Enrichissement IA",
                "search_service - Recherche lexicale (AVEC initialisation)"
            ],
            "services_coming_soon": [
                "conversation_service - Assistant IA avec AutoGen + DeepSeek"
            ],
            "endpoints": {
                "/health": "Contrôle santé",
                "/status": "Statut des services",
                "/api/v1/users/*": "Gestion utilisateurs",
                "/api/v1/sync/*": "Synchronisation",
                "/api/v1/transactions/*": "Transactions",
                "/api/v1/accounts/*": "Comptes",
                "/api/v1/categories/*": "Catégories",
                "/api/v1/enrichment/*": "Enrichissement IA",
                "/api/v1/search/*": "Recherche lexicale (INITIALISÉ)"
            }
        }

    return app

# Créer l'app
app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("heroku_app:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))