"""
Application Harena pour développement local.
Version sans conversation_service - Services core uniquement.

✅ SERVICES MAINTENUS:
- User Service: Gestion utilisateurs
- Sync Service: Synchronisation Bridge API
- Enrichment Service: Elasticsearch uniquement (v2.0)
- Search Service: Recherche lexicale simplifiée

❌ SUPPRIMÉ:
- Conversation Service: Complètement retiré
"""

import logging
import os
import sys
from pathlib import Path
from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

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
DATABASE_URL = os.environ.get("DATABASE_URL")
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    os.environ["DATABASE_URL"] = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Ajouter le répertoire courant au path
current_dir = Path(__file__).parent.absolute()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

class ServiceLoader:
    """Chargeur de services sans conversation_service"""
    
    def __init__(self):
        self.services_status = {}
        self.search_service_initialized = False
        self.search_service_error = None
    
    async def initialize_search_service(self, app: FastAPI):
        """Initialise le search_service - COPIE EXACTE"""
        logger.info("🔍 Initialisation du search_service...")
        
        try:
            # Vérifier BONSAI_URL
            bonsai_url = os.environ.get("BONSAI_URL")
            if not bonsai_url:
                raise ValueError("BONSAI_URL n'est pas configurée")
            
            logger.info(f"📡 BONSAI_URL configurée: {bonsai_url[:50]}...")
            
            # Import des modules search_service avec nouvelle architecture
            from search_service.core import initialize_default_client
            from search_service.api import initialize_search_engine
            
            # Initialiser le client Elasticsearch
            logger.info("📡 Initialisation du client Elasticsearch...")
            elasticsearch_client = await initialize_default_client()
            logger.info("✅ Client Elasticsearch initialisé")
            
            # Test de connexion
            health = await elasticsearch_client.health_check()
            if health.get("status") != "healthy":
                logger.warning(f"⚠️ Elasticsearch health: {health}")
            else:
                logger.info("✅ Test de connexion Elasticsearch réussi")
            
            # Initialiser le moteur de recherche
            logger.info("🔍 Initialisation du moteur de recherche...")
            initialize_search_engine(elasticsearch_client)
            logger.info("✅ Moteur de recherche initialisé")
            
            # Mettre les composants dans app.state pour les routes
            app.state.service_initialized = True
            app.state.elasticsearch_client = elasticsearch_client
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
            app.state.initialization_error = error_msg
            
            self.search_service_initialized = False
            self.search_service_error = error_msg
            return False
    
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
    """Créer l'application FastAPI principale sans conversation_service"""
    
    app = FastAPI(
        title="Harena Finance Platform - Local Dev",
        description="Plateforme de gestion financière - Version développement sans conversation_service",
        version="1.0.0-dev"
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
        logger.info("🚀 Démarrage Harena Finance Platform - LOCAL DEV sans conversation_service")
        
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

        # 3. Enrichment Service - VERSION ELASTICSEARCH UNIQUEMENT
        logger.info("🔍 Chargement et initialisation enrichment_service (Elasticsearch uniquement)...")
        try:
            # Vérifier BONSAI_URL pour enrichment_service
            bonsai_url = os.environ.get("BONSAI_URL")
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

        # 4. Search Service
        logger.info("🔍 Chargement et initialisation du search_service...")
        try:
            # D'abord initialiser les composants Elasticsearch
            search_init_success = await loader.initialize_search_service(app)
            
            # Ensuite charger les routes avec la méthode standardisée
            if search_init_success:
                router_success = loader.load_service_router(
                    app, 
                    "search_service", 
                    "search_service.api.routes", 
                    "/api/v1/search"
                )
                
                if router_success:
                    # Initialiser le moteur dans les routes
                    try:
                        from search_service.api import initialize_search_engine
                        elasticsearch_client = getattr(app.state, 'elasticsearch_client', None)
                        if elasticsearch_client:
                            initialize_search_engine(elasticsearch_client)
                            logger.info("✅ Search engine initialisé dans les routes")
                    except Exception as e:
                        logger.warning(f"⚠️ Erreur initialisation search engine dans routes: {e}")
                    
                    loader.services_status["search_service"]["initialized"] = True
                    loader.services_status["search_service"]["architecture"] = "simplified_unified"
                    logger.info("✅ search_service: Complètement initialisé")
                else:
                    logger.error("❌ search_service: Initialisation OK mais router non chargé")
                    loader.services_status["search_service"] = {
                        "status": "degraded", 
                        "routes": 0, 
                        "prefix": "/api/v1/search",
                        "initialized": True,
                        "error": "Router loading failed",
                        "architecture": "simplified_unified"
                    }
            else:
                logger.error("❌ search_service: Initialisation des composants échouée")
                loader.services_status["search_service"] = {
                    "status": "error", 
                    "error": loader.search_service_error,
                    "architecture": "simplified_unified"
                }
                    
        except Exception as e:
            logger.error(f"❌ search_service: Erreur générale - {str(e)}")
            loader.services_status["search_service"] = {
                "status": "error", 
                "error": str(e),
                "architecture": "simplified_unified"
            }

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
        
        logger.info("🎉 Plateforme Harena complètement déployée (sans conversation_service)!")

    @app.get("/health")
    async def health():
        """Health check global sans conversation_service"""
        ok_services = [name for name, status in loader.services_status.items() 
                      if status.get("status") == "ok"]
        degraded_services = [name for name, status in loader.services_status.items() 
                           if status.get("status") == "degraded"]
        
        # Détails spéciaux pour search_service et enrichment_service
        search_status = loader.services_status.get("search_service", {})
        enrichment_status = loader.services_status.get("enrichment_service", {})
        
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
            }
        }

    @app.get("/status")
    async def status():
        """Statut détaillé sans conversation_service"""
        return {
            "platform": "Harena Finance",
            "services": loader.services_status,
            "environment": os.environ.get("ENVIRONMENT", "development"),
            "search_service_details": {
                "initialized": loader.search_service_initialized,
                "error": loader.search_service_error,
                "architecture": "simplified_unified"
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
            }
        }

    @app.get("/")
    async def root():
        """Page d'accueil sans conversation_service"""
        return {
            "message": "🏦 Harena Finance Platform - LOCAL DEVELOPMENT (Core Services)",
            "version": "1.0.0-dev-core",
            "services_available": [
                "user_service - Gestion utilisateurs",
                "sync_service - Synchronisation Bridge API", 
                "enrichment_service - Enrichissement Elasticsearch (v2.0)",
                "search_service - Recherche lexicale (Architecture simplifiée)"
            ],
            "services_removed": [
                "conversation_service - Complètement supprimé"
            ],
            "endpoints": {
                "/health": "Contrôle santé",
                "/status": "Statut des services",
                "/docs": "Documentation interactive",
                "/api/v1/users/*": "Gestion utilisateurs",
                "/api/v1/sync/*": "Synchronisation",
                "/api/v1/transactions/*": "Transactions",
                "/api/v1/accounts/*": "Comptes",
                "/api/v1/categories/*": "Catégories",
                "/api/v1/enrichment/elasticsearch/*": "Enrichissement Elasticsearch (v2.0)",
                "/api/v1/search/*": "Recherche lexicale (Architecture unifiée)"
            },
            "development_mode": {
                "hot_reload": True,
                "debug_logs": True,
                "local_services": ["PostgreSQL", "Redis", "Elasticsearch (requis pour enrichment + search)"],
                "docs_url": "http://localhost:8000/docs"
            },
            "architecture_updates": {
                "core_services_only": {
                    "description": "Version allégée sans intelligence conversationnelle",
                    "maintained_services": [
                        "user_service - API utilisateurs complète",
                        "sync_service - Synchronisation Bridge API",
                        "enrichment_service - Elasticsearch v2.0",
                        "search_service - Recherche lexicale optimisée"
                    ],
                    "benefits": [
                        "🚀 Démarrage plus rapide",
                        "💾 Moins d'utilisation mémoire",
                        "🔧 Maintenance simplifiée",
                        "📦 Architecture core stable"
                    ]
                },
                "enrichment_service": {
                    "version": "2.0.0-elasticsearch",
                    "changes": [
                        "Suppression complète de Qdrant et embeddings",
                        "Architecture Elasticsearch uniquement",
                        "Nouveaux endpoints: /api/v1/enrichment/elasticsearch/*",
                        "Performance optimisée pour l'indexation bulk",
                        "Simplification drastique du code"
                    ]
                },
                "search_service": {
                    "version": "simplified_unified",
                    "changes": [
                        "Recherche lexicale pure Elasticsearch",
                        "Architecture simplifiée sans IA",
                        "Performance optimisée < 50ms",
                        "Cache intelligent des résultats"
                    ]
                }
            },
            "available_features": {
                "user_management": {
                    "endpoints": "/api/v1/users/*",
                    "features": ["Registration", "Authentication", "Profile management"]
                },
                "data_sync": {
                    "endpoints": "/api/v1/sync/*, /api/v1/transactions/*",
                    "features": ["Bridge API sync", "Transaction management", "Account sync"]
                },
                "data_enrichment": {
                    "endpoints": "/api/v1/enrichment/elasticsearch/*",
                    "features": ["Transaction enrichment", "Elasticsearch indexing", "Batch processing"]
                },
                "search": {
                    "endpoints": "/api/v1/search/*",
                    "features": ["Lexical search", "Transaction filtering", "Performance optimization"]
                }
            },
            "testing": {
                "status": "Tous les services core testables",
                "quick_tests": {
                    "health": "GET /health",
                    "user_service": "GET /api/v1/users/me",
                    "enrichment": "GET /api/v1/enrichment/health",
                    "search": "GET /api/v1/search/health"
                }
            }
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
    logger.info("🏦 Services Core: User, Sync, Enrichment, Search")
    logger.info("❌ Conversation Service: Complètement supprimé")
    logger.info("✅ Architecture allégée pour développement core")
    
    uvicorn.run(
        "local_app:app", 
        host="0.0.0.0", 
        port=8000,
        reload=True,
        log_level="info"
    )