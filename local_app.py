"""
Application Harena pour tests locaux avant déploiement.
Version de développement avec diagnostics étendus et hot reload.
"""

import logging
import os
import sys
import asyncio
from pathlib import Path
from typing import Dict, List, Tuple
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Configuration du logging pour développement
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger("harena_local")

# Configuration locale par défaut
def setup_local_environment():
    """Configure l'environnement local avec des valeurs par défaut."""
    
    # Base de données locale par défaut
    if not os.environ.get("DATABASE_URL"):
        os.environ["DATABASE_URL"] = "postgresql://localhost:5432/harena_dev"
        logger.info("🔧 DATABASE_URL définie pour développement local")
    
    # Variables d'environnement par défaut pour les tests
    default_env = {
        "ENVIRONMENT": "development",
        "DEBUG": "true",
        "ELASTICSEARCH_URL": "http://localhost:9200",
        "QDRANT_URL": "http://localhost:6333",
        "REDIS_URL": "redis://localhost:6379",
        # Variables OpenAI pour les tests (à remplacer par vos clés)
        "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", "sk-test-key"),
        "EMBEDDING_MODEL": "text-embedding-3-small",
        # Variables Bridge API (optionnelles pour tests)
        "BRIDGE_BASE_URL": "https://sync.bankin.com",
        "BRIDGE_CLIENT_ID": os.environ.get("BRIDGE_CLIENT_ID", ""),
        "BRIDGE_CLIENT_SECRET": os.environ.get("BRIDGE_CLIENT_SECRET", ""),
    }
    
    for key, value in default_env.items():
        if not os.environ.get(key):
            os.environ[key] = value
    
    logger.info("✅ Environnement local configuré")

# Ajouter le répertoire courant au path
current_dir = Path(__file__).parent.absolute()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

class LocalServiceTester:
    """Testeur de services pour développement local."""
    
    def __init__(self):
        self.services_status = {}
        self.detailed_errors = {}
    
    def test_database_connection(self) -> bool:
        """Test de connexion à la base de données."""
        try:
            from db_service.session import engine
            from sqlalchemy import text
            with engine.connect() as conn:
                result = conn.execute(text("SELECT version()"))
                version = result.fetchone()[0]
                logger.info(f"✅ PostgreSQL connecté: {version[:50]}...")
                return True
        except Exception as e:
            logger.error(f"❌ Base de données: {e}")
            self.detailed_errors["database"] = str(e)
            return False
    
    def test_external_dependencies(self) -> Dict[str, bool]:
        """Test des dépendances externes (Elasticsearch, Qdrant, Redis)."""
        dependencies = {}
        
        # Test Elasticsearch
        try:
            import httpx
            response = httpx.get(os.environ.get("ELASTICSEARCH_URL", "http://localhost:9200"), timeout=5)
            if response.status_code == 200:
                logger.info("✅ Elasticsearch accessible")
                dependencies["elasticsearch"] = True
            else:
                logger.warning("⚠️ Elasticsearch non accessible")
                dependencies["elasticsearch"] = False
        except Exception as e:
            logger.warning(f"⚠️ Elasticsearch: {e}")
            dependencies["elasticsearch"] = False
        
        # Test Qdrant
        try:
            import httpx
            response = httpx.get(f"{os.environ.get('QDRANT_URL', 'http://localhost:6333')}/collections", timeout=5)
            logger.info("✅ Qdrant accessible")
            dependencies["qdrant"] = True
        except Exception as e:
            logger.warning(f"⚠️ Qdrant: {e}")
            dependencies["qdrant"] = False
        
        # Test Redis
        try:
            import redis
            r = redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379"))
            r.ping()
            logger.info("✅ Redis accessible")
            dependencies["redis"] = True
        except Exception as e:
            logger.warning(f"⚠️ Redis: {e}")
            dependencies["redis"] = False
        
        return dependencies
    
    def test_service_imports(self) -> Dict[str, bool]:
        """Test d'import de tous les services avec fallbacks."""
        services_config = {
            "user_service": ["user_service.main", "user_service"],
            "db_service": ["db_service.main", "db_service", "db_service.session"],
            "sync_service": ["sync_service.main", "sync_service"],
            "enrichment_service": ["enrichment_service.main", "enrichment_service"],
            "search_service": ["search_service.main", "search_service"],
            "conversation_service": ["conversation_service.main", "conversation_service"]
        }
        
        results = {}
        for service_name, module_paths in services_config.items():
            imported = False
            last_error = None
            
            for module_path in module_paths:
                try:
                    __import__(module_path)
                    logger.info(f"✅ {service_name}: Import OK ({module_path})")
                    results[service_name] = True
                    imported = True
                    break
                except Exception as e:
                    last_error = str(e)
                    continue
            
            if not imported:
                logger.error(f"❌ {service_name}: {last_error}")
                results[service_name] = False
                self.detailed_errors[service_name] = last_error
        
        return results
    
    def load_service_router(self, app: FastAPI, service_name: str, router_path: str, prefix: str):
        """Charge et enregistre un router avec diagnostics étendus."""
        try:
            # Import dynamique du router
            module = __import__(router_path, fromlist=["router"])
            router = getattr(module, "router", None)
            
            if router:
                # Enregistrer le router
                app.include_router(router, prefix=prefix, tags=[service_name])
                routes_count = len(router.routes) if hasattr(router, 'routes') else 0
                
                # Lister les routes pour debug
                if hasattr(router, 'routes'):
                    routes_info = []
                    for route in router.routes:
                        methods = getattr(route, 'methods', {'GET'})
                        path = getattr(route, 'path', 'unknown')
                        routes_info.append(f"{list(methods)[0]} {prefix}{path}")
                    
                    logger.info(f"✅ {service_name}: {routes_count} routes chargées")
                    logger.debug(f"   Routes: {', '.join(routes_info[:3])}{'...' if len(routes_info) > 3 else ''}")
                
                self.services_status[service_name] = {
                    "status": "ok", 
                    "routes": routes_count, 
                    "prefix": prefix,
                    "routes_detail": routes_info if hasattr(router, 'routes') else []
                }
                return True
            else:
                logger.error(f"❌ {service_name}: Pas de router trouvé dans {router_path}")
                self.services_status[service_name] = {"status": "error", "error": "Pas de router"}
                return False
                
        except Exception as e:
            logger.error(f"❌ {service_name}: {str(e)}")
            self.services_status[service_name] = {"status": "error", "error": str(e)}
            self.detailed_errors[f"{service_name}_router"] = str(e)
            return False
    
    def get_comprehensive_report(self) -> Dict:
        """Génère un rapport complet pour debug."""
        ok_services = [name for name, status in self.services_status.items() 
                      if status.get("status") == "ok"]
        
        total_routes = sum(status.get("routes", 0) for status in self.services_status.values() 
                          if status.get("status") == "ok")
        
        return {
            "summary": {
                "services_loaded": len(ok_services),
                "total_services": len(self.services_status),
                "total_routes": total_routes,
                "status": "ready" if len(ok_services) >= 3 else "degraded"
            },
            "services": self.services_status,
            "errors": self.detailed_errors if self.detailed_errors else None,
            "environment": os.environ.get("ENVIRONMENT"),
            "database_url_set": bool(os.environ.get("DATABASE_URL")),
            "openai_key_set": bool(os.environ.get("OPENAI_API_KEY", "").startswith("sk-"))
        }

def create_local_app():
    """Créer l'application FastAPI pour tests locaux."""
    
    # Configuration environnement local
    setup_local_environment()
    
    app = FastAPI(
        title="Harena Finance Platform - Local Dev",
        description="Version de développement avec diagnostics étendus",
        version="1.0.0-dev",
        debug=True
    )

    # CORS permissif pour développement
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    tester = LocalServiceTester()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Gestionnaire du cycle de vie de l'application."""
        logger.info("🚀 Démarrage Harena Finance Platform - MODE DÉVELOPPEMENT")
        
        # Tests préliminaires
        logger.info("🔍 Tests des dépendances...")
        
        # Test DB critique
        if not tester.test_database_connection():
            logger.error("💥 ARRÊT: Base de données non accessible")
            logger.info("💡 Vérifiez que PostgreSQL est démarré et accessible")
            # En mode dev, on continue quand même pour voir les autres erreurs
        
        # Test dépendances externes (non bloquant)
        dependencies = tester.test_external_dependencies()
        missing_deps = [name for name, status in dependencies.items() if not status]
        if missing_deps:
            logger.warning(f"⚠️ Dépendances manquantes: {', '.join(missing_deps)}")
            logger.info("💡 Certaines fonctionnalités seront limitées")
        
        # Test imports des services
        logger.info("📦 Test des imports de services...")
        import_results = tester.test_service_imports()
        failed_imports = [name for name, status in import_results.items() if not status]
        if failed_imports:
            logger.warning(f"⚠️ Services non importables: {', '.join(failed_imports)}")
        
        # Chargement des routers avec fallbacks adaptés
        logger.info("📋 Chargement des routes des services...")
        
        service_routers = [
            ("user_service", ["user_service.api.endpoints.users"], "/api/v1/users"),
            ("sync_service", ["sync_service.api.router", "sync_service.api.routes"], "/api/v1/sync"),
            ("enrichment_service", ["enrichment_service.api.routes"], "/api/v1/enrichment"),
            ("search_service", ["search_service.api.routes", "search_service.routes"], "/api/v1/search"),
            ("conversation_service", ["conversation_service.api.routes"], "/api/v1/conversation"),
        ]
        
        successful = 0
        for service_name, router_paths, prefix in service_routers:
            router_loaded = False
            
            for router_path in router_paths:
                if tester.load_service_router(app, service_name, router_path, prefix):
                    successful += 1
                    router_loaded = True
                    break
            
            if not router_loaded:
                logger.warning(f"⚠️ {service_name}: Aucun router trouvé dans {router_paths}")
        
        # Fallback spécial pour sync_service avec modules individuels
        if "sync_service" not in [s for s, status in tester.services_status.items() if status.get("status") == "ok"]:
            logger.info("🔄 Tentative de chargement des modules sync individuels...")
            sync_modules = [
                ("sync_transactions", "sync_service.api.endpoints.transactions", "/api/v1/transactions"),
                ("sync_accounts", "sync_service.api.endpoints.accounts", "/api/v1/accounts"),
                ("sync_categories", "sync_service.api.endpoints.categories", "/api/v1/categories"),
            ]
            
            for service_name, router_path, prefix in sync_modules:
                if tester.load_service_router(app, service_name, router_path, prefix):
                    successful += 1
        
        # Rapport final
        report = tester.get_comprehensive_report()
        logger.info(f"✅ Démarrage terminé: {successful} services chargés")
        logger.info(f"📊 Statut global: {report['summary']['status'].upper()}")
        
        if report['summary']['status'] == 'degraded':
            logger.warning("⚠️ Application en mode dégradé - Vérifiez les erreurs ci-dessus")
        
        yield  # Point de démarrage de l'application
        
        # Cleanup optionnel ici
        logger.info("🔄 Arrêt de l'application")

    app = FastAPI(
        title="Harena Finance Platform - Local Dev",
        description="Version de développement avec diagnostics étendus",
        version="1.0.0-dev",
        debug=True,
        lifespan=lifespan
    )

    # CORS permissif pour développement
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    async def startup():
        logger.info("🚀 Démarrage Harena Finance Platform - MODE DÉVELOPPEMENT")
        
        # Tests préliminaires
        logger.info("🔍 Tests des dépendances...")
        
        # Test DB critique
        if not tester.test_database_connection():
            logger.error("💥 ARRÊT: Base de données non accessible")
            logger.info("💡 Vérifiez que PostgreSQL est démarré et accessible")
            # En mode dev, on continue quand même pour voir les autres erreurs
        
        # Test dépendances externes (non bloquant)
        dependencies = tester.test_external_dependencies()
        missing_deps = [name for name, status in dependencies.items() if not status]
        if missing_deps:
            logger.warning(f"⚠️ Dépendances manquantes: {', '.join(missing_deps)}")
            logger.info("💡 Certaines fonctionnalités seront limitées")
        
        # Test imports des services
        logger.info("📦 Test des imports de services...")
        import_results = tester.test_service_imports()
        failed_imports = [name for name, status in import_results.items() if not status]
        if failed_imports:
            logger.warning(f"⚠️ Services non importables: {', '.join(failed_imports)}")
        
        # Chargement des routers avec fallbacks adaptés
        logger.info("📋 Chargement des routes des services...")
        
        service_routers = [
            ("user_service", ["user_service.api.endpoints.users"], "/api/v1/users"),
            ("sync_service", ["sync_service.api.router", "sync_service.api.routes"], "/api/v1/sync"),
            ("enrichment_service", ["enrichment_service.api.routes"], "/api/v1/enrichment"),
            ("search_service", ["search_service.api.routes", "search_service.routes"], "/api/v1/search"),
            ("conversation_service", ["conversation_service.api.routes"], "/api/v1/conversation"),
        ]
        
        successful = 0
        for service_name, router_paths, prefix in service_routers:
            router_loaded = False
            
            for router_path in router_paths:
                if tester.load_service_router(app, service_name, router_path, prefix):
                    successful += 1
                    router_loaded = True
                    break
            
            if not router_loaded:
                logger.warning(f"⚠️ {service_name}: Aucun router trouvé dans {router_paths}")
        
        # Fallback spécial pour sync_service avec modules individuels
        if "sync_service" not in [s for s, status in tester.services_status.items() if status.get("status") == "ok"]:
            logger.info("🔄 Tentative de chargement des modules sync individuels...")
            sync_modules = [
                ("sync_transactions", "sync_service.api.endpoints.transactions", "/api/v1/transactions"),
                ("sync_accounts", "sync_service.api.endpoints.accounts", "/api/v1/accounts"),
                ("sync_categories", "sync_service.api.endpoints.categories", "/api/v1/categories"),
            ]
            
            for service_name, router_path, prefix in sync_modules:
                if tester.load_service_router(app, service_name, router_path, prefix):
                    successful += 1
        
        # Rapport final
        report = tester.get_comprehensive_report()
        logger.info(f"✅ Démarrage terminé: {successful} services chargés")
        logger.info(f"📊 Statut global: {report['summary']['status'].upper()}")
        
        if report['summary']['status'] == 'degraded':
            logger.warning("⚠️ Application en mode dégradé - Vérifiez les erreurs ci-dessus")

    @app.get("/health")
    async def health():
        """Health check pour développement."""
        return tester.get_comprehensive_report()

    @app.get("/debug")
    async def debug():
        """Endpoint de debug détaillé."""
        return {
            "services": tester.services_status,
            "errors": tester.detailed_errors,
            "environment_vars": {
                "DATABASE_URL": "***" if os.environ.get("DATABASE_URL") else None,
                "OPENAI_API_KEY": "***" if os.environ.get("OPENAI_API_KEY") else None,
                "ELASTICSEARCH_URL": os.environ.get("ELASTICSEARCH_URL"),
                "QDRANT_URL": os.environ.get("QDRANT_URL"),
                "REDIS_URL": os.environ.get("REDIS_URL"),
            },
            "python_path": sys.path[:3],
            "current_dir": str(current_dir)
        }

    @app.get("/")
    async def root():
        """Page d'accueil développement."""
        report = tester.get_comprehensive_report()
        return {
            "message": "🏦 Harena Finance Platform - MODE DÉVELOPPEMENT",
            "status": report['summary']['status'],
            "services_loaded": f"{report['summary']['services_loaded']}/{report['summary']['total_services']}",
            "total_routes": report['summary']['total_routes'],
            "endpoints": {
                "/health": "Rapport complet de santé",
                "/debug": "Informations de debug détaillées",
                "/docs": "Documentation API Swagger",
                "/api/v1/*": "APIs des services métier"
            },
            "tips": [
                "Utilisez /debug pour voir les erreurs détaillées",
                "Vérifiez /docs pour l'API interactive",
                "Consultez les logs de la console pour plus d'infos"
            ]
        }

    return app

# Créer l'app
app = create_local_app()

def run_dev_server():
    """Lance le serveur de développement avec hot reload."""
    logger.info("🔥 Lancement du serveur de développement avec hot reload")
    logger.info("📡 Accès: http://localhost:8000")
    logger.info("📚 Docs: http://localhost:8000/docs")
    logger.info("🔍 Debug: http://localhost:8000/debug")
    
    uvicorn.run(
        "local_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=[str(current_dir)],
        log_level="info"
    )

if __name__ == "__main__":
    run_dev_server()