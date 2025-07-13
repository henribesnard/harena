"""
Application Harena simplifiée pour Heroku.
Utilise les main.py de chaque service au lieu de dupliquer les diagnostics.
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
    """Chargeur de services simplifié."""
    
    def __init__(self):
        self.services_status = {}
    
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
        """Vérifie rapidement la santé d'un service via son main.py."""
        try:
            # Import du main du service
            main_module = __import__(f"{module_path}.main", fromlist=["app"])
            
            # Vérifier l'existence de l'app
            if hasattr(main_module, "app") or hasattr(main_module, "create_app"):
                logger.info(f"✅ {service_name}: Module principal OK")
                return True
            else:
                logger.warning(f"⚠️ {service_name}: Pas d'app FastAPI trouvée")
                return False
                
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
        
        # Vérifier santé des services (appel à leur main.py)
        services_health = [
            ("user_service", "user_service"),
            ("db_service", "db_service"),
            ("sync_service", "sync_service"),
            ("enrichment_service", "enrichment_service"),
            ("search_service", "search_service"),
            ("conversation_service", "conversation_service")
        ]
        
        for service_name, module_path in services_health:
            loader.check_service_health(service_name, module_path)
        
        # Charger les routers des services
        logger.info("📋 Chargement des routes des services...")
        
        # Définition des routers
        service_routers = [
            ("user_service", "user_service.api.endpoints.users", "/api/v1/users"),
            ("sync_service", "sync_service.api.routes", "/api/v1/sync"),
            ("enrichment_service", "enrichment_service.api.routes", "/api/v1/enrichment"),
            ("search_service", "search_service.api.routes", "/api/v1/search"),
            ("conversation_service", "conversation_service.api.routes", "/api/v1/conversation"),
        ]
        
        # Chargement avec fallback pour sync_service (modules multiples)
        successful = 0
        for service_name, router_path, prefix in service_routers:
            if loader.load_service_router(app, service_name, router_path, prefix):
                successful += 1
        
        # Gérer les modules multiples du sync_service si le principal échoue
        if "sync_service" not in [s for s, status in loader.services_status.items() if status.get("status") == "ok"]:
            sync_modules = [
                ("sync_transactions", "sync_service.api.endpoints.transactions", "/api/v1/transactions"),
                ("sync_accounts", "sync_service.api.endpoints.accounts", "/api/v1/accounts"),
                ("sync_categories", "sync_service.api.endpoints.categories", "/api/v1/categories"),
            ]
            
            for service_name, router_path, prefix in sync_modules:
                if loader.load_service_router(app, service_name, router_path, prefix):
                    successful += 1
        
        logger.info(f"✅ Démarrage terminé: {successful} services/modules chargés")

    @app.get("/health")
    async def health():
        """Health check global."""
        ok_services = [name for name, status in loader.services_status.items() 
                      if status.get("status") == "ok"]
        
        return {
            "status": "healthy" if ok_services else "degraded",
            "services_ok": len(ok_services),
            "total_services": len(loader.services_status),
            "services": list(ok_services)
        }

    @app.get("/status")
    async def status():
        """Statut détaillé."""
        return {
            "platform": "Harena Finance",
            "services": loader.services_status,
            "environment": os.environ.get("ENVIRONMENT", "production")
        }

    @app.get("/")
    async def root():
        """Page d'accueil."""
        return {
            "message": "🏦 Harena Finance Platform",
            "version": "1.0.0",
            "endpoints": {
                "/health": "Contrôle santé",
                "/status": "Statut des services",
                "/api/v1/*": "APIs métier"
            }
        }

    return app

# Créer l'app
app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("heroku_app:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))