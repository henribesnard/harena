"""
Application Harena pour développement local.
Structure EXACTEMENT identique à heroku_app.py avec configurations locales.
"""

import logging
import os
import sys
from pathlib import Path
from fastapi import FastAPI
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

# PAS de configuration par défaut - TOUT vient du .env
# Les variables sont chargées par load_dotenv() uniquement

# Ajouter le répertoire courant au path
current_dir = Path(__file__).parent.absolute()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

class ServiceLoader:
    """Chargeur de services - COPIE EXACTE de heroku_app.py"""
    
    def __init__(self):
        self.services_status = {}
        self.search_service_initialized = False
        self.search_service_error = None
        self.conversation_service_initialized = False
        self.conversation_service_error = None
    
    async def initialize_search_service(self, app: FastAPI):
        """Initialise le search_service - COPIE EXACTE de heroku_app.py"""
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
    
    async def initialize_conversation_service(self, app: FastAPI):
        """✅ Initialise le conversation_service - VERSION CORRIGÉE"""
        logger.info("🤖 Initialisation du conversation_service...")
        
        try:
            # Vérifier DEEPSEEK_API_KEY
            deepseek_key = os.environ.get("DEEPSEEK_API_KEY")
            if not deepseek_key:
                raise ValueError("DEEPSEEK_API_KEY n'est pas configurée")
            
            logger.info(f"🔑 DEEPSEEK_API_KEY configurée: {deepseek_key[:20]}...")
            
            # Import progressif et sécurisé
            from config_service.config import settings
            from conversation_service.clients.deepseek_client import DeepSeekClient
            
            # Validation de la configuration
            logger.info("⚙️ Validation de la configuration...")
            validation = settings.validate_configuration()
            if not validation["valid"]:
                raise ValueError(f"Configuration invalide: {validation['errors']}")
            
            if validation["warnings"]:
                logger.warning(f"⚠️ Avertissements: {validation['warnings']}")
            
            # Test de connexion DeepSeek
            logger.info("🔍 Test de connexion DeepSeek...")
            deepseek_client = DeepSeekClient()
            
            # ✅ Utiliser la méthode correcte pour tester DeepSeek
            try:
                # Test simple avec une requête basique
                test_response = await deepseek_client.chat_completion(
                    messages=[{"role": "user", "content": "test"}],
                    model="deepseek-chat"
                )
                logger.info("✅ DeepSeek connecté et fonctionnel")
            except Exception as e:
                logger.warning(f"⚠️ Test DeepSeek échoué mais continuons: {e}")
                # Ne pas bloquer l'initialisation pour un simple test
            
            # ✅ Test de l'agent de classification - VERSION CORRIGÉE
            logger.info("🤖 Test de l'agent de classification...")
            from conversation_service.agents.intent_classifier import IntentClassifier
            
            # Créer une instance de l'agent au lieu d'utiliser une fonction inexistante
            intent_agent = IntentClassifier()
            
            # Test simple pour valider l'initialisation
            test_result = await intent_agent.classify_intent("test", "system")
            
            # ✅ Utilisation correcte de la méthode get_agent_metrics()
            agent_metrics = intent_agent.get_agent_metrics()
            logger.info(f"✅ Agent initialisé - Classifications: {agent_metrics['total_classifications']}")
            logger.info(f"🎯 Seuil confiance: {settings.MIN_CONFIDENCE_THRESHOLD}")
            
            # Mettre les composants dans app.state
            app.state.conversation_service_initialized = True
            app.state.deepseek_client = deepseek_client
            app.state.intent_classifier = intent_agent
            app.state.conversation_initialization_error = None
            
            self.conversation_service_initialized = True
            self.conversation_service_error = None
            
            logger.info("🎉 Conversation Service complètement initialisé!")
            return True
            
        except Exception as e:
            error_msg = f"Erreur initialisation conversation_service: {str(e)}"
            logger.error(f"❌ {error_msg}")
            
            # Marquer l'échec dans app.state
            app.state.conversation_service_initialized = False
            app.state.deepseek_client = None
            app.state.intent_classifier = None
            app.state.conversation_initialization_error = error_msg
            
            self.conversation_service_initialized = False
            self.conversation_service_error = error_msg
            return False
    
    def load_service_router(self, app: FastAPI, service_name: str, router_path: str, prefix: str):
        """Charge et enregistre un router de service - COPIE EXACTE de heroku_app.py"""
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
        """Vérifie rapidement la santé d'un service - COPIE EXACTE de heroku_app.py"""
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
    """Créer l'application FastAPI principale - STRUCTURE EXACTE de heroku_app.py"""
    
    app = FastAPI(
        title="Harena Finance Platform - Local Dev",
        description="Plateforme de gestion financière avec IA - Version développement",
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

        # 2. Sync Service - modules principaux (EXACTEMENT comme heroku_app.py)
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

        # 3. ✅ ENRICHMENT SERVICE - VERSION ELASTICSEARCH UNIQUEMENT
        logger.info("🔍 Chargement enrichment_service (Elasticsearch uniquement)...")
        try:
            # Vérifier BONSAI_URL pour enrichment_service
            bonsai_url = os.environ.get("BONSAI_URL")
            if not bonsai_url:
                logger.warning("⚠️ BONSAI_URL non configurée - enrichment_service sera en mode dégradé")
                enrichment_elasticsearch_available = False
            else:
                logger.info(f"📡 BONSAI_URL configurée pour enrichment: {bonsai_url[:50]}...")
                enrichment_elasticsearch_available = True
            
            # Charger les routes enrichment_service
            from enrichment_service.api.routes import router as enrichment_router
            app.include_router(enrichment_router, prefix="/api/v1/enrichment", tags=["enrichment"])
            routes_count = len(enrichment_router.routes) if hasattr(enrichment_router, 'routes') else 0
            
            if enrichment_elasticsearch_available:
                logger.info(f"✅ enrichment_service: {routes_count} routes sur /api/v1/enrichment (Elasticsearch configuré)")
                loader.services_status["enrichment_service"] = {
                    "status": "ok", 
                    "routes": routes_count, 
                    "prefix": "/api/v1/enrichment",
                    "architecture": "elasticsearch_only",
                    "version": "2.0.0-elasticsearch",
                    "elasticsearch_available": True
                }
            else:
                logger.warning(f"⚠️ enrichment_service: {routes_count} routes chargées SANS Elasticsearch")
                loader.services_status["enrichment_service"] = {
                    "status": "degraded", 
                    "routes": routes_count, 
                    "prefix": "/api/v1/enrichment",
                    "architecture": "elasticsearch_only",
                    "version": "2.0.0-elasticsearch",
                    "elasticsearch_available": False,
                    "error": "BONSAI_URL not configured"
                }
                
        except Exception as e:
            logger.error(f"❌ Enrichment Service: {e}")
            loader.services_status["enrichment_service"] = {
                "status": "error", 
                "error": str(e),
                "architecture": "elasticsearch_only",
                "version": "2.0.0-elasticsearch"
            }

        # 4. ✅ Search Service - EXACTEMENT COMME HEROKU_APP.PY
        logger.info("🔍 Chargement et initialisation du search_service...")
        try:
            # D'abord initialiser les composants Elasticsearch
            search_init_success = await loader.initialize_search_service(app)
            
            # Ensuite charger les routes
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
                        "initialized": True,
                        "architecture": "simplified_unified"
                    }
                else:
                    logger.warning(f"⚠️ search_service: {routes_count} routes chargées SANS initialisation")
                    loader.services_status["search_service"] = {
                        "status": "degraded", 
                        "routes": routes_count, 
                        "prefix": "/api/v1/search",
                        "initialized": False,
                        "error": loader.search_service_error,
                        "architecture": "simplified_unified"
                    }
                    
            except ImportError as e:
                logger.error(f"❌ search_service: Impossible de charger les routes - {str(e)}")
                loader.services_status["search_service"] = {
                    "status": "error", 
                    "error": f"Routes import failed: {str(e)}",
                    "architecture": "simplified_unified"
                }
                    
        except Exception as e:
            logger.error(f"❌ search_service: Erreur générale - {str(e)}")
            loader.services_status["search_service"] = {
                "status": "error", 
                "error": str(e),
                "architecture": "simplified_unified"
            }

        # 5. ✅ CONVERSATION SERVICE - VERSION CORRIGÉE
        logger.info("🤖 Chargement et initialisation du conversation_service...")
        try:
            # D'abord initialiser les composants DeepSeek
            conversation_init_success = await loader.initialize_conversation_service(app)
            
            # Ensuite charger les routes avec gestion des imports circulaires
            try:
                # ✅ Import sécurisé avec gestion d'erreurs détaillée
                logger.info("📦 Tentative d'import des routes conversation_service...")
                
                # Tentative 1: Import direct
                try:
                    from conversation_service.api.routes import router as conversation_router
                    router_imported = True
                    import_method = "direct"
                except Exception as e1:
                    logger.warning(f"⚠️ Import direct échoué: {str(e1)[:100]}...")
                    
                    # Tentative 2: Import alternatif
                    try:
                        import conversation_service.api
                        conversation_router = getattr(conversation_service.api, 'router', None)
                        if conversation_router:
                            router_imported = True
                            import_method = "alternative"
                        else:
                            raise AttributeError("Pas de router trouvé")
                    except Exception as e2:
                        logger.warning(f"⚠️ Import alternatif échoué: {str(e2)[:100]}...")
                        router_imported = False
                        import_method = "failed"
                
                if router_imported:
                    app.include_router(conversation_router, prefix="/api/v1/conversation")
                    routes_count = len(conversation_router.routes) if hasattr(conversation_router, 'routes') else 0
                    
                    if conversation_init_success:
                        logger.info(f"✅ conversation_service: {routes_count} routes sur /api/v1/conversation (AVEC initialisation - {import_method})")
                        loader.services_status["conversation_service"] = {
                            "status": "ok", 
                            "routes": routes_count, 
                            "prefix": "/api/v1/conversation",
                            "initialized": True,
                            "architecture": "mvp_intent_classifier",
                            "model": "deepseek-chat",
                            "import_method": import_method
                        }
                    else:
                        logger.warning(f"⚠️ conversation_service: {routes_count} routes chargées SANS initialisation complète")
                        loader.services_status["conversation_service"] = {
                            "status": "degraded", 
                            "routes": routes_count, 
                            "prefix": "/api/v1/conversation",
                            "initialized": False,
                            "error": loader.conversation_service_error,
                            "architecture": "mvp_intent_classifier",
                            "model": "deepseek-chat",
                            "import_method": import_method
                        }
                else:
                    raise ImportError("Toutes les tentatives d'import ont échoué")
                    
            except Exception as e:
                logger.error(f"❌ conversation_service: Impossible de charger les routes - {str(e)}")
                loader.services_status["conversation_service"] = {
                    "status": "error", 
                    "error": f"Routes import failed: {str(e)}",
                    "architecture": "mvp_intent_classifier"
                }
                    
        except Exception as e:
            logger.error(f"❌ conversation_service: Erreur générale - {str(e)}")
            loader.services_status["conversation_service"] = {
                "status": "error", 
                "error": str(e),
                "architecture": "mvp_intent_classifier"
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
        
        logger.info("🎉 Plateforme Harena complètement déployée - LOCAL DEV!")

    @app.get("/health")
    async def health():
        """Health check global - EXACTEMENT COMME HEROKU_APP.PY"""
        ok_services = [name for name, status in loader.services_status.items() 
                      if status.get("status") == "ok"]
        degraded_services = [name for name, status in loader.services_status.items() 
                           if status.get("status") == "degraded"]
        
        # Détails spéciaux pour search_service, conversation_service et enrichment_service
        search_status = loader.services_status.get("search_service", {})
        conversation_status = loader.services_status.get("conversation_service", {})
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
            "conversation_service": {
                "status": conversation_status.get("status"),
                "initialized": conversation_status.get("initialized", False),
                "error": conversation_status.get("error"),
                "architecture": conversation_status.get("architecture"),
                "model": conversation_status.get("model")
            },
            "enrichment_service": {
                "status": enrichment_status.get("status"),
                "architecture": enrichment_status.get("architecture"),
                "version": enrichment_status.get("version"),
                "elasticsearch_available": enrichment_status.get("elasticsearch_available", False),
                "error": enrichment_status.get("error")
            }
        }

    @app.get("/status")
    async def status():
        """Statut détaillé - EXACTEMENT COMME HEROKU_APP.PY"""
        return {
            "platform": "Harena Finance",
            "services": loader.services_status,
            "environment": os.environ.get("ENVIRONMENT", "development"),
            "search_service_details": {
                "initialized": loader.search_service_initialized,
                "error": loader.search_service_error,
                "architecture": "simplified_unified"
            },
            "conversation_service_details": {
                "initialized": loader.conversation_service_initialized,
                "error": loader.conversation_service_error,
                "architecture": "mvp_intent_classifier",
                "model": "deepseek-chat"
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
        """Page d'accueil - VERSION LOCAL DEV"""
        return {
            "message": "🏦 Harena Finance Platform - LOCAL DEVELOPMENT",
            "version": "1.0.0-dev",
            "services_available": [
                "user_service - Gestion utilisateurs",
                "sync_service - Synchronisation Bridge API", 
                "enrichment_service - Enrichissement Elasticsearch (v2.0)",
                "search_service - Recherche lexicale (Architecture simplifiée)",
                "conversation_service - Assistant IA avec DeepSeek (MVP)"
            ],
            "services_coming_soon": [
                "conversation_service v2 - Assistant IA avec AutoGen + équipes d'agents"
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
                "/api/v1/search/*": "Recherche lexicale (Architecture unifiée)",
                "/api/v1/conversation/*": "Assistant IA conversationnel (DeepSeek MVP)"
            },
            "development_mode": {
                "hot_reload": True,
                "debug_logs": True,
                "local_services": ["PostgreSQL", "Redis", "Elasticsearch (requis pour enrichment + search)"],
                "docs_url": "http://localhost:8000/docs"
            },
            "architecture_updates": {
                "enrichment_service": {
                    "version": "2.0.0-elasticsearch",
                    "changes": [
                        "Suppression complète de Qdrant et embeddings",
                        "Architecture Elasticsearch uniquement",
                        "Nouveaux endpoints: /api/v1/enrichment/elasticsearch/*",
                        "Performance optimisée pour l'indexation bulk",
                        "Simplification drastique du code"
                    ]
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
    
    uvicorn.run(
        "local_app:app", 
        host="0.0.0.0", 
        port=8000,
        reload=True,
        log_level="info"
    )