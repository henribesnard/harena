"""
Application Harena pour tests locaux avant déploiement.
Version synchronisée avec heroku_app.py avec support conversation_service Redis + TinyBERT.
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
logger = logging.getLogger("harena_local")

# Fix DATABASE_URL pour tests locaux (si PostgreSQL format)
DATABASE_URL = os.environ.get("DATABASE_URL")
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    os.environ["DATABASE_URL"] = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Configuration environnement local par défaut
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
        
        # Search Service - Elasticsearch
        "BONSAI_URL": "http://localhost:9200",
        "ELASTICSEARCH_URL": "http://localhost:9200",
        
        # Conversation Service - Redis + TinyBERT
        "REDIS_URL": "redis://localhost:6379",
        "REDIS_CACHE_ENABLED": "true",
        "REDIS_CACHE_PREFIX": "conversation_service_local",
        "REDIS_DB": "0",
        "REDIS_PASSWORD": "",
        
        # TinyBERT Configuration
        "TINYBERT_MODEL_NAME": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "EMBEDDING_CACHE_SIZE": "1000",
        "MIN_CONFIDENCE_THRESHOLD": "0.7",
        
        # Cache TTL Configuration
        "CACHE_TTL_INTENT": "300",     # 5min
        "CACHE_TTL_ENTITY": "180",     # 3min  
        "CACHE_TTL_RESPONSE": "60",    # 1min
        "MEMORY_CACHE_SIZE": "500",
        "MEMORY_CACHE_TTL": "300",
        
        # Performance optimisations
        "ENABLE_ASYNC_PIPELINE": "true",
        "PARALLEL_PROCESSING_ENABLED": "true",
        "THREAD_POOL_SIZE": "4",
        "MAX_CONCURRENT_REQUESTS": "50",
        
        # Circuit breaker
        "CIRCUIT_BREAKER_ENABLED": "true",
        "CIRCUIT_BREAKER_FAILURE_THRESHOLD": "5",
        "CIRCUIT_BREAKER_TIMEOUT": "60",
        
        # Autres services (optionnels pour tests)
        "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", "sk-test-key"),
        "BRIDGE_BASE_URL": "https://sync.bankin.com",
        "BRIDGE_CLIENT_ID": os.environ.get("BRIDGE_CLIENT_ID", ""),
        "BRIDGE_CLIENT_SECRET": os.environ.get("BRIDGE_CLIENT_SECRET", ""),
    }
    
    for key, value in default_env.items():
        if not os.environ.get(key):
            os.environ[key] = value
    
    logger.info("✅ Environnement local configuré pour conversation_service Redis + TinyBERT")

# Ajouter le répertoire courant au path
current_dir = Path(__file__).parent.absolute()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

class ServiceLoader:
    """Chargeur de services simplifié inspiré de heroku_app.py."""
    
    def __init__(self):
        self.services_status = {}
        self.search_service_initialized = False
        self.search_service_error = None
        self.conversation_service_initialized = False
        self.conversation_service_error = None
    
    async def initialize_search_service(self, app: FastAPI):
        """Initialise le search_service avec la nouvelle architecture simplifiée."""
        logger.info("🔍 Initialisation du search_service...")
        
        try:
            # Vérifier BONSAI_URL ou ELASTICSEARCH_URL
            elasticsearch_url = os.environ.get("BONSAI_URL") or os.environ.get("ELASTICSEARCH_URL")
            if not elasticsearch_url:
                raise ValueError("BONSAI_URL ou ELASTICSEARCH_URL n'est pas configurée")
            
            logger.info(f"📡 Elasticsearch URL configurée: {elasticsearch_url}")
            
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
        """Initialise le conversation_service avec Redis + TinyBERT."""
        logger.info("🤖 Initialisation du conversation_service avec Redis + TinyBERT...")
        
        try:
            # Vérifier REDIS_URL
            redis_url = os.environ.get("REDIS_URL")
            if not redis_url:
                raise ValueError("REDIS_URL n'est pas configurée")
            
            logger.info(f"💾 REDIS_URL configurée: {redis_url}")
            
            # Import progressif et sécurisé du conversation service
            logger.info("📦 Import des modules conversation_service...")
            
            # Import basique de la configuration
            try:
                from conversation_service.config import settings
                logger.info("✅ Configuration conversation_service importée")
            except Exception as e:
                logger.error(f"❌ Erreur import configuration: {e}")
                raise
            
            # Test de validation configuration avec fallback
            try:
                logger.info("⚙️ Validation de la configuration conversation_service...")
                if hasattr(settings, 'validate_configuration'):
                    validation = settings.validate_configuration()
                    if not validation["valid"]:
                        raise ValueError(f"Configuration invalide: {validation['errors']}")
                    
                    if validation["warnings"]:
                        logger.warning(f"⚠️ Avertissements: {validation['warnings']}")
                else:
                    logger.info("ℹ️ Pas de méthode validate_configuration - continuons")
            except Exception as e:
                logger.warning(f"⚠️ Validation configuration échouée, mais continuons: {e}")
            
            # Test Redis avec retry et fallback gracieux
            logger.info("💾 Test de connexion Redis...")
            redis_available = False
            try:
                import redis
                r = redis.from_url(redis_url)
                r.ping()
                redis_available = True
                logger.info("✅ Redis accessible")
            except Exception as e:
                logger.warning(f"⚠️ Redis non accessible: {e} - Mode sans cache")
                redis_available = False
            
            # Test TinyBERT avec fallback gracieux  
            logger.info("🧠 Test d'initialisation TinyBERT...")
            tinybert_available = False
            try:
                # Test simple d'import sentence-transformers
                import sentence_transformers
                tinybert_available = True
                logger.info("✅ TinyBERT/sentence-transformers disponible")
            except ImportError:
                logger.warning("⚠️ sentence-transformers non installé - Mode sans embeddings")
                tinybert_available = False
            except Exception as e:
                logger.warning(f"⚠️ TinyBERT non disponible: {e} - Mode dégradé")
                tinybert_available = False
            
            # Mettre les composants dans app.state (mode dégradé OK)
            app.state.conversation_service_initialized = True
            app.state.redis_available = redis_available
            app.state.tinybert_available = tinybert_available
            app.state.conversation_initialization_error = None
            
            self.conversation_service_initialized = True
            self.conversation_service_error = None
            
            status = "complète" if (redis_available and tinybert_available) else "dégradée"
            logger.info(f"🎉 Conversation Service initialisé en mode {status}!")
            return True
            
        except Exception as e:
            error_msg = f"Erreur initialisation conversation_service: {str(e)}"
            logger.error(f"❌ {error_msg}")
            
            # Marquer l'échec dans app.state
            app.state.conversation_service_initialized = False
            app.state.redis_available = False
            app.state.tinybert_available = False
            app.state.conversation_initialization_error = error_msg
            
            self.conversation_service_initialized = False
            self.conversation_service_error = error_msg
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
        """Vérifie rapidement la santé d'un service."""
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
    
    # Configuration environnement local
    setup_local_environment()
    
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
        logger.info("🚀 Démarrage Harena Finance Platform - MODE DÉVELOPPEMENT")
        
        # Test DB critique avec gestion d'erreurs d'encoding
        try:
            logger.info("🔍 Test de connexion base de données...")
            from db_service.session import engine
            from sqlalchemy import text
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("✅ Base de données connectée")
        except Exception as e:
            error_str = str(e)
            if "'utf-8' codec can't decode" in error_str:
                logger.error("❌ DB critique: Problème d'encoding détecté")
                logger.info("💡 Vérifiez l'encoding de vos fichiers de configuration")
                logger.info("💡 Ou vérifiez que PostgreSQL est démarré avec la bonne locale")
            else:
                logger.error(f"❌ DB critique: {error_str}")
                logger.info("💡 Vérifiez que PostgreSQL est démarré et accessible")
            # En mode dev, on continue quand même pour voir les autres erreurs
        
        # Vérifier santé des services existants
        services_health = [
            ("user_service", "user_service"),
            ("db_service", "db_service"),
            ("sync_service", "sync_service"),
            ("enrichment_service", "enrichment_service"),
        ]
        
        for service_name, module_path in services_health:
            try:
                loader.check_service_health(service_name, module_path)
            except Exception as e:
                error_str = str(e)
                if "'utf-8' codec can't decode" in error_str:
                    logger.warning(f"⚠️ {service_name}: Problème d'encoding détecté - fichier corrompu ?")
                else:
                    logger.warning(f"⚠️ {service_name}: {error_str}")
                # Continue malgré les erreurs en mode dev
        
        # Charger les routers des services
        logger.info("📋 Chargement des routes des services...")
        
        # 1. User Service avec gestion d'erreurs d'encoding
        try:
            from user_service.api.endpoints.users import router as user_router
            app.include_router(user_router, prefix="/api/v1/users", tags=["users"])
            routes_count = len(user_router.routes) if hasattr(user_router, 'routes') else 0
            logger.info(f"✅ user_service: {routes_count} routes sur /api/v1/users")
            loader.services_status["user_service"] = {"status": "ok", "routes": routes_count, "prefix": "/api/v1/users"}
        except Exception as e:
            error_str = str(e)
            if "'utf-8' codec can't decode" in error_str:
                logger.error("❌ User Service: Problème d'encoding détecté dans les fichiers")
                logger.info("💡 Vérifiez l'encoding UTF-8 des fichiers Python dans user_service/")
            else:
                logger.error(f"❌ User Service: {error_str}")
            loader.services_status["user_service"] = {"status": "error", "error": error_str[:100] + "..."}

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
                error_str = str(e)
                if "'utf-8' codec can't decode" in error_str:
                    logger.error(f"❌ {module_path}: Problème d'encoding détecté")
                else:
                    logger.error(f"❌ {module_path}: {error_str}")
                loader.services_status[f"sync_{module_path.split('.')[-1]}"] = {"status": "error", "error": error_str[:100] + "..."}

        # 3. Enrichment Service avec gestion d'erreurs d'encoding
        try:
            from enrichment_service.api.routes import router as enrichment_router
            app.include_router(enrichment_router, prefix="/api/v1/enrichment", tags=["enrichment"])
            routes_count = len(enrichment_router.routes) if hasattr(enrichment_router, 'routes') else 0
            logger.info(f"✅ enrichment_service: {routes_count} routes sur /api/v1/enrichment")
            loader.services_status["enrichment_service"] = {"status": "ok", "routes": routes_count, "prefix": "/api/v1/enrichment"}
        except Exception as e:
            error_str = str(e)
            if "'utf-8' codec can't decode" in error_str:
                logger.error("❌ Enrichment Service: Problème d'encoding détecté")
            else:
                logger.error(f"❌ Enrichment Service: {error_str}")
            loader.services_status["enrichment_service"] = {"status": "error", "error": error_str[:100] + "..."}

        # 4. ✅ Search Service - AVEC NOUVELLE ARCHITECTURE SIMPLIFIÉE
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

        # 5. ✅ CONVERSATION SERVICE - NOUVEAU avec Redis + TinyBERT
        logger.info("🤖 Chargement et initialisation du conversation_service...")
        try:
            # D'abord initialiser les composants Redis + TinyBERT
            conversation_init_success = await loader.initialize_conversation_service(app)
            
            # Ensuite charger les routes avec gestion sécurisée des imports circulaires
            try:
                # Import sécurisé avec fallback 
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
                            "architecture": "redis_tinybert_classifier",
                            "cache": "redis_multi_level",
                            "model": "tiny_bert_multilingual",
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
                            "architecture": "redis_tinybert_classifier",
                            "cache": "redis_multi_level",
                            "model": "tiny_bert_multilingual",
                            "import_method": import_method
                        }
                else:
                    raise ImportError("Toutes les tentatives d'import ont échoué")
                    
            except Exception as e:
                logger.error(f"❌ conversation_service: Impossible de charger les routes - {str(e)[:100]}...")
                loader.services_status["conversation_service"] = {
                    "status": "error", 
                    "error": f"Routes import failed: {str(e)[:100]}...",
                    "architecture": "redis_tinybert_classifier",
                    "initialized": conversation_init_success  # Au moins la partie init peut avoir marché
                }
                    
        except Exception as e:
            logger.error(f"❌ conversation_service: Erreur générale - {str(e)[:100]}...")
            loader.services_status["conversation_service"] = {
                "status": "error", 
                "error": str(e)[:100] + "...",
                "architecture": "redis_tinybert_classifier"
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
        
        logger.info("🎉 Plateforme Harena complètement déployée avec Conversation Service Redis + TinyBERT!")

    @app.get("/health")
    async def health():
        """Health check global avec détails search_service et conversation_service."""
        ok_services = [name for name, status in loader.services_status.items() 
                      if status.get("status") == "ok"]
        degraded_services = [name for name, status in loader.services_status.items() 
                           if status.get("status") == "degraded"]
        
        # Détails spéciaux pour search_service et conversation_service
        search_status = loader.services_status.get("search_service", {})
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
            "conversation_service": {
                "status": conversation_status.get("status"),
                "initialized": conversation_status.get("initialized", False),
                "error": conversation_status.get("error"),
                "architecture": conversation_status.get("architecture"),
                "cache": conversation_status.get("cache"),
                "model": conversation_status.get("model")
            }
        }

    @app.get("/status")
    async def status():
        """Statut détaillé."""
        return {
            "platform": "Harena Finance",
            "environment": "development",
            "services": loader.services_status,
            "search_service_details": {
                "initialized": loader.search_service_initialized,
                "error": loader.search_service_error,
                "architecture": "simplified_unified"
            },
            "conversation_service_details": {
                "initialized": loader.conversation_service_initialized,
                "error": loader.conversation_service_error,
                "architecture": "redis_tinybert_classifier",
                "cache": "redis_multi_level",
                "model": "tiny_bert_multilingual"
            }
        }

    @app.get("/debug")
    async def debug():
        """Endpoint de debug détaillé pour développement."""
        return {
            "services": loader.services_status,
            "environment_vars": {
                "DATABASE_URL": "***" if os.environ.get("DATABASE_URL") else None,
                "REDIS_URL": "***" if os.environ.get("REDIS_URL") else None,
                "BONSAI_URL": "***" if os.environ.get("BONSAI_URL") else None,
                "ELASTICSEARCH_URL": os.environ.get("ELASTICSEARCH_URL"),
                "TINYBERT_MODEL_NAME": os.environ.get("TINYBERT_MODEL_NAME"),
                "REDIS_CACHE_ENABLED": os.environ.get("REDIS_CACHE_ENABLED"),
                "MIN_CONFIDENCE_THRESHOLD": os.environ.get("MIN_CONFIDENCE_THRESHOLD"),
            },
            "python_path": sys.path[:3],
            "current_dir": str(current_dir)
        }

    @app.get("/")
    async def root():
        """Page d'accueil."""
        return {
            "message": "🏦 Harena Finance Platform - DÉVELOPPEMENT",
            "version": "1.0.0-dev",
            "services_available": [
                "user_service - Gestion utilisateurs",
                "sync_service - Synchronisation Bridge API", 
                "enrichment_service - Enrichissement IA",
                "search_service - Recherche lexicale (Architecture simplifiée)",
                "conversation_service - Assistant IA avec Redis + TinyBERT"
            ],
            "endpoints": {
                "/health": "Contrôle santé",
                "/status": "Statut des services",
                "/debug": "Informations de debug",
                "/docs": "Documentation API Swagger",
                "/api/v1/users/*": "Gestion utilisateurs",
                "/api/v1/sync/*": "Synchronisation",
                "/api/v1/transactions/*": "Transactions",
                "/api/v1/accounts/*": "Comptes",
                "/api/v1/categories/*": "Catégories",
                "/api/v1/enrichment/*": "Enrichissement IA",
                "/api/v1/search/*": "Recherche lexicale (Architecture unifiée)",
                "/api/v1/conversation/*": "Assistant IA conversationnel (Redis + TinyBERT)"
            },
            "development_tips": [
                "Utilisez /debug pour voir les erreurs détaillées",
                "Vérifiez /docs pour l'API interactive",
                "Assurez-vous que PostgreSQL, Redis sont démarrés localement",
                "Elasticsearch optionnel pour search_service"
            ]
        }

    return app

# Créer l'app
app = create_app()

def run_dev_server():
    """Lance le serveur de développement avec hot reload."""
    logger.info("🔥 Lancement du serveur de développement avec hot reload")
    logger.info("📡 Accès: http://localhost:8000")
    logger.info("📚 Docs: http://localhost:8000/docs")
    logger.info("🔍 Debug: http://localhost:8000/debug")
    
    import uvicorn
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