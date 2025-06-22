"""
Application Harena pour déploiement Heroku - Diagnostic complet avec initialisation DIRECTE du Search Service.

Focus: Statut détaillé de chaque service avec initialisation directe et injection forcée pour le Search Service.
"""

import logging
import os
import sys
import time
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(name)s] - %(levelname)s - %(module)s:%(lineno)d - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger("heroku_app")

# ==================== CONFIGURATION INITIALE ====================

try:
    logger.info("🚀 Démarrage Harena Finance Platform - Diagnostic avec Search Service Direct")
    
    # Correction DATABASE_URL pour Heroku
    DATABASE_URL = os.environ.get("DATABASE_URL")
    if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
        os.environ["DATABASE_URL"] = DATABASE_URL.replace("postgres://", "postgresql://", 1)
        logger.info("✅ DATABASE_URL corrigé pour SQLAlchemy")

    # Ajout du répertoire courant au Python path
    current_dir = Path(__file__).parent.absolute()
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))

    # Imports FastAPI
    from fastapi import FastAPI, Request, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse

    logger.info("✅ Imports de base réussis")

    # ==================== VARIABLES GLOBALES ====================

    startup_time = None
    all_services_status = {}
    search_service_initialization_result = None

    # ==================== FONCTIONS DE DIAGNOSTIC DÉTAILLÉ ====================

    async def test_user_service() -> Dict[str, Any]:
        """Test complet du User Service."""
        logger.info("🔍 Test User Service...")
        result = {
            "service": "user_service",
            "healthy": False,
            "details": {
                "importable": False,
                "routes_count": 0,
                "database_tables": False,
                "security_config": False,
                "jwt_secret_configured": False
            },
            "endpoints": [],
            "error": None
        }
        
        try:
            # Test d'import
            from user_service.api.endpoints.users import router as user_router
            result["details"]["importable"] = True
            result["details"]["routes_count"] = len(user_router.routes) if hasattr(user_router, 'routes') else 0
            
            # Extraire les endpoints
            if hasattr(user_router, 'routes'):
                for route in user_router.routes:
                    if hasattr(route, 'methods') and hasattr(route, 'path'):
                        for method in route.methods:
                            result["endpoints"].append(f"{method} {route.path}")
            
            # Test config JWT
            jwt_secret = os.environ.get("JWT_SECRET_KEY") or os.environ.get("SECRET_KEY")
            result["details"]["jwt_secret_configured"] = bool(jwt_secret and len(jwt_secret) >= 32)
            
            # Test des tables de base de données
            try:
                from db_service.session import engine
                from sqlalchemy import inspect
                
                with engine.connect() as conn:
                    inspector = inspect(engine)
                    tables = inspector.get_table_names()
                    result["details"]["database_tables"] = "users" in tables
            except Exception as e:
                logger.warning(f"DB tables check failed: {e}")
            
            # Test sécurité
            try:
                from user_service.core.security import verify_password, get_password_hash
                test_hash = get_password_hash("test123")
                result["details"]["security_config"] = verify_password("test123", test_hash)
            except Exception as e:
                logger.warning(f"Security test failed: {e}")
                result["details"]["security_config"] = False
            
            # Service considéré comme sain s'il est importable et a des routes
            result["healthy"] = (
                result["details"]["importable"] and 
                result["details"]["routes_count"] > 0
            )
                
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"❌ User Service test failed: {e}")
            
        return result

    async def test_db_service() -> Dict[str, Any]:
        """Test complet du DB Service."""
        logger.info("🔍 Test DB Service...")
        result = {
            "service": "db_service",
            "healthy": False,
            "details": {
                "connection": False,
                "models_loaded": False,
                "tables_count": 0,
                "session_factory": False,
                "engine_info": {}
            },
            "error": None
        }
        
        try:
            from db_service.session import engine, SessionLocal
            from sqlalchemy import text, inspect
            
            # Test de connexion
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
                result["details"]["connection"] = True
            
            # Info moteur
            result["details"]["engine_info"] = {
                "url_scheme": str(engine.url).split("://")[0] if hasattr(engine, 'url') else "unknown",
                "pool_size": getattr(engine.pool, 'size', 'unknown') if hasattr(engine, 'pool') else 'unknown'
            }
            
            # Test factory de session
            db = SessionLocal()
            db.close()
            result["details"]["session_factory"] = True
            
            # Compter les tables
            inspector = inspect(engine)
            tables = inspector.get_table_names()
            result["details"]["tables_count"] = len(tables)
            
            # Test des modèles
            try:
                from db_service.models import User, BridgeConnection, RawTransaction
                result["details"]["models_loaded"] = True
            except Exception as e:
                logger.warning(f"Models test failed: {e}")
                result["details"]["models_loaded"] = False
            
            # Service sain si connexion OK et tables > 0
            result["healthy"] = (
                result["details"]["connection"] and 
                result["details"]["tables_count"] > 0
            )
                
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"❌ DB Service test failed: {e}")
            
        return result

    async def test_sync_service() -> Dict[str, Any]:
        """Test complet du Sync Service."""
        logger.info("🔍 Test Sync Service...")
        result = {
            "service": "sync_service",
            "healthy": False,
            "details": {
                "modules_imported": {},
                "bridge_config": False,
                "bridge_endpoints": [],
                "total_routes": 0
            },
            "error": None
        }
        
        sync_modules = [
            ("sync_service.api.endpoints.sync", "/api/v1/sync"),
            ("sync_service.api.endpoints.transactions", "/api/v1/transactions"), 
            ("sync_service.api.endpoints.accounts", "/api/v1/accounts"),
            ("sync_service.api.endpoints.categories", "/api/v1/categories"),
            ("sync_service.api.endpoints.items", "/api/v1/items"),
            ("sync_service.api.endpoints.webhooks", "/webhooks")
        ]
        
        try:
            successful_imports = 0
            total_routes = 0
            
            # Test d'import de chaque module
            for module_path, endpoint in sync_modules:
                try:
                    module = __import__(module_path, fromlist=["router"])
                    router = getattr(module, "router", None)
                    if router and hasattr(router, 'routes'):
                        routes_count = len(router.routes)
                        result["details"]["modules_imported"][module_path] = {
                            "success": True,
                            "routes_count": routes_count,
                            "endpoint": endpoint
                        }
                        successful_imports += 1
                        total_routes += routes_count
                    else:
                        result["details"]["modules_imported"][module_path] = {
                            "success": False,
                            "error": "No router or routes found"
                        }
                except Exception as e:
                    result["details"]["modules_imported"][module_path] = {
                        "success": False,
                        "error": str(e)
                    }
            
            result["details"]["total_routes"] = total_routes
            
            # Test config Bridge
            bridge_url = os.environ.get("BRIDGE_BASE_URL")
            bridge_client_id = os.environ.get("BRIDGE_CLIENT_ID") 
            bridge_client_secret = os.environ.get("BRIDGE_CLIENT_SECRET")
            result["details"]["bridge_config"] = all([bridge_url, bridge_client_id, bridge_client_secret])
            
            if bridge_url:
                result["details"]["bridge_endpoints"] = [
                    f"Base URL: {bridge_url}",
                    f"Client ID: {'✅' if bridge_client_id else '❌'}",
                    f"Client Secret: {'✅' if bridge_client_secret else '❌'}"
                ]
            
            # Service sain si au moins 4 modules sur 6 sont importés
            result["healthy"] = successful_imports >= 4
                
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"❌ Sync Service test failed: {e}")
            
        return result

    async def test_enrichment_service() -> Dict[str, Any]:
        """Test complet de l'Enrichment Service."""
        logger.info("🔍 Test Enrichment Service...")
        result = {
            "service": "enrichment_service", 
            "healthy": False,
            "details": {
                "importable": False,
                "routes_count": 0,
                "ai_configs": {
                    "openai": False,
                    "cohere": False,
                    "deepseek": False
                },
                "capabilities": []
            },
            "error": None
        }
        
        try:
            # Test d'import
            from enrichment_service.api.routes import router as enrichment_router
            result["details"]["importable"] = True
            result["details"]["routes_count"] = len(enrichment_router.routes) if hasattr(enrichment_router, 'routes') else 0
            
            # Test des configs API
            result["details"]["ai_configs"]["openai"] = bool(os.environ.get("OPENAI_API_KEY"))
            result["details"]["ai_configs"]["cohere"] = bool(os.environ.get("COHERE_KEY"))
            result["details"]["ai_configs"]["deepseek"] = bool(os.environ.get("DEEPSEEK_API_KEY"))
            
            # Déterminer les capacités
            if result["details"]["ai_configs"]["openai"]:
                result["details"]["capabilities"].append("OpenAI Embeddings")
            if result["details"]["ai_configs"]["cohere"]:
                result["details"]["capabilities"].append("Cohere Reranking")
            if result["details"]["ai_configs"]["deepseek"]:
                result["details"]["capabilities"].append("DeepSeek Processing")
            
            # Service sain s'il est importable et a au moins une config IA
            result["healthy"] = (
                result["details"]["importable"] and 
                any(result["details"]["ai_configs"].values())
            )
            
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"❌ Enrichment Service test failed: {e}")
            
        return result

    async def test_conversation_service() -> Dict[str, Any]:
        """Test complet du Conversation Service."""
        logger.info("🔍 Test Conversation Service...")
        result = {
            "service": "conversation_service",
            "healthy": False,
            "details": {
                "importable": False,
                "routes_count": 0,
                "deepseek_config": False,
                "components": {
                    "intent_detection": False,
                    "response_generation": False,
                    "token_counter": False
                }
            },
            "error": None
        }
        
        try:
            # Test d'import
            from conversation_service.api.routes import router as conversation_router
            result["details"]["importable"] = True
            result["details"]["routes_count"] = len(conversation_router.routes) if hasattr(conversation_router, 'routes') else 0
            
            # Test config DeepSeek
            result["details"]["deepseek_config"] = bool(os.environ.get("DEEPSEEK_API_KEY"))
            
            # Test des composants
            try:
                from conversation_service.core.intent_detection import IntentDetector
                result["details"]["components"]["intent_detection"] = True
            except Exception:
                pass
            
            try:
                from conversation_service.core.deepseek_client import DeepSeekClient
                result["details"]["components"]["response_generation"] = True
            except Exception:
                pass
                
            try:
                from conversation_service.utils.token_counter import TokenCounter
                result["details"]["components"]["token_counter"] = True
            except Exception:
                pass
            
            # Service sain s'il est importable et a la config DeepSeek
            result["healthy"] = (
                result["details"]["importable"] and 
                result["details"]["deepseek_config"]
            )
                
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"❌ Conversation Service test failed: {e}")
            
        return result

    async def initialize_search_service_direct() -> Dict[str, Any]:
        """Initialise directement le Search Service en appelant les fonctions d'initialisation."""
        logger.info("🔧 === INITIALISATION DIRECTE SEARCH SERVICE ===")
        
        initialization_result = {
            "service": "search_service",
            "healthy": False,
            "details": {
                "importable": False,
                "routes_count": 0,
                "config": {
                    "elasticsearch_url": bool(os.environ.get("BONSAI_URL")),
                    "qdrant_url": bool(os.environ.get("QDRANT_URL")),
                    "openai_key": bool(os.environ.get("OPENAI_API_KEY")),
                    "cohere_key": bool(os.environ.get("COHERE_KEY"))
                },
                "elasticsearch_initialized": False,
                "qdrant_initialized": False,
                "clients_injected": False,
                "initialization_time": 0,
                "connectivity": {
                    "elasticsearch": {
                        "reachable": False,
                        "ping_latency_ms": None,
                        "error": None
                    },
                    "qdrant": {
                        "reachable": False,
                        "ping_latency_ms": None,
                        "error": None
                    }
                },
                "capabilities": {
                    "lexical_search": False,
                    "semantic_search": False,
                    "hybrid_search": False,
                    "caching": False
                }
            },
            "recommendations": [],
            "error": None
        }
        
        start_time = time.time()
        
        try:
            logger.info("📋 Vérification configuration...")
            if not any([
                initialization_result["details"]["config"]["elasticsearch_url"],
                initialization_result["details"]["config"]["qdrant_url"]
            ]):
                initialization_result["error"] = "No search service URLs configured"
                return initialization_result
            
            # Import des modules nécessaires
            logger.info("📦 Import des modules de recherche...")
            try:
                from search_service.storage.elastic_client_hybrid import HybridElasticClient
                from search_service.storage.qdrant_client import QdrantClient
                from config_service.config import settings
            except ImportError as e:
                initialization_result["error"] = f"Import failed: {e}"
                return initialization_result
            
            # Initialisation directe des clients
            elastic_client = None
            qdrant_client = None
            
            # Elasticsearch
            if settings.BONSAI_URL:
                logger.info("🔍 Initialisation directe Elasticsearch...")
                try:
                    elastic_client = HybridElasticClient()
                    success = await elastic_client.initialize()
                    if success and hasattr(elastic_client, '_initialized') and elastic_client._initialized:
                        initialization_result["details"]["elasticsearch_initialized"] = True
                        logger.info("✅ Elasticsearch initialisé directement")
                        
                        # Test de connectivité
                        try:
                            ping_start = time.time()
                            is_healthy = await elastic_client.is_healthy()
                            ping_time = round((time.time() - ping_start) * 1000, 2)
                            
                            initialization_result["details"]["connectivity"]["elasticsearch"]["reachable"] = is_healthy
                            initialization_result["details"]["connectivity"]["elasticsearch"]["ping_latency_ms"] = ping_time
                            
                            if is_healthy:
                                initialization_result["details"]["capabilities"]["lexical_search"] = True
                                logger.info(f"✅ Elasticsearch connectivité OK ({ping_time}ms)")
                        except Exception as health_error:
                            initialization_result["details"]["connectivity"]["elasticsearch"]["error"] = str(health_error)
                            logger.warning(f"⚠️ Test connectivité Elasticsearch échoué: {health_error}")
                    else:
                        logger.error("❌ Échec initialisation Elasticsearch")
                        elastic_client = None
                except Exception as e:
                    logger.error(f"❌ Erreur Elasticsearch: {e}")
                    elastic_client = None
            
            # Qdrant
            if settings.QDRANT_URL:
                logger.info("🎯 Initialisation directe Qdrant...")
                try:
                    qdrant_client = QdrantClient()
                    success = await qdrant_client.initialize()
                    if success and hasattr(qdrant_client, '_initialized') and qdrant_client._initialized:
                        initialization_result["details"]["qdrant_initialized"] = True
                        logger.info("✅ Qdrant initialisé directement")
                        
                        # Test de connectivité
                        try:
                            ping_start = time.time()
                            is_healthy = await qdrant_client.is_healthy()
                            ping_time = round((time.time() - ping_start) * 1000, 2)
                            
                            initialization_result["details"]["connectivity"]["qdrant"]["reachable"] = is_healthy
                            initialization_result["details"]["connectivity"]["qdrant"]["ping_latency_ms"] = ping_time
                            
                            if is_healthy:
                                initialization_result["details"]["capabilities"]["semantic_search"] = True
                                logger.info(f"✅ Qdrant connectivité OK ({ping_time}ms)")
                        except Exception as health_error:
                            initialization_result["details"]["connectivity"]["qdrant"]["error"] = str(health_error)
                            logger.warning(f"⚠️ Test connectivité Qdrant échoué: {health_error}")
                    else:
                        logger.error("❌ Échec initialisation Qdrant")
                        qdrant_client = None
                except Exception as e:
                    logger.error(f"❌ Erreur Qdrant: {e}")
                    qdrant_client = None
            
            # Capacités avancées
            initialization_result["details"]["capabilities"]["hybrid_search"] = (
                initialization_result["details"]["capabilities"]["lexical_search"] and 
                initialization_result["details"]["capabilities"]["semantic_search"]
            )
            
            # Injection DIRECTE dans les routes
            if elastic_client or qdrant_client:
                logger.info("🔗 Injection directe dans les routes...")
                try:
                    # Forcer l'import et l'injection
                    import search_service.api.routes as routes
                    
                    # Injection directe
                    routes.elastic_client = elastic_client
                    routes.qdrant_client = qdrant_client
                    routes.embedding_service = None  # Pas critique pour le test
                    routes.reranker_service = None  # Pas critique pour le test
                    routes.search_cache = None  # Pas critique pour le test
                    routes.metrics_collector = None  # Pas critique pour le test
                    
                    # Vérification
                    elastic_injected = hasattr(routes, 'elastic_client') and routes.elastic_client is not None
                    qdrant_injected = hasattr(routes, 'qdrant_client') and routes.qdrant_client is not None
                    
                    initialization_result["details"]["clients_injected"] = elastic_injected or qdrant_injected
                    
                    logger.info(f"✅ Injection directe réussie:")
                    logger.info(f"   - Elasticsearch: {'✅' if elastic_injected else '❌'}")
                    logger.info(f"   - Qdrant: {'✅' if qdrant_injected else '❌'}")
                    
                    # Import des routes pour l'enregistrement
                    from search_service.api.routes import router as search_router
                    initialization_result["details"]["importable"] = True
                    initialization_result["details"]["routes_count"] = len(search_router.routes) if hasattr(search_router, 'routes') else 0
                    
                    # Succès si au moins un client injecté
                    initialization_result["healthy"] = initialization_result["details"]["clients_injected"]
                    
                    # Générer les recommandations
                    recommendations = []
                    if not initialization_result["details"]["config"]["elasticsearch_url"]:
                        recommendations.append("⚠️ Configurez BONSAI_URL pour Elasticsearch")
                    if not initialization_result["details"]["config"]["qdrant_url"]:
                        recommendations.append("⚠️ Configurez QDRANT_URL pour Qdrant")
                    if not initialization_result["details"]["config"]["openai_key"] and not initialization_result["details"]["config"]["cohere_key"]:
                        recommendations.append("⚠️ Configurez OPENAI_API_KEY ou COHERE_KEY")
                        
                    if not elastic_injected and initialization_result["details"]["config"]["elasticsearch_url"]:
                        recommendations.append("🔧 Problème d'injection du client Elasticsearch")
                    if not qdrant_injected and initialization_result["details"]["config"]["qdrant_url"]:
                        recommendations.append("🔧 Problème d'injection du client Qdrant")
                        
                    if elastic_injected and not initialization_result["details"]["connectivity"]["elasticsearch"]["reachable"]:
                        recommendations.append("🌐 Vérifiez la connectivité réseau vers Elasticsearch")
                    if qdrant_injected and not initialization_result["details"]["connectivity"]["qdrant"]["reachable"]:
                        recommendations.append("🌐 Vérifiez la connectivité réseau vers Qdrant")
                        
                    if not recommendations and initialization_result["healthy"]:
                        recommendations.append("✅ Search Service complètement opérationnel")
                        
                    initialization_result["recommendations"] = recommendations
                    
                except Exception as e:
                    logger.error(f"❌ Erreur injection: {e}")
                    initialization_result["error"] = f"Injection failed: {e}"
            else:
                initialization_result["error"] = "No clients initialized successfully"
            
            initialization_result["details"]["initialization_time"] = time.time() - start_time
            
        except Exception as e:
            logger.error(f"💥 Erreur générale: {e}")
            initialization_result["error"] = str(e)
            initialization_result["details"]["initialization_time"] = time.time() - start_time
        
        total_time = time.time() - start_time
        logger.info(f"🏁 Initialisation directe terminée en {total_time:.2f}s")
        
        return initialization_result

    async def run_complete_services_diagnostic() -> Dict[str, Any]:
        """Lance un diagnostic complet de tous les services avec traitement spécial pour Search Service."""
        logger.info("🔍 Lancement du diagnostic complet de tous les services...")
        
        # Tests en parallèle pour les services standards
        standard_tests = await asyncio.gather(
            test_user_service(),
            test_db_service(), 
            test_sync_service(),
            test_enrichment_service(),
            test_conversation_service(),
            return_exceptions=True
        )
        
        services_status = {}
        standard_services = ["user_service", "db_service", "sync_service", "enrichment_service", "conversation_service"]
        
        for i, test_result in enumerate(standard_tests):
            service_name = standard_services[i]
            if isinstance(test_result, Exception):
                services_status[service_name] = {
                    "healthy": False,
                    "error": str(test_result),
                    "details": {}
                }
                logger.error(f"❌ {service_name} diagnostic failed: {test_result}")
            else:
                services_status[service_name] = test_result
                status_icon = "✅" if test_result["healthy"] else "❌"
                logger.info(f"{status_icon} {service_name}: {'Healthy' if test_result['healthy'] else 'Unhealthy'}")
        
        # Traitement spécial pour le Search Service avec initialisation directe
        logger.info("🔧 Initialisation directe du Search Service...")
        search_service_result = await initialize_search_service_direct()
        services_status["search_service"] = search_service_result
        
        status_icon = "✅" if search_service_result["healthy"] else "❌"
        logger.info(f"{status_icon} search_service: {'Healthy' if search_service_result['healthy'] else 'Unhealthy'}")
        
        if search_service_result["healthy"]:
            capabilities = search_service_result["details"]["capabilities"]
            logger.info(f"   🎯 Capacités: Lexical={capabilities['lexical_search']}, Semantic={capabilities['semantic_search']}, Hybrid={capabilities['hybrid_search']}")
        
        global search_service_initialization_result
        search_service_initialization_result = search_service_result
        
        return services_status

    # ==================== REGISTRE DE SERVICES ====================

    class ServiceRegistry:
        def __init__(self):
            self.services = {}
            self.failed_services = {}
        
        def register(self, name: str, router, prefix: str, description: str = ""):
            try:
                if router:
                    routes_count = len(router.routes) if hasattr(router, 'routes') else 0
                    self.services[name] = {
                        "router": router,
                        "prefix": prefix,
                        "description": description,
                        "routes_count": routes_count,
                        "registered_at": datetime.now()
                    }
                    logger.info(f"✅ {name}: {prefix} ({routes_count} routes)")
                    return True
                else:
                    raise ValueError("Router is None")
            except Exception as e:
                self.failed_services[name] = {
                    "error": str(e),
                    "failed_at": datetime.now()
                }
                logger.error(f"❌ {name}: {e}")
                return False
        
        def get_summary(self):
            return {
                "registered": len(self.services),
                "failed": len(self.failed_services),
                "total_routes": sum(s.get("routes_count", 0) for s in self.services.values())
            }

    service_registry = ServiceRegistry()

    # ==================== CYCLE DE VIE ====================

    async def startup():
        """Initialisation de l'application avec diagnostic complet et Search Service direct."""
        global startup_time, all_services_status
        startup_time = time.time()
        logger.info("📋 Démarrage application Harena avec Search Service direct...")
        
        # Test de connexion DB immédiat
        try:
            from db_service.session import engine
            from sqlalchemy import text
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("✅ Base de données connectée")
        except Exception as e:
            logger.error(f"❌ Erreur DB critique: {e}")
            raise RuntimeError("Database connection failed")
        
        # Lancer le diagnostic complet avec traitement spécial Search Service
        all_services_status = await run_complete_services_diagnostic()
        
        total_time = time.time() - startup_time
        healthy_count = sum(1 for s in all_services_status.values() if s.get("healthy"))
        logger.info(f"✅ Démarrage terminé en {total_time:.2f}s - {healthy_count}/{len(all_services_status)} services sains")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await startup()
        yield

    # ==================== APPLICATION FASTAPI ====================

    app = FastAPI(
        title="Harena Finance Platform",
        description="Plateforme de gestion financière avec recherche hybride directe",
        version="1.0.0",
        lifespan=lifespan
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ==================== ENDPOINTS DE DIAGNOSTIC ====================

    @app.get("/")
    async def root():
        """Statut général avec focus spécial sur le Search Service."""
        uptime = time.time() - startup_time if startup_time else 0
        
        # Résumé global
        healthy_services = [name for name, status in all_services_status.items() if status.get("healthy")]
        failed_services = [name for name, status in all_services_status.items() if not status.get("healthy")]
        
        # Focus spécial Search Service avec données détaillées
        search_status = all_services_status.get("search_service", {})
        search_details = search_status.get("details", {})
        
        return {
            "service": "Harena Finance API",
            "status": "online",
            "version": "1.0.0",
            "uptime_seconds": round(uptime, 2),
            "services": {
                "healthy": len(healthy_services),
                "failed": len(failed_services),
                "healthy_list": healthy_services,
                "failed_list": failed_services
            },
            "search_service": {
                "configured": search_details.get("config", {}).get("elasticsearch_url", False) and search_details.get("config", {}).get("qdrant_url", False),
                "healthy": search_status.get("healthy", False),
                "clients_injected": search_details.get("clients_injected", False),
                "elasticsearch_reachable": search_details.get("connectivity", {}).get("elasticsearch", {}).get("reachable", False),
                "qdrant_reachable": search_details.get("connectivity", {}).get("qdrant", {}).get("reachable", False),
                "status": "fully_operational" if search_status.get("healthy") and search_details.get("capabilities", {}).get("hybrid_search") else "degraded",
                "initialization_time": search_details.get("initialization_time", 0),
                "capabilities": search_details.get("capabilities", {})
            },
            "timestamp": datetime.now().isoformat()
        }

    @app.get("/health")
    async def health_check():
        """Vérification de santé ultra-détaillée de tous les services."""
        uptime = time.time() - startup_time if startup_time else 0
        
        registry_summary = service_registry.get_summary()
        
        return {
            "status": "healthy" if all(s.get("healthy") for s in all_services_status.values()) else "degraded",
            "uptime_seconds": round(uptime, 2),
            "timestamp": datetime.now().isoformat(),
            "registry": registry_summary,
            "services": all_services_status,
            "search_service_initialization": search_service_initialization_result
        }

    @app.get("/search-service")
    async def search_service_ultra_detailed():
        """Statut ultra-détaillé du Search Service avec toutes les métriques d'initialisation."""
        search_status = all_services_status.get("search_service", {})
        
        return {
            "service": "search_service",
            "priority": "critical",
            "timestamp": datetime.now().isoformat(),
            "overall_status": "fully_operational" if search_status.get("healthy") else "degraded",
            "health_summary": {
                "importable": search_status.get("details", {}).get("importable", False),
                "routes_count": search_status.get("details", {}).get("routes_count", 0),
                "clients_injected": search_status.get("details", {}).get("clients_injected", False),
                "elasticsearch_initialized": search_status.get("details", {}).get("elasticsearch_initialized", False),
                "qdrant_initialized": search_status.get("details", {}).get("qdrant_initialized", False),
                "connectivity_ok": any([
                    search_status.get("details", {}).get("connectivity", {}).get("elasticsearch", {}).get("reachable", False),
                    search_status.get("details", {}).get("connectivity", {}).get("qdrant", {}).get("reachable", False)
                ])
            },
            "initialization_metrics": {
                "method": "direct_initialization",
                "initialization_time": search_status.get("details", {}).get("initialization_time", 0),
                "clients_injected": search_status.get("details", {}).get("clients_injected", False)
            },
            "configuration": search_status.get("details", {}).get("config", {}),
            "clients_status": {
                "elasticsearch_initialized": search_status.get("details", {}).get("elasticsearch_initialized", False),
                "qdrant_initialized": search_status.get("details", {}).get("qdrant_initialized", False)
            },
            "connectivity_tests": search_status.get("details", {}).get("connectivity", {}),
            "capabilities": search_status.get("details", {}).get("capabilities", {}),
            "recommendations": search_status.get("recommendations", []),
            "error": search_status.get("error"),
            "endpoints": [
                "POST /api/v1/search/search - Recherche de transactions" if search_status.get("healthy") else "❌ POST /api/v1/search/search - Indisponible",
                "GET /api/v1/search/health - Santé du service" if search_status.get("healthy") else "❌ GET /api/v1/search/health - Indisponible",
                "GET /api/v1/search/debug/injection - Debug injection" if search_status.get("healthy") else "❌ GET /api/v1/search/debug/injection - Indisponible",
                "GET /search-service - Ce diagnostic détaillé"
            ]
        }

    @app.get("/services-summary")
    async def services_summary():
        """Résumé synthétique de tous les services avec focus Search Service."""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "services_count": len(all_services_status),
            "healthy_count": sum(1 for s in all_services_status.values() if s.get("healthy")),
            "registry": service_registry.get_summary(),
            "services": {}
        }
        
        for service_name, status in all_services_status.items():
            summary["services"][service_name] = {
                "healthy": status.get("healthy", False),
                "error": status.get("error"),
                "key_metrics": {
                    "importable": status.get("details", {}).get("importable", False),
                    "routes_count": status.get("details", {}).get("routes_count", 0)
                }
            }
            
            # Métriques spéciales par service
            if service_name == "search_service":
                summary["services"][service_name]["key_metrics"].update({
                    "initialization_method": "direct",
                    "clients_injected": status.get("details", {}).get("clients_injected", False),
                    "elasticsearch_initialized": status.get("details", {}).get("elasticsearch_initialized", False),
                    "qdrant_initialized": status.get("details", {}).get("qdrant_initialized", False),
                    "elasticsearch_reachable": status.get("details", {}).get("connectivity", {}).get("elasticsearch", {}).get("reachable", False),
                    "qdrant_reachable": status.get("details", {}).get("connectivity", {}).get("qdrant", {}).get("reachable", False),
                    "hybrid_search": status.get("details", {}).get("capabilities", {}).get("hybrid_search", False),
                    "initialization_time": status.get("details", {}).get("initialization_time", 0)
                })
            elif service_name == "db_service":
                summary["services"][service_name]["key_metrics"].update({
                    "connection": status.get("details", {}).get("connection", False),
                    "tables_count": status.get("details", {}).get("tables_count", 0)
                })
            elif service_name == "sync_service":
                summary["services"][service_name]["key_metrics"].update({
                    "bridge_config": status.get("details", {}).get("bridge_config", False),
                    "modules_success": sum(1 for m in status.get("details", {}).get("modules_imported", {}).values() if m.get("success"))
                })
        
        return summary

    # ==================== IMPORTATION DES SERVICES ====================

    logger.info("📦 Importation des services...")

    # 1. User Service
    try:
        from user_service.api.endpoints.users import router as user_router
        if service_registry.register("user_service", user_router, "/api/v1/users", "Gestion utilisateurs"):
            app.include_router(user_router, prefix="/api/v1/users", tags=["users"])
    except Exception as e:
        logger.error(f"❌ User Service: {e}")

    # 2. Sync Service - modules principaux
    sync_modules = [
        ("sync_service.api.endpoints.sync", "/api/v1/sync", "Synchronisation"),
        ("sync_service.api.endpoints.transactions", "/api/v1/transactions", "Transactions"),
        ("sync_service.api.endpoints.accounts", "/api/v1/accounts", "Comptes"),
        ("sync_service.api.endpoints.categories", "/api/v1/categories", "Catégories"),
        ("sync_service.api.endpoints.items", "/api/v1/items", "Items Bridge"),
        ("sync_service.api.endpoints.webhooks", "/webhooks", "Webhooks"),
    ]

    for module_path, prefix, description in sync_modules:
        try:
            module = __import__(module_path, fromlist=["router"])
            router = getattr(module, "router")
            service_name = f"sync_{module_path.split('.')[-1]}"
            if service_registry.register(service_name, router, prefix, description):
                app.include_router(router, prefix=prefix, tags=[module_path.split('.')[-1]])
        except Exception as e:
            logger.error(f"❌ {module_path}: {e}")

    # 3. Enrichment Service
    try:
        from enrichment_service.api.routes import router as enrichment_router
        if service_registry.register("enrichment_service", enrichment_router, "/api/v1/enrichment", "Enrichissement IA"):
            app.include_router(enrichment_router, prefix="/api/v1/enrichment", tags=["enrichment"])
    except Exception as e:
        logger.error(f"❌ Enrichment Service: {e}")

    # 4. Search Service (CRITIQUE) - Enregistrement IMMÉDIAT après initialisation directe
    # Note: Le Search Service sera enregistré automatiquement après le startup
    # car l'initialisation directe aura déjà injecté les clients

    # 5. Conversation Service
    try:
        from conversation_service.api.routes import router as conversation_router
        if service_registry.register("conversation_service", conversation_router, "/api/v1/conversation", "Assistant IA"):
            app.include_router(conversation_router, prefix="/api/v1/conversation", tags=["conversation"])
    except Exception as e:
        logger.error(f"❌ Conversation Service: {e}")

    # ==================== ENREGISTREMENT SEARCH SERVICE POST-STARTUP ====================

    @app.on_event("startup") 
    async def register_search_service_after_direct_initialization():
        """Enregistre le Search Service après son initialisation directe."""
        await asyncio.sleep(0.1)  # Petit délai pour s'assurer que le diagnostic est terminé
        
        if search_service_initialization_result and search_service_initialization_result.get("healthy", False):
            try:
                from search_service.api.routes import router as search_router
                if service_registry.register("search_service", search_router, "/api/v1/search", "🔍 CRITIQUE: Recherche hybride (Direct)"):
                    app.include_router(search_router, prefix="/api/v1/search", tags=["search"])
                    logger.info("🎉 Search Service enregistré avec succès après initialisation directe")
                else:
                    logger.error("🚨 Search Service: Échec enregistrement après initialisation directe réussie")
            except Exception as e:
                logger.error(f"💥 Search Service post-direct registration: {e}")
        else:
            logger.error("🚨 Search Service: Non initialisé directement - enregistrement ignoré")
            if search_service_initialization_result:
                logger.error(f"   Erreur: {search_service_initialization_result.get('error', 'Unknown error')}")

    # ==================== ENDPOINTS UTILITAIRES ====================

    @app.get("/version")
    async def version():
        return {
            "version": "1.0.0",
            "build": "heroku-search-service-direct-fix",
            "python": sys.version.split()[0],
            "environment": os.environ.get("ENVIRONMENT", "production"),
            "diagnostic_features": [
                "Complete services health check",
                "Search Service DIRECT initialization (no polling)",
                "Client injection immediate after init",
                "Connectivity tests with latency monitoring",
                "Service capabilities assessment",
                "Detailed initialization metrics"
            ]
        }

    @app.get("/robots.txt", include_in_schema=False)
    async def robots():
        return JSONResponse("User-agent: *\nDisallow: /", media_type="text/plain")

    # ==================== GESTIONNAIRE D'ERREURS ====================

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        search_status = all_services_status.get("search_service", {})
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.detail,
                "path": request.url.path,
                "search_service_healthy": search_status.get("healthy", False),
                "healthy_services": [name for name, status in all_services_status.items() if status.get("healthy")],
                "timestamp": datetime.now().isoformat()
            }
        )

    # ==================== RAPPORT FINAL ====================

    logger.info("=" * 80)
    logger.info("🎯 HARENA FINANCE PLATFORM - SEARCH SERVICE DIRECT FIX")
    logger.info(f"📊 Services enregistrés: {service_registry.get_summary()['registered']}")
    logger.info(f"❌ Services échoués: {service_registry.get_summary()['failed']}")
    logger.info("🔧 Fonctionnalités de diagnostic avancées:")
    logger.info("   📋 Diagnostic complet de tous les services")
    logger.info("   🔍 Initialisation DIRECTE Search Service (pas de polling)")
    logger.info("   🔗 Injection immédiate clients après initialisation")
    logger.info("   🌐 Tests connectivité Elasticsearch & Qdrant avec latence")
    logger.info("   📊 Monitoring capacités en temps réel")
    logger.info("   ⏱️ Métriques d'initialisation détaillées")
    logger.info("🌐 Endpoints de diagnostic:")
    logger.info("   GET  / - Statut général avec métriques Search Service")
    logger.info("   GET  /health - Santé ultra-détaillée tous services")
    logger.info("   GET  /search-service - Diagnostic approfondi Search Service")
    logger.info("   GET  /services-summary - Résumé synthétique")
    logger.info("🔧 Endpoints principaux:")
    logger.info("   POST /api/v1/search/search - Recherche de transactions")
    logger.info("   GET  /api/v1/search/debug/injection - Debug injection")
    logger.info("   POST /api/v1/conversation/chat - Assistant IA")
    logger.info("   GET  /api/v1/sync - Synchronisation Bridge")
    logger.info("   POST /api/v1/users/register - Enregistrement utilisateur")
    logger.info("=" * 80)
    logger.info("✅ Application Harena prête avec Search Service direct fix")

except Exception as critical_error:
    logger.critical(f"💥 ERREUR CRITIQUE: {critical_error}")
    raise

# Point d'entrée Heroku
if 'app' not in locals():
    raise RuntimeError("FastAPI app not created")

# Mode développement
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("heroku_app:app", host="0.0.0.0", port=8000, reload=True)