"""
Application Harena pour déploiement Heroku - Diagnostic complet de tous les services.

Focus: Statut détaillé de chaque service avec diagnostic approfondi du Search Service.
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
    logger.info("🚀 Démarrage Harena Finance Platform - Diagnostic Complet")
    
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

    async def test_search_service_deep() -> Dict[str, Any]:
        """Test ultra-approfondi du Search Service avec connectivité."""
        logger.info("🔍 Test Search Service (Approfondi)...")
        result = {
            "service": "search_service",
            "healthy": False,
            "details": {
                "importable": False,
                "routes_count": 0,
                "config": {
                    "elasticsearch_url": False,
                    "qdrant_url": False,
                    "openai_key": False,
                    "cohere_key": False
                },
                "clients_injection": {
                    "elasticsearch_injected": False,
                    "qdrant_injected": False,
                    "embedding_service_injected": False,
                    "cache_injected": False
                },
                "connectivity": {
                    "elasticsearch": {
                        "reachable": False,
                        "index_accessible": False,
                        "ping_latency_ms": None,
                        "error": None
                    },
                    "qdrant": {
                        "reachable": False,
                        "collections_accessible": False,
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
        
        try:
            # Test d'import
            from search_service.api.routes import router as search_router
            result["details"]["importable"] = True
            result["details"]["routes_count"] = len(search_router.routes) if hasattr(search_router, 'routes') else 0
            
            # Test de la configuration
            result["details"]["config"]["elasticsearch_url"] = bool(os.environ.get("BONSAI_URL"))
            result["details"]["config"]["qdrant_url"] = bool(os.environ.get("QDRANT_URL"))  
            result["details"]["config"]["openai_key"] = bool(os.environ.get("OPENAI_API_KEY"))
            result["details"]["config"]["cohere_key"] = bool(os.environ.get("COHERE_KEY"))
            
            # Test d'injection des clients
            try:
                from search_service.api import routes as search_routes
                result["details"]["clients_injection"]["elasticsearch_injected"] = (
                    hasattr(search_routes, 'elastic_client') and 
                    search_routes.elastic_client is not None
                )
                result["details"]["clients_injection"]["qdrant_injected"] = (
                    hasattr(search_routes, 'qdrant_client') and 
                    search_routes.qdrant_client is not None
                )
                result["details"]["clients_injection"]["embedding_service_injected"] = (
                    hasattr(search_routes, 'embedding_service') and 
                    search_routes.embedding_service is not None
                )
                result["details"]["clients_injection"]["cache_injected"] = (
                    hasattr(search_routes, 'search_cache') and 
                    search_routes.search_cache is not None
                )
            except Exception as e:
                logger.warning(f"Client injection test failed: {e}")
            
            # Test de connectivité Elasticsearch
            if result["details"]["clients_injection"]["elasticsearch_injected"]:
                try:
                    from search_service.api import routes as search_routes
                    elastic_client = search_routes.elastic_client
                    
                    if elastic_client:
                        start_time = time.time()
                        # Test de ping/santé
                        if hasattr(elastic_client, 'is_healthy'):
                            is_healthy = await elastic_client.is_healthy()
                            result["details"]["connectivity"]["elasticsearch"]["reachable"] = is_healthy
                            result["details"]["connectivity"]["elasticsearch"]["ping_latency_ms"] = round((time.time() - start_time) * 1000, 2)
                            
                            if is_healthy:
                                result["details"]["capabilities"]["lexical_search"] = True
                        elif hasattr(elastic_client, 'ping'):
                            ping_result = elastic_client.ping()
                            result["details"]["connectivity"]["elasticsearch"]["reachable"] = bool(ping_result)
                            result["details"]["connectivity"]["elasticsearch"]["ping_latency_ms"] = round((time.time() - start_time) * 1000, 2)
                            
                            if ping_result:
                                result["details"]["capabilities"]["lexical_search"] = True
                                
                except Exception as e:
                    result["details"]["connectivity"]["elasticsearch"]["error"] = str(e)
                    logger.warning(f"Elasticsearch connectivity test failed: {e}")
            
            # Test de connectivité Qdrant
            if result["details"]["clients_injection"]["qdrant_injected"]:
                try:
                    from search_service.api import routes as search_routes
                    qdrant_client = search_routes.qdrant_client
                    
                    if qdrant_client:
                        start_time = time.time()
                        # Test de ping/santé
                        if hasattr(qdrant_client, 'is_healthy'):
                            is_healthy = await qdrant_client.is_healthy()
                            result["details"]["connectivity"]["qdrant"]["reachable"] = is_healthy
                            result["details"]["connectivity"]["qdrant"]["ping_latency_ms"] = round((time.time() - start_time) * 1000, 2)
                            
                            if is_healthy:
                                result["details"]["capabilities"]["semantic_search"] = True
                        elif hasattr(qdrant_client, 'get_collections'):
                            # Test basique de récupération des collections
                            collections = await qdrant_client.get_collections()
                            result["details"]["connectivity"]["qdrant"]["reachable"] = True
                            result["details"]["connectivity"]["qdrant"]["collections_accessible"] = True
                            result["details"]["connectivity"]["qdrant"]["ping_latency_ms"] = round((time.time() - start_time) * 1000, 2)
                            result["details"]["capabilities"]["semantic_search"] = True
                                
                except Exception as e:
                    result["details"]["connectivity"]["qdrant"]["error"] = str(e)
                    logger.warning(f"Qdrant connectivity test failed: {e}")
            
            # Capacités avancées
            result["details"]["capabilities"]["hybrid_search"] = (
                result["details"]["capabilities"]["lexical_search"] and 
                result["details"]["capabilities"]["semantic_search"]
            )
            result["details"]["capabilities"]["caching"] = result["details"]["clients_injection"]["cache_injected"]
            
            # Générer les recommandations
            recommendations = []
            if not result["details"]["config"]["elasticsearch_url"]:
                recommendations.append("⚠️ Configurez BONSAI_URL pour Elasticsearch")
            if not result["details"]["config"]["qdrant_url"]:
                recommendations.append("⚠️ Configurez QDRANT_URL pour Qdrant")
            if not result["details"]["config"]["openai_key"] and not result["details"]["config"]["cohere_key"]:
                recommendations.append("⚠️ Configurez OPENAI_API_KEY ou COHERE_KEY")
                
            if not result["details"]["clients_injection"]["elasticsearch_injected"] and result["details"]["config"]["elasticsearch_url"]:
                recommendations.append("🔧 Problème d'injection du client Elasticsearch")
            if not result["details"]["clients_injection"]["qdrant_injected"] and result["details"]["config"]["qdrant_url"]:
                recommendations.append("🔧 Problème d'injection du client Qdrant")
                
            if result["details"]["clients_injection"]["elasticsearch_injected"] and not result["details"]["connectivity"]["elasticsearch"]["reachable"]:
                recommendations.append("🌐 Vérifiez la connectivité réseau vers Elasticsearch")
            if result["details"]["clients_injection"]["qdrant_injected"] and not result["details"]["connectivity"]["qdrant"]["reachable"]:
                recommendations.append("🌐 Vérifiez la connectivité réseau vers Qdrant")
                
            if not recommendations:
                recommendations.append("✅ Search Service complètement opérationnel")
                
            result["recommendations"] = recommendations
            
            # Déterminer la santé globale
            clients_ok = (
                result["details"]["clients_injection"]["elasticsearch_injected"] or 
                result["details"]["clients_injection"]["qdrant_injected"]
            )
            connectivity_ok = (
                result["details"]["connectivity"]["elasticsearch"]["reachable"] or 
                result["details"]["connectivity"]["qdrant"]["reachable"]
            )
            
            result["healthy"] = (
                result["details"]["importable"] and 
                clients_ok and 
                connectivity_ok
            )
            
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"❌ Search Service deep test failed: {e}")
            
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

    async def run_complete_services_diagnostic() -> Dict[str, Any]:
        """Lance un diagnostic complet de tous les services en parallèle."""
        logger.info("🔍 Lancement du diagnostic complet de tous les services...")
        
        # Tests en parallèle pour plus d'efficacité
        tests = await asyncio.gather(
            test_user_service(),
            test_db_service(), 
            test_sync_service(),
            test_enrichment_service(),
            test_search_service_deep(),  # Test approfondi
            test_conversation_service(),
            return_exceptions=True
        )
        
        services_status = {}
        services = ["user_service", "db_service", "sync_service", "enrichment_service", "search_service", "conversation_service"]
        
        for i, test_result in enumerate(tests):
            service_name = services[i]
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
        """Initialisation de l'application avec diagnostic complet."""
        global startup_time, all_services_status
        startup_time = time.time()
        logger.info("📋 Démarrage application Harena avec diagnostic complet...")
        
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
        
        # Attendre que tous les services soient enregistrés
        await asyncio.sleep(3.0)
        
        # Lancer le diagnostic complet de tous les services
        all_services_status = await run_complete_services_diagnostic()
        
        total_time = time.time() - startup_time
        healthy_count = sum(1 for s in all_services_status.values() if s.get("healthy"))
        logger.info(f"✅ Démarrage terminé en {total_time:.2f}s - {healthy_count}/{len(all_services_status)} services sains")

    async def shutdown():
        """Arrêt de l'application."""
        logger.info("⏹️ Arrêt application Harena...")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await startup()
        yield
        await shutdown()

    # ==================== APPLICATION FASTAPI ====================

    app = FastAPI(
        title="Harena Finance Platform",
        description="Plateforme de gestion financière avec recherche hybride",
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
        """Statut général avec focus sur tous les services."""
        uptime = time.time() - startup_time if startup_time else 0
        
        # Résumé global
        healthy_services = [name for name, status in all_services_status.items() if status.get("healthy")]
        failed_services = [name for name, status in all_services_status.items() if not status.get("healthy")]
        
        # Focus spécial Search Service
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
                "clients_injected": search_details.get("clients_injection", {}).get("elasticsearch_injected", False) or search_details.get("clients_injection", {}).get("qdrant_injected", False),
                "elasticsearch_reachable": search_details.get("connectivity", {}).get("elasticsearch", {}).get("reachable", False),
                "qdrant_reachable": search_details.get("connectivity", {}).get("qdrant", {}).get("reachable", False),
                "status": "fully_operational" if search_status.get("healthy") and search_details.get("capabilities", {}).get("hybrid_search") else "degraded"
            },
            "timestamp": datetime.now().isoformat()
        }

    @app.get("/health")
    async def health_check():
        """Vérification de santé ultra-détaillée de tous les services."""
        uptime = time.time() - startup_time if startup_time else 0
        
        # Relancer le diagnostic pour avoir des données fraîches
        fresh_diagnostic = await run_complete_services_diagnostic()
        
        registry_summary = service_registry.get_summary()
        
        return {
            "status": "healthy" if all(s.get("healthy") for s in fresh_diagnostic.values()) else "degraded",
            "uptime_seconds": round(uptime, 2),
            "timestamp": datetime.now().isoformat(),
            "registry": registry_summary,
            "services": fresh_diagnostic
        }

    @app.get("/search-service")
    async def search_service_ultra_detailed():
        """Statut ultra-détaillé du Search Service uniquement."""
        search_diagnostic = await test_search_service_deep()
        
        return {
            "service": "search_service",
            "priority": "critical",
            "timestamp": datetime.now().isoformat(),
            "overall_status": "fully_operational" if search_diagnostic.get("healthy") else "degraded",
            "health_summary": {
                "importable": search_diagnostic.get("details", {}).get("importable", False),
                "routes_count": search_diagnostic.get("details", {}).get("routes_count", 0),
                "clients_injected": any(search_diagnostic.get("details", {}).get("clients_injection", {}).values()),
                "connectivity_ok": any([
                    search_diagnostic.get("details", {}).get("connectivity", {}).get("elasticsearch", {}).get("reachable", False),
                    search_diagnostic.get("details", {}).get("connectivity", {}).get("qdrant", {}).get("reachable", False)
                ])
            },
            "configuration": search_diagnostic.get("details", {}).get("config", {}),
            "clients_injection": search_diagnostic.get("details", {}).get("clients_injection", {}),
            "connectivity_tests": search_diagnostic.get("details", {}).get("connectivity", {}),
            "capabilities": search_diagnostic.get("details", {}).get("capabilities", {}),
            "recommendations": search_diagnostic.get("recommendations", []),
            "error": search_diagnostic.get("error"),
            "endpoints": [
                "POST /api/v1/search/search - Recherche de transactions",
                "GET /api/v1/search/suggest - Suggestions de recherche",
                "GET /search-service - Ce diagnostic détaillé"
            ]
        }

    @app.get("/services-summary")
    async def services_summary():
        """Résumé synthétique de tous les services."""
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
                    "elasticsearch_injected": status.get("details", {}).get("clients_injection", {}).get("elasticsearch_injected", False),
                    "qdrant_injected": status.get("details", {}).get("clients_injection", {}).get("qdrant_injected", False),
                    "elasticsearch_reachable": status.get("details", {}).get("connectivity", {}).get("elasticsearch", {}).get("reachable", False),
                    "qdrant_reachable": status.get("details", {}).get("connectivity", {}).get("qdrant", {}).get("reachable", False)
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

    # 4. Search Service (CRITIQUE)
    try:
        from search_service.api.routes import router as search_router
        if service_registry.register("search_service", search_router, "/api/v1/search", "🔍 CRITIQUE: Recherche hybride"):
            app.include_router(search_router, prefix="/api/v1/search", tags=["search"])
            logger.info("🎉 Search Service enregistré avec succès")
        else:
            logger.error("🚨 Search Service: Échec enregistrement critique")
    except Exception as e:
        logger.error(f"💥 Search Service: {e}")

    # 5. Conversation Service
    try:
        from conversation_service.api.routes import router as conversation_router
        if service_registry.register("conversation_service", conversation_router, "/api/v1/conversation", "Assistant IA"):
            app.include_router(conversation_router, prefix="/api/v1/conversation", tags=["conversation"])
    except Exception as e:
        logger.error(f"❌ Conversation Service: {e}")

    # ==================== ENDPOINTS UTILITAIRES ====================

    @app.get("/version")
    async def version():
        return {
            "version": "1.0.0",
            "build": "heroku-complete-diagnostic",
            "python": sys.version.split()[0],
            "environment": os.environ.get("ENVIRONMENT", "production"),
            "diagnostic_features": [
                "Complete services health check",
                "Search Service deep connectivity tests",
                "Real-time client injection monitoring",
                "Elasticsearch & Qdrant ping tests",
                "Service capabilities assessment"
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
    logger.info("🎯 HARENA FINANCE PLATFORM - DIAGNOSTIC COMPLET")
    logger.info(f"📊 Services enregistrés: {service_registry.get_summary()['registered']}")
    logger.info(f"❌ Services échoués: {service_registry.get_summary()['failed']}")
    logger.info("🔍 Fonctionnalités de diagnostic:")
    logger.info("   📋 Diagnostic complet de tous les services")
    logger.info("   🔗 Test d'injection des clients Search Service")
    logger.info("   🌐 Tests de connectivité Elasticsearch & Qdrant")
    logger.info("   📊 Monitoring en temps réel des capacités")
    logger.info("🌐 Endpoints de diagnostic:")
    logger.info("   GET  / - Statut général avec résumé tous services")
    logger.info("   GET  /health - Santé ultra-détaillée tous services")
    logger.info("   GET  /search-service - Diagnostic approfondi Search Service")
    logger.info("   GET  /services-summary - Résumé synthétique")
    logger.info("🔧 Endpoints principaux:")
    logger.info("   POST /api/v1/search/search - Recherche de transactions")
    logger.info("   POST /api/v1/conversation/chat - Assistant IA")
    logger.info("   GET  /api/v1/sync - Synchronisation Bridge")
    logger.info("   POST /api/v1/users/register - Enregistrement utilisateur")
    logger.info("=" * 80)
    logger.info("✅ Application Harena prête avec diagnostic complet")

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