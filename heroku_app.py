"""
Application Harena pour déploiement Heroku - Version mise à jour.

MODIFICATIONS APPORTÉES:
1. Enrichment Service mis à jour avec dual storage (Qdrant + Elasticsearch)
2. Search Service complètement supprimé pour réécriture
3. Tests de diagnostic améliorés pour tenir compte des nouvelles architectures
4. Gestion des injections directes pour l'Enrichment Service

SERVICES INCLUS:
- user_service: Gestion utilisateurs et authentification
- db_service: Base de données PostgreSQL
- sync_service: Synchronisation avec Bridge API
- enrichment_service: Enrichissement IA avec dual storage (NOUVEAU)
- conversation_service: Assistant IA conversationnel

SEARCH SERVICE: SUPPRIMÉ - Sera réécrit avec nouvelle architecture
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
    logger.info("🚀 Démarrage Harena Finance Platform - Version mise à jour")
    
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
    enrichment_service_initialization_result = None

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

    async def test_enrichment_service_dual_storage() -> Dict[str, Any]:
        """Test complet de l'Enrichment Service avec dual storage (Qdrant + Elasticsearch)."""
        logger.info("🔍 Test Enrichment Service (Dual Storage)...")
        result = {
            "service": "enrichment_service", 
            "healthy": False,
            "details": {
                "importable": False,
                "routes_count": 0,
                "dual_storage_config": {
                    "qdrant_url": bool(os.environ.get("QDRANT_URL")),
                    "elasticsearch_url": bool(os.environ.get("BONSAI_URL")),
                    "openai_key": bool(os.environ.get("OPENAI_API_KEY")),
                    "cohere_key": bool(os.environ.get("COHERE_KEY"))
                },
                "storage_initialization": {
                    "qdrant": False,
                    "elasticsearch": False,
                    "embeddings": False
                },
                "injection_status": {
                    "routes_injected": False,
                    "processors_available": False
                },
                "capabilities": {
                    "legacy_processing": False,
                    "dual_storage_processing": False,
                    "transaction_enrichment": False,
                    "user_sync": False
                },
                "endpoints": []
            },
            "recommendations": [],
            "error": None
        }
        
        try:
            # Test d'import des routes
            from enrichment_service.api.routes import router as enrichment_router
            result["details"]["importable"] = True
            result["details"]["routes_count"] = len(enrichment_router.routes) if hasattr(enrichment_router, 'routes') else 0
            
            # Extraire les endpoints disponibles
            if hasattr(enrichment_router, 'routes'):
                for route in enrichment_router.routes:
                    if hasattr(route, 'methods') and hasattr(route, 'path'):
                        for method in route.methods:
                            result["details"]["endpoints"].append(f"{method} {route.path}")
            
            # Test des composants d'initialisation
            try:
                # Test du service d'embeddings
                from enrichment_service.core.embeddings import embedding_service
                if hasattr(embedding_service, 'initialized') and embedding_service.initialized:
                    result["details"]["storage_initialization"]["embeddings"] = True
                    result["details"]["capabilities"]["transaction_enrichment"] = True
            except Exception as e:
                logger.warning(f"Embeddings service test failed: {e}")
            
            # Test Qdrant Storage
            try:
                from enrichment_service.storage.qdrant import QdrantStorage
                result["details"]["storage_initialization"]["qdrant"] = True
                result["details"]["capabilities"]["legacy_processing"] = True
            except Exception as e:
                logger.warning(f"Qdrant storage test failed: {e}")
            
            # Test Elasticsearch Client
            try:
                from enrichment_service.storage.elasticsearch_client import ElasticsearchClient
                result["details"]["storage_initialization"]["elasticsearch"] = True
            except Exception as e:
                logger.warning(f"Elasticsearch client test failed: {e}")
            
            # Test des processeurs
            try:
                from enrichment_service.core.processor import TransactionProcessor, DualStorageTransactionProcessor
                result["details"]["capabilities"]["legacy_processing"] = True
                result["details"]["capabilities"]["dual_storage_processing"] = True
                result["details"]["capabilities"]["user_sync"] = True
            except Exception as e:
                logger.warning(f"Processors test failed: {e}")
            
            # Vérifier l'injection dans les routes
            try:
                import enrichment_service.api.routes as routes
                injection_indicators = [
                    hasattr(routes, 'qdrant_storage'),
                    hasattr(routes, 'elasticsearch_client'),
                    hasattr(routes, 'transaction_processor'),
                    hasattr(routes, 'dual_processor')
                ]
                result["details"]["injection_status"]["routes_injected"] = any(injection_indicators)
                result["details"]["injection_status"]["processors_available"] = all([
                    hasattr(routes, 'transaction_processor'),
                    hasattr(routes, 'dual_processor')
                ])
            except Exception as e:
                logger.warning(f"Routes injection test failed: {e}")
            
            # Déterminer les capacités avancées
            dual_storage_ready = (
                result["details"]["storage_initialization"]["qdrant"] and 
                result["details"]["storage_initialization"]["elasticsearch"] and
                result["details"]["storage_initialization"]["embeddings"]
            )
            
            if dual_storage_ready:
                result["details"]["capabilities"]["dual_storage_processing"] = True
                
            # Service sain si importable et au moins le mode legacy fonctionne
            result["healthy"] = (
                result["details"]["importable"] and 
                result["details"]["routes_count"] > 0 and
                (result["details"]["capabilities"]["legacy_processing"] or 
                 result["details"]["capabilities"]["dual_storage_processing"])
            )
            
            # Générer des recommandations
            recommendations = []
            if not result["details"]["dual_storage_config"]["qdrant_url"]:
                recommendations.append("⚠️ Configurez QDRANT_URL pour le stockage vectoriel")
            if not result["details"]["dual_storage_config"]["elasticsearch_url"]:
                recommendations.append("⚠️ Configurez BONSAI_URL pour l'indexation Elasticsearch")
            if not result["details"]["dual_storage_config"]["openai_key"]:
                recommendations.append("⚠️ Configurez OPENAI_API_KEY pour les embeddings")
            if not result["details"]["injection_status"]["routes_injected"]:
                recommendations.append("🔧 Vérifiez l'injection des clients dans les routes")
            if dual_storage_ready:
                recommendations.append("✅ Dual storage complètement opérationnel")
            elif result["details"]["capabilities"]["legacy_processing"]:
                recommendations.append("⚠️ Mode legacy uniquement - considérez l'activation du dual storage")
            else:
                recommendations.append("🚨 Service en mode dégradé critique")
                
            result["recommendations"] = recommendations
                
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

    async def initialize_enrichment_service_dual_storage() -> Dict[str, Any]:
        """Initialise directement l'Enrichment Service avec dual storage."""
        logger.info("🔧 === INITIALISATION DIRECTE ENRICHMENT SERVICE (DUAL STORAGE) ===")
        
        initialization_result = {
            "service": "enrichment_service",
            "healthy": False,
            "method": "direct_dual_storage_initialization",
            "details": {
                "config_check": {
                    "qdrant_url": bool(os.environ.get("QDRANT_URL")),
                    "elasticsearch_url": bool(os.environ.get("BONSAI_URL")),
                    "openai_key": bool(os.environ.get("OPENAI_API_KEY"))
                },
                "components_initialized": {
                    "embeddings": False,
                    "qdrant_storage": False,
                    "elasticsearch_client": False,
                    "legacy_processor": False,
                    "dual_processor": False
                },
                "injection_successful": False,
                "initialization_time": 0
            },
            "recommendations": [],
            "error": None
        }
        
        start_time = time.time()
        
        try:
            # Import des composants nécessaires
            from enrichment_service.core.embeddings import embedding_service
            from enrichment_service.storage.qdrant import QdrantStorage
            from enrichment_service.storage.elasticsearch_client import ElasticsearchClient
            from enrichment_service.core.processor import TransactionProcessor, DualStorageTransactionProcessor
            
            # 1. Initialiser le service d'embeddings
            try:
                await embedding_service.initialize()
                initialization_result["details"]["components_initialized"]["embeddings"] = True
                logger.info("✅ Service d'embeddings initialisé")
            except Exception as e:
                logger.warning(f"⚠️ Embeddings initialization failed: {e}")
            
            # 2. Initialiser Qdrant Storage
            qdrant_storage = None
            try:
                qdrant_storage = QdrantStorage()
                await qdrant_storage.initialize()
                initialization_result["details"]["components_initialized"]["qdrant_storage"] = True
                logger.info("✅ Qdrant Storage initialisé")
            except Exception as e:
                logger.warning(f"⚠️ Qdrant initialization failed: {e}")
            
            # 3. Initialiser Elasticsearch Client
            elasticsearch_client = None
            try:
                elasticsearch_client = ElasticsearchClient()
                await elasticsearch_client.initialize()
                initialization_result["details"]["components_initialized"]["elasticsearch_client"] = True
                logger.info("✅ Elasticsearch Client initialisé")
            except Exception as e:
                logger.warning(f"⚠️ Elasticsearch initialization failed: {e}")
            
            # 4. Créer les processeurs
            transaction_processor = None
            dual_processor = None
            
            if qdrant_storage:
                try:
                    transaction_processor = TransactionProcessor(qdrant_storage)
                    initialization_result["details"]["components_initialized"]["legacy_processor"] = True
                    logger.info("✅ Legacy Transaction Processor créé")
                except Exception as e:
                    logger.warning(f"⚠️ Legacy processor creation failed: {e}")
            
            if qdrant_storage and elasticsearch_client:
                try:
                    dual_processor = DualStorageTransactionProcessor(qdrant_storage, elasticsearch_client)
                    initialization_result["details"]["components_initialized"]["dual_processor"] = True
                    logger.info("✅ Dual Storage Processor créé")
                except Exception as e:
                    logger.warning(f"⚠️ Dual processor creation failed: {e}")
            
            # 5. Injection dans les routes
            try:
                import enrichment_service.api.routes as routes
                
                routes.qdrant_storage = qdrant_storage
                routes.elasticsearch_client = elasticsearch_client
                routes.transaction_processor = transaction_processor
                routes.dual_processor = dual_processor
                
                initialization_result["details"]["injection_successful"] = True
                logger.info("✅ Injection dans les routes réussie")
                
            except Exception as e:
                logger.error(f"❌ Injection failed: {e}")
                initialization_result["error"] = f"Injection failed: {e}"
            
            # Déterminer le statut de santé
            any_processor = (
                initialization_result["details"]["components_initialized"]["legacy_processor"] or
                initialization_result["details"]["components_initialized"]["dual_processor"]
            )
            
            initialization_result["healthy"] = (
                initialization_result["details"]["injection_successful"] and
                any_processor
            )
            
            # Générer les recommandations
            recommendations = []
            if initialization_result["details"]["components_initialized"]["dual_processor"]:
                recommendations.append("✅ Dual storage complètement opérationnel")
            elif initialization_result["details"]["components_initialized"]["legacy_processor"]:
                recommendations.append("⚠️ Mode legacy uniquement - vérifiez Elasticsearch")
            else:
                recommendations.append("🚨 Aucun processeur disponible - service dégradé")
                
            if not initialization_result["details"]["config_check"]["qdrant_url"]:
                recommendations.append("⚠️ QDRANT_URL manquant")
            if not initialization_result["details"]["config_check"]["elasticsearch_url"]:
                recommendations.append("⚠️ BONSAI_URL manquant")
            if not initialization_result["details"]["config_check"]["openai_key"]:
                recommendations.append("⚠️ OPENAI_API_KEY manquant")
                
            initialization_result["recommendations"] = recommendations
            initialization_result["details"]["initialization_time"] = time.time() - start_time
            
        except Exception as e:
            logger.error(f"💥 Erreur générale initialisation enrichment: {e}")
            initialization_result["error"] = str(e)
            initialization_result["details"]["initialization_time"] = time.time() - start_time
        
        total_time = time.time() - start_time
        logger.info(f"🏁 Initialisation enrichment terminée en {total_time:.2f}s")
        
        return initialization_result

    async def run_complete_services_diagnostic() -> Dict[str, Any]:
        """Lance un diagnostic complet de tous les services (sans search_service)."""
        logger.info("🔍 Lancement du diagnostic complet de tous les services...")
        
        # Tests en parallèle pour les services standards
        standard_tests = await asyncio.gather(
            test_user_service(),
            test_db_service(), 
            test_sync_service(),
            test_conversation_service(),
            return_exceptions=True
        )
        
        services_status = {}
        standard_services = ["user_service", "db_service", "sync_service", "conversation_service"]
        
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
        
        # Traitement spécial pour l'Enrichment Service avec dual storage
        logger.info("🔧 Initialisation directe de l'Enrichment Service (Dual Storage)...")
        enrichment_result = await initialize_enrichment_service_dual_storage()
        services_status["enrichment_service"] = enrichment_result
        
        status_icon = "✅" if enrichment_result["healthy"] else "❌"
        logger.info(f"{status_icon} enrichment_service: {'Healthy' if enrichment_result['healthy'] else 'Unhealthy'}")
        
        if enrichment_result["healthy"]:
            components = enrichment_result["details"]["components_initialized"]
            logger.info(f"   🎯 Composants: Embeddings={components['embeddings']}, Qdrant={components['qdrant_storage']}, Elasticsearch={components['elasticsearch_client']}")
            logger.info(f"   🔧 Processeurs: Legacy={components['legacy_processor']}, Dual={components['dual_processor']}")
        
        global enrichment_service_initialization_result
        enrichment_service_initialization_result = enrichment_result
        
        # Ajouter une note sur le search_service
        services_status["search_service_note"] = {
            "status": "removed_for_rewrite",
            "message": "Search Service supprimé pour réécriture complète",
            "details": {
                "reason": "Architecture obsolète",
                "planned_replacement": "Nouveau système de recherche en développement",
                "temporary_solution": "Utiliser enrichment_service pour les fonctionnalités de base"
            }
        }
        
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
                        "registered_at": datetime.now().isoformat()
                    }
                    logger.info(f"✅ {name} enregistré: {routes_count} routes sur {prefix}")
                    return True
                else:
                    self.failed_services[name] = f"Router is None"
                    logger.error(f"❌ {name}: Router est None")
                    return False
            except Exception as e:
                self.failed_services[name] = str(e)
                logger.error(f"❌ {name}: {e}")
                return False
        
        def get_summary(self):
            return {
                "registered": len(self.services),
                "failed": len(self.failed_services),
                "total_routes": sum(s["routes_count"] for s in self.services.values()),
                "services": list(self.services.keys()),
                "failed_services": self.failed_services
            }

    service_registry = ServiceRegistry()

    # ==================== CYCLE DE VIE ====================

    async def startup():
        """Initialisation de l'application avec diagnostic complet."""
        global startup_time, all_services_status
        startup_time = time.time()
        logger.info("📋 Démarrage application Harena (sans Search Service)...")
        
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
        
        # Lancer le diagnostic complet
        all_services_status = await run_complete_services_diagnostic()
        
        total_time = time.time() - startup_time
        healthy_count = sum(1 for s in all_services_status.values() if s.get("healthy"))
        total_count = len([s for s in all_services_status.values() if s.get("service")]) # Exclure les notes
        logger.info(f"✅ Démarrage terminé en {total_time:.2f}s - {healthy_count}/{total_count} services sains")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await startup()
        yield

    # ==================== APPLICATION FASTAPI ====================

    app = FastAPI(
        title="Harena Finance Platform",
        description="Plateforme de gestion financière avec enrichissement IA (Search Service en cours de réécriture)",
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

    # ==================== ENREGISTREMENT DES SERVICES ====================

    logger.info("📋 Enregistrement des services...")

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

    # 3. Enrichment Service (Dual Storage)
    try:
        from enrichment_service.api.routes import router as enrichment_router
        if service_registry.register("enrichment_service", enrichment_router, "/api/v1/enrichment", "🧠 Enrichissement IA avec dual storage (Qdrant + Elasticsearch)"):
            app.include_router(enrichment_router, prefix="/api/v1/enrichment", tags=["enrichment"])
    except Exception as e:
        logger.error(f"❌ Enrichment Service: {e}")

    # 4. Conversation Service
    try:
        from conversation_service.api.routes import router as conversation_router
        if service_registry.register("conversation_service", conversation_router, "/api/v1/conversation", "Assistant IA"):
            app.include_router(conversation_router, prefix="/api/v1/conversation", tags=["conversation"])
    except Exception as e:
        logger.error(f"❌ Conversation Service: {e}")

    # NOTE: Search Service intentionnellement supprimé pour réécriture

    # ==================== ENDPOINTS DE DIAGNOSTIC ====================

    @app.get("/")
    async def root():
        """Statut général avec focus spécial sur l'Enrichment Service."""
        uptime = time.time() - startup_time if startup_time else 0
        
        # Résumé global (exclure les notes)
        real_services = {k: v for k, v in all_services_status.items() if v.get("service")}
        healthy_services = [name for name, status in real_services.items() if status.get("healthy")]
        failed_services = [name for name, status in real_services.items() if not status.get("healthy")]
        
        # Focus spécial Enrichment Service avec données détaillées
        enrichment_status = all_services_status.get("enrichment_service", {})
        enrichment_details = enrichment_status.get("details", {})
        
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
            "enrichment_service": {
                "healthy": enrichment_status.get("healthy", False),
                "dual_storage_ready": (
                    enrichment_details.get("components_initialized", {}).get("qdrant_storage", False) and
                    enrichment_details.get("components_initialized", {}).get("elasticsearch_client", False)
                ),
                "embeddings_ready": enrichment_details.get("components_initialized", {}).get("embeddings", False),
                "processors_available": {
                    "legacy": enrichment_details.get("components_initialized", {}).get("legacy_processor", False),
                    "dual_storage": enrichment_details.get("components_initialized", {}).get("dual_processor", False)
                },
                "injection_successful": enrichment_details.get("injection_successful", False),
                "initialization_time": enrichment_details.get("initialization_time", 0),
                "endpoints_registered": "enrichment_service" in service_registry.get_summary().get("services", [])
            },
            "search_service_status": {
                "status": "removed_for_rewrite",
                "message": "Search Service supprimé pour refonte complète",
                "alternative": "Utilisez enrichment_service pour les fonctionnalités de base"
            },
            "timestamp": datetime.now().isoformat()
        }

    @app.get("/health")
    async def health_check():
        """Vérification de santé ultra-détaillée de tous les services."""
        uptime = time.time() - startup_time if startup_time else 0
        
        registry_summary = service_registry.get_summary()
        
        return {
            "status": "healthy" if all(s.get("healthy", False) for s in all_services_status.values() if s.get("service")) else "degraded",
            "uptime_seconds": round(uptime, 2),
            "timestamp": datetime.now().isoformat(),
            "registry": registry_summary,
            "services": all_services_status,
            "enrichment_service_initialization": enrichment_service_initialization_result
        }

    @app.get("/enrichment-service")
    async def enrichment_service_detailed():
        """Statut ultra-détaillé de l'Enrichment Service avec dual storage."""
        enrichment_status = all_services_status.get("enrichment_service", {})
        
        return {
            "service": "enrichment_service",
            "priority": "high",
            "timestamp": datetime.now().isoformat(),
            "overall_status": "fully_operational" if enrichment_status.get("healthy") else "degraded",
            "dual_storage_architecture": {
                "method": enrichment_status.get("method", "unknown"),
                "qdrant_storage": enrichment_status.get("details", {}).get("components_initialized", {}).get("qdrant_storage", False),
                "elasticsearch_client": enrichment_status.get("details", {}).get("components_initialized", {}).get("elasticsearch_client", False),
                "embeddings_service": enrichment_status.get("details", {}).get("components_initialized", {}).get("embeddings", False),
                "injection_successful": enrichment_status.get("details", {}).get("injection_successful", False)
            },
            "processors": {
                "legacy_processor": enrichment_status.get("details", {}).get("components_initialized", {}).get("legacy_processor", False),
                "dual_storage_processor": enrichment_status.get("details", {}).get("components_initialized", {}).get("dual_processor", False)
            },
            "configuration": enrichment_status.get("details", {}).get("config_check", {}),
            "initialization_metrics": {
                "initialization_time": enrichment_status.get("details", {}).get("initialization_time", 0),
                "method": "direct_dual_storage_initialization"
            },
            "recommendations": enrichment_status.get("recommendations", []),
            "error": enrichment_status.get("error"),
            "endpoints": [
                "POST /api/v1/enrichment/enrich/transaction - Enrichissement legacy (Qdrant)",
                "POST /api/v1/enrichment/enrich/batch - Enrichissement par lot legacy",
                "POST /api/v1/enrichment/dual/enrich-transaction - Enrichissement dual storage",
                "POST /api/v1/enrichment/dual/sync-user - Synchronisation utilisateur dual",
                "GET /api/v1/enrichment/dual/sync-status/{user_id} - Statut synchronisation",
                "GET /api/v1/enrichment/dual/health - Santé dual storage",
                "GET /enrichment-service - Ce diagnostic détaillé"
            ],
            "storage_note": "Service configuré avec dual storage (Qdrant + Elasticsearch) pour redondance et performance"
        }

    @app.get("/services-summary")
    async def services_summary():
        """Résumé synthétique de tous les services avec focus Enrichment Service."""
        real_services = {k: v for k, v in all_services_status.items() if v.get("service")}
        healthy_count = sum(1 for s in real_services.values() if s.get("healthy"))
        total_count = len(real_services)
        
        enrichment_status = all_services_status.get("enrichment_service", {})
        registry_summary = service_registry.get_summary()
        
        return {
            "summary": {
                "total_services": total_count,
                "healthy_services": healthy_count,
                "health_percentage": round((healthy_count / total_count) * 100, 1) if total_count > 0 else 0,
                "enrichment_service_critical": enrichment_status.get("healthy", False)
            },
            "registry": {
                "registered_routers": registry_summary["registered"],
                "failed_routers": registry_summary["failed"],
                "total_routes": registry_summary["total_routes"]
            },
            "enrichment_service_status": {
                "healthy": enrichment_status.get("healthy", False),
                "dual_storage_ready": (
                    enrichment_status.get("details", {}).get("components_initialized", {}).get("qdrant_storage", False) and
                    enrichment_status.get("details", {}).get("components_initialized", {}).get("elasticsearch_client", False)
                ),
                "endpoints_registered": "enrichment_service" in registry_summary.get("services", []),
                "error": enrichment_status.get("error")
            },
            "quick_diagnostics": {
                "database_configured": all_services_status.get("db_service", {}).get("healthy", False),
                "user_management": all_services_status.get("user_service", {}).get("healthy", False),
                "sync_available": all_services_status.get("sync_service", {}).get("healthy", False),
                "ai_features": all_services_status.get("conversation_service", {}).get("healthy", False),
                "enrichment_operational": enrichment_status.get("healthy", False),
                "search_service_note": "Supprimé pour réécriture complète"
            },
            "timestamp": datetime.now().isoformat()
        }

    @app.get("/search-service-status")
    async def search_service_status():
        """Informations sur le statut du Search Service."""
        return {
            "service": "search_service",
            "status": "removed_for_rewrite",
            "message": "Le Search Service a été intentionnellement supprimé pour une refonte complète",
            "details": {
                "reason": "Architecture obsolète nécessitant une réécriture",
                "planned_replacement": "Nouveau système de recherche en développement",
                "current_alternative": "Utilisez enrichment_service pour les fonctionnalités de base",
                "expected_timeline": "À déterminer selon les priorités"
            },
            "alternatives": {
                "enrichment_service": {
                    "available": all_services_status.get("enrichment_service", {}).get("healthy", False),
                    "endpoints": [
                        "POST /api/v1/enrichment/dual/enrich-transaction",
                        "GET /api/v1/enrichment/dual/health"
                    ]
                }
            },
            "timestamp": datetime.now().isoformat()
        }

    @app.get("/version")
    async def version():
        return {
            "version": "1.0.0",
            "build": "heroku-enrichment-dual-storage-no-search",
            "python": sys.version.split()[0],
            "environment": os.environ.get("ENVIRONMENT", "production"),
            "architecture_changes": [
                "✅ Enrichment Service mis à jour avec dual storage (Qdrant + Elasticsearch)",
                "🔧 Injection directe des clients dans les routes",
                "🚨 Search Service supprimé pour réécriture complète",
                "📊 Diagnostics améliorés pour dual storage",
                "⚡ Initialisation optimisée avec gestion d'erreurs"
            ],
            "services_included": [
                "user_service - Gestion utilisateurs et authentification",
                "db_service - Base de données PostgreSQL",
                "sync_service - Synchronisation Bridge API",
                "enrichment_service - Enrichissement IA dual storage",
                "conversation_service - Assistant IA conversationnel"
            ],
            "services_excluded": [
                "search_service - Supprimé pour refonte architecture"
            ]
        }

    @app.get("/robots.txt", include_in_schema=False)
    async def robots():
        return JSONResponse("User-agent: *\nDisallow: /", media_type="text/plain")

    # ==================== GESTIONNAIRE D'ERREURS ====================

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        enrichment_status = all_services_status.get("enrichment_service", {})
        real_services = {k: v for k, v in all_services_status.items() if v.get("service")}
        
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.detail,
                "path": request.url.path,
                "enrichment_service_healthy": enrichment_status.get("healthy", False),
                "healthy_services": [name for name, status in real_services.items() if status.get("healthy")],
                "search_service_note": "Search Service supprimé - utilisez enrichment_service",
                "timestamp": datetime.now().isoformat()
            }
        )

    # ==================== RAPPORT FINAL ====================

    logger.info("=" * 80)
    logger.info("🎯 HARENA FINANCE PLATFORM - VERSION MISE À JOUR")
    logger.info(f"📊 Services enregistrés: {service_registry.get_summary()['registered']}")
    logger.info(f"❌ Services échoués: {service_registry.get_summary()['failed']}")
    logger.info("🔧 Modifications importantes:")
    logger.info("   ✅ Enrichment Service avec dual storage (Qdrant + Elasticsearch)")
    logger.info("   🔧 Injection directe des clients d'enrichissement")
    logger.info("   🚨 Search Service SUPPRIMÉ pour réécriture complète")
    logger.info("   📊 Diagnostics améliorés pour dual storage")
    logger.info("   ⚡ Initialisation optimisée avec gestion d'erreurs robuste")
    logger.info("🌐 Endpoints de diagnostic:")
    logger.info("   GET  / - Statut général avec métriques Enrichment Service")
    logger.info("   GET  /health - Santé ultra-détaillée tous services")
    logger.info("   GET  /enrichment-service - Diagnostic approfondi Enrichment Service")
    logger.info("   GET  /search-service-status - Statut Search Service (supprimé)")
    logger.info("   GET  /services-summary - Résumé synthétique")
    logger.info("🔧 Endpoints principaux:")
    logger.info("   POST /api/v1/enrichment/enrich/transaction - Enrichissement legacy")
    logger.info("   POST /api/v1/enrichment/dual/enrich-transaction - Enrichissement dual storage")
    logger.info("   POST /api/v1/enrichment/dual/sync-user - Synchronisation dual storage")
    logger.info("   GET  /api/v1/enrichment/dual/health - Santé dual storage")
    logger.info("   POST /api/v1/conversation/chat - Assistant IA")
    logger.info("   GET  /api/v1/sync - Synchronisation Bridge")
    logger.info("   POST /api/v1/users/register - Enregistrement utilisateur")
    logger.info("⚠️  SEARCH SERVICE:")
    logger.info("   🚨 Intentionnellement supprimé pour réécriture complète")
    logger.info("   🔄 Utilisez enrichment_service pour les fonctionnalités de base")
    logger.info("   📋 Nouvelle architecture en cours de développement")
    logger.info("=" * 80)
    logger.info("✅ Application Harena prête avec Enrichment Service dual storage")

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