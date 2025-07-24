"""
Application Harena pour développement local.
Structure adaptée pour le nouveau conversation_service avec TinyBERT minimaliste.

✅ CONVERSATION SERVICE CORRIGÉ:
- Router direct dans local_app.py (pas de mount)
- TinyBERT partagé via app.state
- Fix du problème "Modèle TinyBERT non chargé"
"""

import logging
import os
import sys
from pathlib import Path
from fastapi import FastAPI, APIRouter, HTTPException
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
    """Chargeur de services avec nouveau conversation_service TinyBERT"""
    
    def __init__(self):
        self.services_status = {}
        self.search_service_initialized = False
        self.search_service_error = None
        self.conversation_service_initialized = False
        self.conversation_service_error = None
    
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
    
    async def initialize_conversation_service(self, app: FastAPI):
        """✅ NOUVEAU: Initialise le conversation_service TinyBERT minimaliste"""
        logger.info("🤖 Initialisation du conversation_service - TinyBERT Minimaliste")
        
        try:
            logger.info("🎯 Chargement TinyBERT Detector...")
            from conversation_service.intent_detector import TinyBERTDetector
            
            # ✅ Créer l'instance TinyBERT Detector
            detector = TinyBERTDetector()
            
            # ✅ Charger le modèle TinyBERT
            logger.info("🤖 Chargement modèle TinyBERT...")
            await detector.load_model()
            logger.info("✅ TinyBERT chargé et prêt")
            
            # ✅ Test fonctionnel du détecteur
            logger.info("🧪 Test fonctionnel TinyBERT...")
            test_intent, test_confidence, test_latency = await detector.detect_intent("quel est mon solde")
            logger.info(f"✅ Test réussi - Intent: {test_intent}, Confiance: {test_confidence:.3f}, Latence: {test_latency:.2f}ms")
            
            # ✅ Récupérer statistiques
            stats = detector.get_stats()
            logger.info(f"📊 TinyBERT Statistics:")
            logger.info(f"   - Modèle chargé: {stats['model_loaded']}")
            logger.info(f"   - Device: {stats['device']}")
            logger.info(f"   - Requêtes totales: {stats['total_requests']}")
            logger.info(f"   - Latence moyenne: {stats['average_latency_ms']:.2f}ms")
            
            # ✅ Mettre le détecteur dans app.state pour les routes
            app.state.conversation_service_initialized = True
            app.state.tinybert_detector = detector
            app.state.conversation_architecture = "tinybert_minimal"
            app.state.conversation_initialization_error = None
            
            self.conversation_service_initialized = True
            self.conversation_service_error = None
            
            logger.info("🎉 Conversation Service TinyBERT complètement initialisé!")
            return True
            
        except Exception as e:
            error_msg = f"Erreur initialisation conversation_service TinyBERT: {str(e)}"
            logger.error(f"❌ {error_msg}")
            
            # Marquer l'échec dans app.state
            app.state.conversation_service_initialized = False
            app.state.tinybert_detector = None
            app.state.conversation_architecture = "failed"
            app.state.conversation_initialization_error = error_msg
            
            self.conversation_service_initialized = False
            self.conversation_service_error = error_msg
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
    
    def load_conversation_service_router(self, app: FastAPI):
        """✅ SOLUTION CORRIGÉE: Créer les routes directement dans local_app"""
        try:
            logger.info("🔗 Création routes conversation_service TinyBERT...")
            
            from conversation_service.models import IntentRequest, IntentResponse, HealthResponse
            
            # Créer router
            router = APIRouter()
            
            @router.post("/detect-intent", response_model=IntentResponse)
            async def detect_intent(request: IntentRequest):
                """🎯 Détection intention financière"""
                try:
                    # Récupérer détecteur depuis app.state
                    detector = getattr(app.state, 'tinybert_detector', None)
                    if not detector:
                        raise HTTPException(status_code=500, detail="TinyBERT detector non disponible")
                    
                    if not detector.is_loaded:
                        raise HTTPException(status_code=500, detail="TinyBERT modèle non chargé")
                    
                    intent, confidence, processing_time_ms = await detector.detect_intent(request.query)
                    
                    response = IntentResponse(
                        intent=intent,
                        confidence=confidence,
                        processing_time_ms=processing_time_ms,
                        query=request.query
                    )
                    
                    logger.info(f"✅ Intent détecté: {intent} ({confidence:.3f}) en {processing_time_ms:.2f}ms")
                    return response
                    
                except HTTPException:
                    raise  # Re-raise HTTPException as-is
                except Exception as e:
                    logger.error(f"❌ Erreur détection: {e}")
                    raise HTTPException(status_code=500, detail=f"Erreur détection: {str(e)}")
            
            @router.get("/health", response_model=HealthResponse)
            async def health_check():
                """Santé du service"""
                try:
                    detector = getattr(app.state, 'tinybert_detector', None)
                    if not detector:
                        return HealthResponse(
                            status="unhealthy",
                            model_loaded=False,
                            total_requests=0,
                            average_latency_ms=0.0
                        )
                    
                    stats = detector.get_stats()
                    return HealthResponse(
                        status="healthy" if stats["model_loaded"] else "unhealthy",
                        model_loaded=stats["model_loaded"],
                        total_requests=stats["total_requests"],
                        average_latency_ms=stats["average_latency_ms"]
                    )
                except Exception as e:
                    logger.error(f"❌ Erreur health check: {e}")
                    return HealthResponse(
                        status="error",
                        model_loaded=False,
                        total_requests=0,
                        average_latency_ms=0.0
                    )
            
            @router.get("/")
            async def root():
                """Page d'accueil"""
                detector = getattr(app.state, 'tinybert_detector', None)
                model_info = {}
                
                if detector:
                    stats = detector.get_stats()
                    model_info = {
                        "model_loaded": stats["model_loaded"],
                        "total_requests": stats["total_requests"],
                        "average_latency_ms": stats["average_latency_ms"],
                        "device": stats["device"]
                    }
                
                return {
                    "service": "conversation_service",
                    "version": "1.0.0-tinybert",
                    "model": "TinyBERT",
                    "architecture": "minimaliste",
                    "endpoints": {
                        "detect_intent": "/detect-intent",
                        "health": "/health"
                    },
                    "model_info": model_info
                }
            
            # Enregistrer le router
            app.include_router(router, prefix="/api/v1/conversation", tags=["conversation"])
            routes_count = len(router.routes)
            
            logger.info(f"✅ conversation_service: {routes_count} routes créées directement")
            self.services_status["conversation_service"] = {
                "status": "ok", 
                "routes": routes_count,
                "prefix": "/api/v1/conversation",
                "architecture": "tinybert_direct_routes",
                "version": "1.0.0-tinybert",
                "initialized": True,
                "endpoints": [
                    "/api/v1/conversation/detect-intent",
                    "/api/v1/conversation/health", 
                    "/api/v1/conversation/"
                ]
            }
            return True
            
        except Exception as e:
            logger.error(f"❌ conversation_service routes: {str(e)}")
            self.services_status["conversation_service"] = {"status": "error", "error": str(e)}
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
    """Créer l'application FastAPI principale avec nouveau conversation_service"""
    
    app = FastAPI(
        title="Harena Finance Platform - Local Dev",
        description="Plateforme de gestion financière avec IA - Version développement avec TinyBERT",
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
        logger.info("🚀 Démarrage Harena Finance Platform - LOCAL DEV avec TinyBERT CORRIGÉ")
        
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

        # 5. ✅ CONVERSATION SERVICE CORRIGÉ - ROUTER DIRECT
        logger.info("🤖 Chargement et initialisation du conversation_service - TinyBERT CORRIGÉ...")
        try:
            # ÉTAPE 1: Initialiser le TinyBERT Detector
            conversation_init_success = await loader.initialize_conversation_service(app)
            
            # ÉTAPE 2: Créer les routes directement (SOLUTION CORRIGÉE)
            if conversation_init_success:
                router_success = loader.load_conversation_service_router(app)
                
                if router_success:
                    loader.services_status["conversation_service"]["initialized"] = True
                    logger.info("✅ conversation_service: TinyBERT complètement initialisé avec routes directes")
                else:
                    logger.error("❌ conversation_service: Initialisation OK mais routes non créées")
                    loader.services_status["conversation_service"] = {
                        "status": "degraded", 
                        "routes": 0, 
                        "prefix": "/api/v1/conversation",
                        "initialized": True,
                        "error": "Routes creation failed",
                        "architecture": "tinybert_direct_routes",
                        "version": "1.0.0-tinybert"
                    }
            else:
                logger.error("❌ conversation_service: Initialisation TinyBERT échouée")
                loader.services_status["conversation_service"] = {
                    "status": "error", 
                    "error": loader.conversation_service_error,
                    "architecture": "tinybert_direct_routes",
                    "version": "1.0.0-tinybert"
                }
                        
        except Exception as e:
            logger.error(f"❌ conversation_service: Erreur générale TinyBERT - {str(e)}")
            loader.services_status["conversation_service"] = {
                "status": "error", 
                "error": str(e),
                "architecture": "tinybert_direct_routes",
                "version": "1.0.0-tinybert"
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
        
        logger.info("🎉 Plateforme Harena complètement déployée avec TinyBERT CORRIGÉ!")

    @app.get("/health")
    async def health():
        """Health check global avec nouveau conversation_service"""
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
                "version": conversation_status.get("version"),
                "endpoints": conversation_status.get("endpoints", [])
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
        """Statut détaillé avec nouveau conversation_service"""
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
                "architecture": "tinybert_direct_routes",
                "version": "1.0.0-tinybert-fixed",
                "model": "TinyBERT",
                "fix_applied": "Router direct dans local_app.py - TinyBERT partagé via app.state",
                "features": [
                    "Détection intentions financières ultra-rapide",
                    "Router direct (pas de mount) - CORRIGÉ",
                    "TinyBERT partagé via app.state - CORRIGÉ",
                    "Mesure précise latence < 50ms",
                    "Métriques intégrées temps réel",
                    "Prêt pour fine-tuning données françaises"
                ],
                "endpoints": {
                    "main": "/api/v1/conversation/detect-intent",
                    "health": "/api/v1/conversation/health",
                    "root": "/api/v1/conversation/"
                },
                "performance_targets": {
                    "latency_ms": "< 50",
                    "accuracy": "> 70%",
                    "model_size": "TinyBERT (50MB)",
                    "memory_usage": "< 500MB"
                }
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
        """Page d'accueil avec conversation_service TinyBERT CORRIGÉ"""
        return {
            "message": "🏦 Harena Finance Platform - LOCAL DEVELOPMENT avec TinyBERT CORRIGÉ",
            "version": "1.0.0-dev-fixed",
            "services_available": [
                "user_service - Gestion utilisateurs",
                "sync_service - Synchronisation Bridge API", 
                "enrichment_service - Enrichissement Elasticsearch (v2.0)",
                "search_service - Recherche lexicale (Architecture simplifiée)",
                "conversation_service - TinyBERT Détection Intentions CORRIGÉ (<50ms)"
            ],
            "conversation_service_fixed": {
                "description": "Service TinyBERT avec router direct CORRIGÉ",
                "architecture": "Input → TinyBERT (via app.state) → Intent Classification → JSON Response",
                "main_endpoint": "/api/v1/conversation/detect-intent",
                "performance_target": "< 50ms latency, > 70% accuracy",
                "fix_applied": {
                    "problem": "Modèle TinyBERT non chargé - mount app isolation",
                    "solution": "Router direct dans local_app.py + TinyBERT via app.state",
                    "status": "CORRIGÉ ✅"
                },
                "capabilities": [
                    "Détection intentions financières",
                    "Accès direct au modèle TinyBERT",
                    "Mesure précise latence",
                    "Métriques temps réel",
                    "Prêt fine-tuning"
                ],
                "test_query": {
                    "url": "POST /api/v1/conversation/detect-intent",
                    "example": '{"query": "quel est mon solde"}',
                    "expected_response": {
                        "intent": "BALANCE_CHECK",
                        "confidence": 0.87,
                        "processing_time_ms": 37.2,
                        "query": "quel est mon solde",
                        "model": "TinyBERT"
                    }
                }
            },
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
                "/api/v1/conversation/detect-intent": "🎯 TinyBERT Détection Intentions CORRIGÉ",
                "/api/v1/conversation/health": "Santé TinyBERT CORRIGÉ",
                "/api/v1/conversation/": "Info conversation service CORRIGÉ"
            },
            "development_mode": {
                "hot_reload": True,
                "debug_logs": True,
                "local_services": ["PostgreSQL", "Redis", "Elasticsearch (requis pour enrichment + search)"],
                "docs_url": "http://localhost:8000/docs",
                "tinybert_test": "http://localhost:8000/api/v1/conversation/detect-intent"
            },
            "architecture_updates": {
                "conversation_service": {
                    "version": "1.0.0-tinybert-fixed",
                    "architecture": "router_direct_dans_local_app",
                    "fix_details": {
                        "problem_original": "app.mount() isolait TinyBERT dans sous-application",
                        "solution_applied": "Router direct avec accès app.state.tinybert_detector",
                        "benefits": [
                            "TinyBERT partagé entre local_app et routes",
                            "Pas d'isolation de contexte",
                            "Accès direct au modèle chargé",
                            "Gestion d'erreurs robuste"
                        ]
                    },
                    "changes": [
                        "🔧 FIX: Router direct au lieu de app.mount()",
                        "🔧 FIX: TinyBERT via app.state au lieu d'instance locale",
                        "🔧 FIX: Gestion d'erreurs HTTP explicites",
                        "✅ Architecture ultra-minimaliste conservée",
                        "✅ Un seul endpoint: /detect-intent",
                        "✅ Mesure précise latence (<50ms)",
                        "✅ Métriques temps réel intégrées"
                    ],
                    "performance_goals": {
                        "latency": "< 50ms (objectif: < 30ms)",
                        "accuracy": "> 70% intentions financières",
                        "memory": "< 500MB RAM",
                        "startup": "< 10s chargement modèle"
                    },
                    "supported_intents": [
                        "BALANCE_CHECK - Consultation soldes",
                        "TRANSFER - Virements", 
                        "EXPENSE_ANALYSIS - Analyse dépenses",
                        "CARD_MANAGEMENT - Gestion cartes",
                        "GREETING - Salutations",
                        "HELP - Aide",
                        "GOODBYE - Au revoir",
                        "UNKNOWN - Non reconnu"
                    ],
                    "next_steps": [
                        "1. Tester avec requêtes réelles (MAINTENANT POSSIBLE)",
                        "2. Mesurer latence et précision",
                        "3. Fine-tuner sur données françaises",
                        "4. Optimiser si < 70% précision"
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
                }
            },
            "testing_conversation_service": {
                "status": "MAINTENANT FONCTIONNEL ✅",
                "quick_test": {
                    "command": "curl -X POST http://localhost:8000/api/v1/conversation/detect-intent -H 'Content-Type: application/json' -d '{\"query\": \"quel est mon solde\"}'",
                    "expected_response": {
                        "intent": "BALANCE_CHECK",
                        "confidence": "0.8+",
                        "processing_time_ms": "< 50",
                        "success": "true"
                    }
                },
                "test_queries": [
                    "quel est mon solde",
                    "faire un virement de 100 euros",
                    "mes dépenses ce mois",
                    "bloquer ma carte",
                    "bonjour",
                    "aide",
                    "au revoir"
                ],
                "performance_monitoring": {
                    "health_endpoint": "GET /api/v1/conversation/health",
                    "metrics": [
                        "total_requests",
                        "average_latency_ms", 
                        "model_loaded",
                        "device (cpu/cuda)"
                    ]
                },
                "troubleshooting": {
                    "previous_error": "Modèle TinyBERT non chargé",
                    "fix_applied": "Router direct + app.state",
                    "status": "RÉSOLU ✅"
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
    logger.info("🤖 TinyBERT Conversation Service CORRIGÉ: http://localhost:8000/api/v1/conversation/")
    logger.info("🎯 Endpoint principal CORRIGÉ: http://localhost:8000/api/v1/conversation/detect-intent")
    logger.info("📊 Santé TinyBERT CORRIGÉ: http://localhost:8000/api/v1/conversation/health")
    logger.info("🧪 Test rapide MAINTENANT FONCTIONNEL: curl -X POST http://localhost:8000/api/v1/conversation/detect-intent -H 'Content-Type: application/json' -d '{\"query\": \"quel est mon solde\"}'")
    logger.info("✅ PROBLÈME 'Modèle TinyBERT non chargé' RÉSOLU avec router direct!")
    
    uvicorn.run(
        "local_app:app", 
        host="0.0.0.0", 
        port=8000,
        reload=True,
        log_level="info"
    )