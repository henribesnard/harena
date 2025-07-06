"""
Point d'entrée principal du search_service - VERSION REFACTORISÉE.

Configuration entièrement centralisée via config_service.
Plus de dépendance à search_service/config - tout vient de config_service.

AMÉLIORATIONS:
- Configuration 100% centralisée via config_service
- Suppression complète de search_service/config
- Contrôle total via variables d'environnement
- Seuils de similarité configurables pour résoudre le problème
- Debug facilité avec endpoints de test
"""
import asyncio
import logging
import logging.config
import sys
import time
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional
import os

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ✅ CONFIGURATION CENTRALISÉE - SEULE SOURCE DE VÉRITÉ
try:
    from config_service.config import settings
except ImportError:
    print("❌ Impossible d'importer la configuration centralisée")
    sys.exit(1)

# Clients
from search_service.clients.elasticsearch_client import ElasticsearchClient
from search_service.clients.qdrant_client import QdrantClient

# Core services - Import des embeddings
try:
    from search_service.core.embeddings import EmbeddingService, EmbeddingManager
    search_embeddings_available = True
    print("✅ Classes d'embeddings importées depuis search_service.core.embeddings")
except ImportError as e:
    print(f"⚠️ Impossible d'importer depuis search_service.core.embeddings: {e}")
    search_embeddings_available = False

# Import fallback depuis enrichment_service
if not search_embeddings_available:
    try:
        from enrichment_service.core.embeddings import EmbeddingService as EnrichmentEmbeddingService
        enrichment_embeddings_available = True
        print("✅ Fallback: EmbeddingService importé depuis enrichment_service")
    except ImportError:
        print("⚠️ Impossible d'importer EmbeddingService depuis enrichment_service")
        enrichment_embeddings_available = False

    # Modèles locaux de fallback
    from dataclasses import dataclass
    from enum import Enum

    class EmbeddingModel(str, Enum):
        TEXT_EMBEDDING_3_SMALL = "text-embedding-3-small"
        TEXT_EMBEDDING_3_LARGE = "text-embedding-3-large"
        TEXT_EMBEDDING_ADA_002 = "text-embedding-ada-002"

    @dataclass
    class EmbeddingConfig:
        model: EmbeddingModel = EmbeddingModel.TEXT_EMBEDDING_3_SMALL
        dimensions: int = 1536
        batch_size: int = 100
        max_tokens: int = 8191
        timeout: int = 30
        max_retries: int = 3

    class EmbeddingService:
        def __init__(self, api_key: str, config: EmbeddingConfig):
            self.api_key = api_key
            self.config = config
            self._service = None
            
        async def initialize(self):
            if enrichment_embeddings_available:
                self._service = EnrichmentEmbeddingService()
            
        async def generate_embedding(self, text: str, use_cache: bool = True) -> list[float]:
            if self._service:
                return await self._service.generate_embedding(text)
            else:
                return [0.0] * self.config.dimensions
        
        def get_dimensions(self) -> int:
            return self.config.dimensions

    class EmbeddingManager:
        def __init__(self, primary_service: EmbeddingService):
            self.primary_service = primary_service
            
        async def generate_embedding(self, text: str, use_cache: bool = True) -> list[float]:
            return await self.primary_service.generate_embedding(text, use_cache)

# Autres core services
from search_service.core.query_processor import QueryProcessor
from search_service.core.lexical_engine import LexicalSearchEngine, LexicalSearchConfig
from search_service.core.semantic_engine import SemanticSearchEngine, SemanticSearchConfig
from search_service.core.result_merger import ResultMerger, FusionConfig
from search_service.core.search_engine import HybridSearchEngine, HybridSearchConfig

# API
from search_service.api.routes import router
from search_service.api.dependencies import get_current_user, validate_search_request, rate_limit

# ==========================================
# 🔧 FONCTIONS DE CONFIGURATION CENTRALISÉES
# ==========================================

def get_logging_config():
    """Configuration des logs depuis settings centralisé."""
    level = "DEBUG" if settings.SEARCH_SERVICE_DEBUG else settings.LOG_LEVEL
    
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "detailed": {
                "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            },
            "simple": {
                "format": "%(levelname)s - %(message)s"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": level,
                "formatter": "detailed" if settings.DETAILED_LOGGING else "simple",
                "stream": "ext://sys.stdout"
            }
        },
        "loggers": {
            "search_service": {
                "level": level,
                "handlers": ["console"],
                "propagate": False
            },
            "search_service.core": {
                "level": level,
                "handlers": ["console"],
                "propagate": False
            },
            "search_service.clients": {
                "level": level,
                "handlers": ["console"],
                "propagate": False
            }
        },
        "root": {
            "level": "WARNING",
            "handlers": ["console"]
        }
    }


def get_elasticsearch_config():
    """Configuration Elasticsearch depuis settings centralisé."""
    return {
        "timeout": settings.ELASTICSEARCH_TIMEOUT,
        "max_results": settings.LEXICAL_MAX_RESULTS,
        "min_score": settings.LEXICAL_MIN_SCORE,
        "boost_config": {
            "exact_phrase": settings.BOOST_EXACT_PHRASE,
            "merchant_name": settings.BOOST_MERCHANT_NAME,
            "primary_description": settings.BOOST_PRIMARY_DESCRIPTION,
            "searchable_text": settings.BOOST_SEARCHABLE_TEXT,
            "clean_description": settings.BOOST_CLEAN_DESCRIPTION
        },
        "features": {
            "fuzzy": settings.ENABLE_FUZZY,
            "wildcards": settings.ENABLE_WILDCARDS,
            "synonyms": settings.ENABLE_SYNONYMS,
            "highlighting": settings.HIGHLIGHT_ENABLED
        },
        "highlighting": {
            "fragment_size": settings.HIGHLIGHT_FRAGMENT_SIZE,
            "max_fragments": settings.HIGHLIGHT_MAX_FRAGMENTS
        }
    }


def get_qdrant_config():
    """Configuration Qdrant depuis settings centralisé."""
    return {
        "url": settings.QDRANT_URL,
        "api_key": settings.QDRANT_API_KEY,
        "timeout": settings.QDRANT_TIMEOUT,
        "max_results": settings.SEMANTIC_MAX_RESULTS,
        "collection_name": settings.QDRANT_COLLECTION_NAME,
        "vector_size": settings.QDRANT_VECTOR_SIZE,
        "distance_metric": settings.QDRANT_DISTANCE_METRIC,
        "similarity_thresholds": {
            "strict": settings.SIMILARITY_THRESHOLD_STRICT,
            "default": settings.SIMILARITY_THRESHOLD_DEFAULT,
            "loose": settings.SIMILARITY_THRESHOLD_LOOSE
        },
        "features": {
            "filtering": settings.SEMANTIC_ENABLE_FILTERING,
            "fallback_unfiltered": settings.SEMANTIC_FALLBACK_UNFILTERED,
            "recommendations": settings.RECOMMENDATION_ENABLED
        }
    }


def get_embedding_config():
    """Configuration des embeddings depuis settings centralisé."""
    return {
        "api_key": settings.OPENAI_API_KEY,
        "model": settings.EMBEDDING_MODEL,
        "dimensions": settings.EMBEDDING_DIMENSIONS,
        "batch_size": settings.EMBEDDING_BATCH_SIZE,
        "timeout": settings.OPENAI_TIMEOUT,
        "max_tokens": 8191,
        "max_retries": settings.MAX_RETRIES,
        "cache": {
            "enabled": settings.EMBEDDING_CACHE_ENABLED,
            "ttl": settings.EMBEDDING_CACHE_TTL,
            "max_size": settings.EMBEDDING_CACHE_MAX_SIZE
        }
    }


def get_hybrid_search_config():
    """Configuration de recherche hybride depuis settings centralisé."""
    return {
        "default_type": settings.DEFAULT_SEARCH_TYPE,
        "weights": {
            "lexical": settings.DEFAULT_LEXICAL_WEIGHT,
            "semantic": settings.DEFAULT_SEMANTIC_WEIGHT
        },
        "thresholds": {
            "similarity_default": settings.SIMILARITY_THRESHOLD_DEFAULT,
            "similarity_strict": settings.SIMILARITY_THRESHOLD_STRICT,
            "similarity_loose": settings.SIMILARITY_THRESHOLD_LOOSE,
            "lexical_min_score": settings.LEXICAL_MIN_SCORE,
            "semantic_min_score": settings.MIN_SEMANTIC_SCORE
        },
        "limits": {
            "default": settings.DEFAULT_SEARCH_LIMIT,
            "max": settings.MAX_SEARCH_LIMIT,
            "per_engine": settings.MAX_RESULTS_PER_ENGINE
        },
        "timeout": settings.SEARCH_TIMEOUT,
        "fusion_options": {
            "score_normalization": settings.SCORE_NORMALIZATION_METHOD,
            "min_score_threshold": settings.MIN_SCORE_THRESHOLD,
            "rrf_k": settings.RRF_K,
            "adaptive_threshold": settings.ADAPTIVE_THRESHOLD,
            "quality_boost": settings.QUALITY_BOOST_FACTOR,
            "deduplication": settings.ENABLE_DEDUPLICATION,
            "dedup_threshold": settings.DEDUP_SIMILARITY_THRESHOLD,
            "diversification": settings.ENABLE_DIVERSIFICATION,
            "diversity_factor": settings.DIVERSITY_FACTOR,
            "max_same_merchant": settings.MAX_SAME_MERCHANT
        }
    }


# Setup logging
logging.config.dictConfig(get_logging_config())
logger = logging.getLogger(__name__)

# Variables globales pour les services
elasticsearch_client: Optional[ElasticsearchClient] = None
qdrant_client: Optional[QdrantClient] = None
embedding_service: Optional[EmbeddingService] = None
embedding_manager: Optional[EmbeddingManager] = None
query_processor: Optional[QueryProcessor] = None
lexical_engine: Optional[LexicalSearchEngine] = None
semantic_engine: Optional[SemanticSearchEngine] = None
result_merger: Optional[ResultMerger] = None
hybrid_engine: Optional[HybridSearchEngine] = None

# Métriques de démarrage
startup_time: Optional[float] = None
initialization_results: Dict[str, Any] = {}

# ==========================================
# 🚀 CYCLE DE VIE DE L'APPLICATION
# ==========================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestion du cycle de vie de l'application."""
    global startup_time
    startup_time = time.time()
    
    # Afficher la configuration critique au démarrage
    logger.info("🔍 Search Service - Configuration centralisée chargée:")
    logger.info(f"   📊 Seuils de similarité: loose={settings.SIMILARITY_THRESHOLD_LOOSE}, default={settings.SIMILARITY_THRESHOLD_DEFAULT}, strict={settings.SIMILARITY_THRESHOLD_STRICT}")
    logger.info(f"   ⚖️ Poids hybride: lexical={settings.DEFAULT_LEXICAL_WEIGHT}, semantic={settings.DEFAULT_SEMANTIC_WEIGHT}")
    logger.info(f"   🎯 Limites: default={settings.DEFAULT_SEARCH_LIMIT}, max={settings.MAX_SEARCH_LIMIT}")
    logger.info(f"   💾 Cache: search={settings.SEARCH_CACHE_ENABLED}, embedding={settings.EMBEDDING_CACHE_ENABLED}")
    logger.info(f"   🚀 Debug: {settings.SEARCH_SERVICE_DEBUG}")
    
    # Validation de la configuration
    validation = settings.validate_search_config()
    if not validation["valid"]:
        logger.error(f"❌ Configuration invalide: {validation['errors']}")
        for error in validation["errors"]:
            logger.error(f"   • {error}")
    
    if validation["warnings"]:
        logger.warning("⚠️ Warnings de configuration:")
        for warning in validation["warnings"]:
            logger.warning(f"   • {warning}")
    
    logger.info("🚀 Démarrage du Search Service...")
    
    # Startup
    try:
        await startup_event()
        logger.info("✅ Search Service démarré avec succès")
    except Exception as e:
        logger.error(f"❌ Échec du démarrage: {e}", exc_info=True)
        raise
    
    try:
        yield
    finally:
        # Shutdown
        await shutdown_event()
        logger.info("🛑 Search Service arrêté")


async def startup_event():
    """Initialisation complète au démarrage."""
    global elasticsearch_client, qdrant_client, embedding_service, embedding_manager
    global query_processor, lexical_engine, semantic_engine, result_merger, hybrid_engine
    
    try:
        # 1. Initialiser les clients
        await initialize_clients()
        
        # 2. Initialiser les services d'embeddings
        await initialize_embedding_services()
        
        # 3. Initialiser les moteurs de recherche
        await initialize_search_engines()
        
        # 4. Initialiser le moteur hybride
        await initialize_hybrid_engine()
        
        # 5. Injecter dans les routes
        inject_dependencies_into_routes()
        
        # 6. Vérifications de santé
        await perform_health_checks()
        
        # 7. Warmup optionnel
        if settings.WARMUP_ENABLED:
            await warmup_search_engines()
        
        logger.info("🎉 Initialisation complète terminée")
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'initialisation: {e}", exc_info=True)
        raise RuntimeError(f"Search Service initialization failed: {e}")


async def initialize_clients():
    """Initialise les clients Elasticsearch et Qdrant."""
    global elasticsearch_client, qdrant_client
    
    logger.info("🔌 Initialisation des clients...")
    
    # Client Elasticsearch
    try:
        if settings.BONSAI_URL:
            es_config = get_elasticsearch_config()
            elasticsearch_client = ElasticsearchClient(
                url=settings.BONSAI_URL,
                **es_config
            )
            
            await elasticsearch_client.health()
            logger.info("✅ Client Elasticsearch connecté")
            initialization_results["elasticsearch"] = {"status": "success", "connected": True}
        else:
            logger.warning("⚠️ BONSAI_URL non configuré")
            initialization_results["elasticsearch"] = {"status": "disabled", "reason": "BONSAI_URL not configured"}
        
    except Exception as e:
        logger.error(f"❌ Échec Elasticsearch: {e}")
        initialization_results["elasticsearch"] = {"status": "failed", "error": str(e)}
        elasticsearch_client = None
    
    # Client Qdrant
    try:
        if settings.QDRANT_URL and settings.QDRANT_API_KEY:
            qdrant_config = get_qdrant_config()
            qdrant_client = QdrantClient(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY,
                **qdrant_config
            )
            
            await qdrant_client.health_check()
            logger.info("✅ Client Qdrant connecté")
            initialization_results["qdrant"] = {"status": "success", "connected": True}
        else:
            logger.warning("⚠️ QDRANT_URL ou QDRANT_API_KEY non configuré")
            initialization_results["qdrant"] = {"status": "disabled", "reason": "QDRANT credentials not configured"}
        
    except Exception as e:
        logger.error(f"❌ Échec Qdrant: {e}")
        initialization_results["qdrant"] = {"status": "failed", "error": str(e)}
        qdrant_client = None
    
    if not elasticsearch_client and not qdrant_client:
        raise RuntimeError("Aucun client de base de données disponible")


async def initialize_embedding_services():
    """Initialise les services d'embeddings."""
    global embedding_service, embedding_manager
    
    logger.info("🤖 Initialisation des services d'embeddings...")
    
    if not settings.OPENAI_API_KEY:
        logger.warning("⚠️ OPENAI_API_KEY non configurée - mode dégradé")
        initialization_results["embeddings"] = {"status": "disabled", "reason": "OPENAI_API_KEY not configured"}
        return
    
    try:
        embedding_config = get_embedding_config()
        
        if search_embeddings_available:
            # Utiliser les vraies classes
            embedding_service = EmbeddingService(api_key=settings.OPENAI_API_KEY)
            logger.info("✅ EmbeddingService créé depuis search_service.core.embeddings")
        else:
            # Mode fallback
            config_obj = EmbeddingConfig(
                model=EmbeddingModel(embedding_config["model"]),
                dimensions=embedding_config["dimensions"],
                batch_size=embedding_config["batch_size"],
                timeout=embedding_config["timeout"],
                max_retries=embedding_config["max_retries"]
            )
            embedding_service = EmbeddingService(
                api_key=settings.OPENAI_API_KEY,
                config=config_obj
            )
            await embedding_service.initialize()
            logger.info("✅ EmbeddingService créé en mode fallback")
        
        # Créer le manager
        embedding_manager = EmbeddingManager(embedding_service)
        
        # Test de validation
        test_embedding = await embedding_service.generate_embedding("test", use_cache=False)
        if not test_embedding or len(test_embedding) == 0:
            raise ValueError("Test embedding failed")
        
        logger.info(f"✅ Services d'embeddings initialisés ({len(test_embedding)} dimensions)")
        initialization_results["embeddings"] = {
            "status": "success",
            "model": embedding_config["model"],
            "dimensions": len(test_embedding),
            "test_successful": True
        }
        
    except Exception as e:
        logger.error(f"❌ Échec embeddings: {e}")
        embedding_service = None
        embedding_manager = None
        initialization_results["embeddings"] = {"status": "failed", "error": str(e)}


async def initialize_search_engines():
    """Initialise les moteurs de recherche."""
    global query_processor, lexical_engine, semantic_engine, result_merger
    
    logger.info("🔍 Initialisation des moteurs de recherche...")
    
    # Query processor
    try:
        query_processor = QueryProcessor()
        logger.info("✅ Query processor initialisé")
        initialization_results["query_processor"] = {"status": "success"}
    except Exception as e:
        logger.error(f"❌ Échec query processor: {e}")
        initialization_results["query_processor"] = {"status": "failed", "error": str(e)}
        raise
    
    # Moteur lexical
    if elasticsearch_client:
        try:
            lexical_config = LexicalSearchConfig()
            lexical_engine = LexicalSearchEngine(
                elasticsearch_client=elasticsearch_client,
                query_processor=query_processor,
                config=lexical_config
            )
            
            health = await lexical_engine.health_check()
            if health["status"] == "healthy":
                logger.info("✅ Moteur lexical initialisé")
                initialization_results["lexical_engine"] = {"status": "success"}
            else:
                raise Exception(f"Moteur lexical non sain: {health}")
                
        except Exception as e:
            logger.error(f"❌ Échec moteur lexical: {e}")
            initialization_results["lexical_engine"] = {"status": "failed", "error": str(e)}
            lexical_engine = None
    else:
        logger.warning("⚠️ Moteur lexical désactivé (Elasticsearch non disponible)")
        initialization_results["lexical_engine"] = {"status": "disabled"}
    
    # Moteur sémantique
    if qdrant_client and embedding_manager:
        try:
            semantic_config = SemanticSearchConfig()
            semantic_engine = SemanticSearchEngine(
                qdrant_client=qdrant_client,
                embedding_manager=embedding_manager,
                query_processor=query_processor,
                config=semantic_config
            )
            
            health = await semantic_engine.health_check()
            if health["status"] in ["healthy", "degraded"]:
                logger.info("✅ Moteur sémantique initialisé")
                initialization_results["semantic_engine"] = {"status": "success"}
            else:
                raise Exception(f"Moteur sémantique non sain: {health}")
                
        except Exception as e:
            logger.error(f"❌ Échec moteur sémantique: {e}")
            initialization_results["semantic_engine"] = {"status": "failed", "error": str(e)}
            semantic_engine = None
    else:
        reason = "Qdrant ou embeddings non disponibles"
        logger.warning(f"⚠️ Moteur sémantique désactivé ({reason})")
        initialization_results["semantic_engine"] = {"status": "disabled", "reason": reason}
    
    # Result merger
    try:
        fusion_config = FusionConfig()
        result_merger = ResultMerger(config=fusion_config)
        logger.info("✅ Result merger initialisé")
        initialization_results["result_merger"] = {"status": "success"}
    except Exception as e:
        logger.error(f"❌ Échec result merger: {e}")
        initialization_results["result_merger"] = {"status": "failed", "error": str(e)}
        raise


async def initialize_hybrid_engine():
    """Initialise le moteur hybride."""
    global hybrid_engine
    
    logger.info("🎯 Initialisation du moteur hybride...")
    
    if not lexical_engine and not semantic_engine:
        raise RuntimeError("Aucun moteur disponible pour le hybride")
    
    try:
        hybrid_config = HybridSearchConfig(**get_hybrid_search_config())
        
        hybrid_engine = HybridSearchEngine(
            lexical_engine=lexical_engine,
            semantic_engine=semantic_engine,
            query_processor=query_processor,
            result_merger=result_merger,
            config=hybrid_config
        )
        
        health = await hybrid_engine.health_check()
        available_engines = sum(
            1 for engine_health in health["engines"].values()
            if engine_health.get("status") in ["healthy", "degraded"]
        )
        
        if available_engines > 0:
            logger.info(f"✅ Moteur hybride initialisé ({available_engines} moteurs)")
            initialization_results["hybrid_engine"] = {
                "status": "success",
                "available_engines": available_engines
            }
        else:
            raise Exception("Aucun moteur sous-jacent disponible")
        
    except Exception as e:
        logger.error(f"❌ Échec moteur hybride: {e}")
        initialization_results["hybrid_engine"] = {"status": "failed", "error": str(e)}
        raise


def inject_dependencies_into_routes():
    """Injecte les services dans les routes."""
    logger.info("💉 Injection des dépendances...")
    
    try:
        from search_service.api import routes
        
        routes.elasticsearch_client = elasticsearch_client
        routes.qdrant_client = qdrant_client
        routes.embedding_manager = embedding_manager
        routes.query_processor = query_processor
        routes.lexical_engine = lexical_engine
        routes.semantic_engine = semantic_engine
        routes.result_merger = result_merger
        routes.hybrid_engine = hybrid_engine
        
        # Injection de la configuration centralisée
        routes.settings = settings
        
        successful = sum(1 for x in [
            elasticsearch_client, qdrant_client, embedding_manager,
            query_processor, lexical_engine, semantic_engine,
            result_merger, hybrid_engine
        ] if x is not None)
        
        logger.info(f"✅ Dépendances injectées: {successful}/8 composants")
        initialization_results["dependency_injection"] = {
            "status": "success",
            "components_injected": successful
        }
        
    except Exception as e:
        logger.error(f"❌ Échec injection: {e}")
        initialization_results["dependency_injection"] = {"status": "failed", "error": str(e)}
        raise


async def perform_health_checks():
    """Vérifications de santé."""
    logger.info("🏥 Vérifications de santé...")
    
    health_results = {}
    
    if elasticsearch_client:
        try:
            es_health = await elasticsearch_client.health()
            health_results["elasticsearch"] = es_health
        except Exception as e:
            health_results["elasticsearch"] = {"status": "unhealthy", "error": str(e)}
    
    if qdrant_client:
        try:
            qdrant_health = await qdrant_client.health_check()
            health_results["qdrant"] = qdrant_health
        except Exception as e:
            health_results["qdrant"] = {"status": "unhealthy", "error": str(e)}
    
    if embedding_service:
        try:
            test_embedding = await embedding_service.generate_embedding("test", use_cache=False)
            health_results["embeddings"] = {
                "status": "healthy" if test_embedding else "unhealthy",
                "test_successful": bool(test_embedding)
            }
        except Exception as e:
            health_results["embeddings"] = {"status": "unhealthy", "error": str(e)}
    
    if hybrid_engine:
        try:
            hybrid_health = await hybrid_engine.health_check()
            health_results["hybrid_engine"] = hybrid_health
        except Exception as e:
            health_results["hybrid_engine"] = {"status": "unhealthy", "error": str(e)}
    
    initialization_results["health_checks"] = health_results
    
    healthy = sum(1 for h in health_results.values() if h.get("status") in ["healthy", "degraded"])
    logger.info(f"✅ Health checks: {healthy}/{len(health_results)} sains")


async def warmup_search_engines():
    """Warmup des moteurs."""
    logger.info("🔥 Warmup des moteurs...")
    
    if not hybrid_engine:
        return
    
    try:
        warmup_queries = settings.get_warmup_queries_list()
        successful = 0
        
        for query in warmup_queries:
            try:
                await hybrid_engine.search(query, user_id=1, limit=1)
                successful += 1
            except Exception as e:
                logger.warning(f"Warmup failed for '{query}': {e}")
        
        initialization_results["warmup"] = {
            "status": "completed",
            "successful": successful,
            "total": len(warmup_queries)
        }
        
        logger.info(f"✅ Warmup: {successful}/{len(warmup_queries)} requêtes")
        
    except Exception as e:
        logger.error(f"❌ Échec warmup: {e}")
        initialization_results["warmup"] = {"status": "failed", "error": str(e)}


async def shutdown_event():
    """Nettoyage à l'arrêt."""
    logger.info("🛑 Arrêt du service...")
    
    if elasticsearch_client:
        try:
            await elasticsearch_client.close()
            logger.info("✅ Elasticsearch fermé")
        except Exception as e:
            logger.error(f"❌ Erreur fermeture Elasticsearch: {e}")
    
    if qdrant_client:
        try:
            await qdrant_client.close()
            logger.info("✅ Qdrant fermé")
        except Exception as e:
            logger.error(f"❌ Erreur fermeture Qdrant: {e}")
    
    if hybrid_engine:
        try:
            hybrid_engine.clear_cache()
            logger.info("✅ Caches vidés")
        except Exception as e:
            logger.error(f"❌ Erreur vidage cache: {e}")


# ==========================================
# 🌐 APPLICATION FASTAPI
# ==========================================

def create_search_app() -> FastAPI:
    """Crée l'application FastAPI."""
    
    app = FastAPI(
        title="Harena Search Service",
        description="Service de recherche hybride centralisé",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs" if settings.SEARCH_SERVICE_DEBUG else None,
        redoc_url="/redoc" if settings.SEARCH_SERVICE_DEBUG else None
    )
    
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.SEARCH_SERVICE_DEBUG else [settings.CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["*"]
    )
    
    # Routes principales
    app.include_router(router, prefix="/api/v1", tags=["search"])
    
    # ==========================================
    # 🔍 ENDPOINTS DE DEBUG ET DIAGNOSTIC
    # ==========================================
    
    @app.get("/health")
    async def health_check():
        """Santé globale du service."""
        uptime = time.time() - startup_time if startup_time else 0
        
        return {
            "status": "healthy",
            "service": "search_service",
            "version": "1.0.0",
            "uptime_seconds": round(uptime, 2),
            "configuration": {
                "similarity_thresholds": {
                    "default": settings.SIMILARITY_THRESHOLD_DEFAULT,
                    "strict": settings.SIMILARITY_THRESHOLD_STRICT,
                    "loose": settings.SIMILARITY_THRESHOLD_LOOSE
                },
                "search_limits": {
                    "default": settings.DEFAULT_SEARCH_LIMIT,
                    "max": settings.MAX_SEARCH_LIMIT
                },
                "cache_enabled": {
                    "search": settings.SEARCH_CACHE_ENABLED,
                    "embedding": settings.EMBEDDING_CACHE_ENABLED
                }
            },
            "initialization": initialization_results,
            "components": {
                "elasticsearch": elasticsearch_client is not None,
                "qdrant": qdrant_client is not None,
                "embeddings": embedding_manager is not None,
                "lexical_engine": lexical_engine is not None,
                "semantic_engine": semantic_engine is not None,
                "hybrid_engine": hybrid_engine is not None
            }
        }
    
    @app.get("/debug/config")
    async def debug_config():
        """🔧 Configuration complète pour debug."""
        if not settings.SEARCH_SERVICE_DEBUG:
            raise HTTPException(status_code=404, detail="Not found")
        
        return {
            "search_config": settings.get_search_config_summary(),
            "validation": settings.validate_search_config(),
            "environment_variables": {
                "SIMILARITY_THRESHOLD_DEFAULT": settings.SIMILARITY_THRESHOLD_DEFAULT,
                "SIMILARITY_THRESHOLD_STRICT": settings.SIMILARITY_THRESHOLD_STRICT,
                "SIMILARITY_THRESHOLD_LOOSE": settings.SIMILARITY_THRESHOLD_LOOSE,
                "MIN_SEMANTIC_SCORE": settings.MIN_SEMANTIC_SCORE,
                "LEXICAL_MIN_SCORE": settings.LEXICAL_MIN_SCORE,
                "DEFAULT_LEXICAL_WEIGHT": settings.DEFAULT_LEXICAL_WEIGHT,
                "DEFAULT_SEMANTIC_WEIGHT": settings.DEFAULT_SEMANTIC_WEIGHT,
                "SEARCH_CACHE_ENABLED": settings.SEARCH_CACHE_ENABLED,
                "EMBEDDING_CACHE_ENABLED": settings.EMBEDDING_CACHE_ENABLED,
                "WARMUP_ENABLED": settings.WARMUP_ENABLED,
                "SEARCH_SERVICE_DEBUG": settings.SEARCH_SERVICE_DEBUG
            },
            "apis_configured": {
                "OPENAI_API_KEY": bool(settings.OPENAI_API_KEY),
                "QDRANT_URL": bool(settings.QDRANT_URL),
                "QDRANT_API_KEY": bool(settings.QDRANT_API_KEY),
                "BONSAI_URL": bool(settings.BONSAI_URL)
            }
        }
    
    @app.get("/debug/embedding")
    async def debug_embedding():
        """🤖 Debug des services d'embeddings."""
        if not settings.SEARCH_SERVICE_DEBUG:
            raise HTTPException(status_code=404, detail="Not found")
        
        return {
            "embedding_service": {
                "exists": embedding_service is not None,
                "type": str(type(embedding_service)) if embedding_service else None,
                "has_generate_method": hasattr(embedding_service, 'generate_embedding') if embedding_service else False
            },
            "embedding_manager": {
                "exists": embedding_manager is not None,
                "type": str(type(embedding_manager)) if embedding_manager else None,
                "has_primary_service": hasattr(embedding_manager, 'primary_service') if embedding_manager else False,
                "primary_service_type": str(type(embedding_manager.primary_service)) if embedding_manager and hasattr(embedding_manager, 'primary_service') else None
            },
            "configuration": {
                "openai_api_key_configured": bool(settings.OPENAI_API_KEY),
                "openai_api_key_preview": f"{settings.OPENAI_API_KEY[:10]}..." if settings.OPENAI_API_KEY else None,
                "embedding_model": settings.EMBEDDING_MODEL,
                "embedding_dimensions": settings.EMBEDDING_DIMENSIONS,
                "embedding_batch_size": settings.EMBEDDING_BATCH_SIZE
            },
            "initialization_results": initialization_results.get("embeddings", {}),
            "semantic_engine": {
                "exists": semantic_engine is not None,
                "type": str(type(semantic_engine)) if semantic_engine else None
            }
        }
    
    @app.get("/debug/thresholds")
    async def debug_thresholds():
        """📊 Debug des seuils de similarité - CRITIQUE !"""
        if not settings.SEARCH_SERVICE_DEBUG:
            raise HTTPException(status_code=404, detail="Not found")
        
        return {
            "🚨_current_thresholds": {
                "similarity_threshold_default": settings.SIMILARITY_THRESHOLD_DEFAULT,
                "similarity_threshold_strict": settings.SIMILARITY_THRESHOLD_STRICT, 
                "similarity_threshold_loose": settings.SIMILARITY_THRESHOLD_LOOSE,
                "min_semantic_score": settings.MIN_SEMANTIC_SCORE,
                "lexical_min_score": settings.LEXICAL_MIN_SCORE
            },
            "🎯_threshold_modes": {
                "get_default": settings.get_similarity_threshold("default"),
                "get_strict": settings.get_similarity_threshold("strict"),
                "get_loose": settings.get_similarity_threshold("loose")
            },
            "💡_recommendations": {
                "if_no_results": "Essayez SIMILARITY_THRESHOLD_DEFAULT=0.1 dans .env",
                "for_debug": "Utilisez SIMILARITY_THRESHOLD_LOOSE=0.05 pour maximum de résultats",
                "for_production": "Utilisez SIMILARITY_THRESHOLD_DEFAULT=0.3 pour équilibré"
            },
            "⚠️_validation": settings.validate_search_config()
        }
    
    @app.post("/test/embedding")
    async def test_embedding_endpoint(text: str = "test query"):
        """🧪 Test de génération d'embedding."""
        if not settings.SEARCH_SERVICE_DEBUG:
            raise HTTPException(status_code=404, detail="Not found")
        
        if not embedding_manager:
            raise HTTPException(status_code=503, detail="Embedding service not available")
        
        try:
            start_time = time.time()
            embedding = await embedding_manager.generate_embedding(text, use_cache=False)
            end_time = time.time()
            
            return {
                "success": True,
                "text": text,
                "embedding_dimensions": len(embedding) if embedding else 0,
                "processing_time_ms": round((end_time - start_time) * 1000, 2),
                "embedding_preview": embedding[:5] if embedding else None,
                "service_type": str(type(embedding_manager.primary_service)) if embedding_manager else None
            }
        except Exception as e:
            return {
                "success": False,
                "text": text,
                "error": str(e),
                "error_type": str(type(e))
            }
    
    @app.post("/test/semantic-search")
    async def test_semantic_search_endpoint(
        query: str = "test", 
        user_id: int = 1, 
        similarity_threshold: Optional[float] = None
    ):
        """🔍 Test de recherche sémantique avec seuil configurable."""
        if not settings.SEARCH_SERVICE_DEBUG:
            raise HTTPException(status_code=404, detail="Not found")
        
        if not semantic_engine:
            raise HTTPException(status_code=503, detail="Semantic search not available")
        
        # Utiliser le seuil configuré ou celui par défaut
        threshold = similarity_threshold or settings.SIMILARITY_THRESHOLD_DEFAULT
        
        try:
            start_time = time.time()
            results = await semantic_engine.search(
                query=query, 
                user_id=user_id, 
                limit=5,
                similarity_threshold=threshold
            )
            end_time = time.time()
            
            return {
                "success": True,
                "query": query,
                "user_id": user_id,
                "similarity_threshold_used": threshold,
                "similarity_threshold_from_config": settings.SIMILARITY_THRESHOLD_DEFAULT,
                "results_count": len(results.results) if hasattr(results, 'results') else 0,
                "processing_time_ms": round((end_time - start_time) * 1000, 2),
                "results": results.results if hasattr(results, 'results') else results,
                "total_found": getattr(results, 'total_found', 0),
                "quality": getattr(results, 'quality', None)
            }
        except Exception as e:
            return {
                "success": False,
                "query": query,
                "similarity_threshold_used": threshold,
                "error": str(e),
                "error_type": str(type(e)),
                "suggestion": "Essayez avec similarity_threshold=0.1 ou moins"
            }
    
    @app.post("/test/threshold-comparison")
    async def test_threshold_comparison(query: str = "restaurant", user_id: int = 1):
        """⚖️ Compare les résultats avec différents seuils."""
        if not settings.SEARCH_SERVICE_DEBUG:
            raise HTTPException(status_code=404, detail="Not found")
        
        if not semantic_engine:
            raise HTTPException(status_code=503, detail="Semantic search not available")
        
        thresholds = {
            "strict": settings.SIMILARITY_THRESHOLD_STRICT,
            "default": settings.SIMILARITY_THRESHOLD_DEFAULT,
            "loose": settings.SIMILARITY_THRESHOLD_LOOSE,
            "very_loose": 0.05,
            "ultra_loose": 0.01
        }
        
        results = {}
        
        for mode, threshold in thresholds.items():
            try:
                start_time = time.time()
                search_result = await semantic_engine.search(
                    query=query,
                    user_id=user_id,
                    limit=10,
                    similarity_threshold=threshold
                )
                end_time = time.time()
                
                results[mode] = {
                    "threshold": threshold,
                    "results_count": len(search_result.results) if hasattr(search_result, 'results') else 0,
                    "total_found": getattr(search_result, 'total_found', 0),
                    "processing_time_ms": round((end_time - start_time) * 1000, 2),
                    "quality": getattr(search_result, 'quality', None),
                    "success": True
                }
            except Exception as e:
                results[mode] = {
                    "threshold": threshold,
                    "success": False,
                    "error": str(e)
                }
        
        # Recommandation basée sur les résultats
        best_threshold = None
        for mode, result in results.items():
            if result.get("success") and result.get("results_count", 0) > 0:
                best_threshold = mode
                break
        
        return {
            "query": query,
            "user_id": user_id,
            "threshold_comparison": results,
            "recommendation": {
                "best_threshold_mode": best_threshold,
                "best_threshold_value": thresholds.get(best_threshold) if best_threshold else None,
                "suggestion": f"Utilisez SIMILARITY_THRESHOLD_DEFAULT={thresholds.get(best_threshold)} dans .env" if best_threshold else "Aucun seuil ne donne de résultats - vérifiez vos données"
            }
        }
    
    @app.get("/info")
    async def service_info():
        """ℹ️ Informations détaillées du service."""
        if not settings.SEARCH_SERVICE_DEBUG:
            raise HTTPException(status_code=404, detail="Not found")
        
        metrics = {}
        if hybrid_engine:
            try:
                metrics = hybrid_engine.get_metrics()
            except:
                metrics = {"error": "Metrics not available"}
        
        return {
            "service": "search_service",
            "version": "1.0.0",
            "configuration_source": "config_service (centralized)",
            "initialization_results": initialization_results,
            "metrics": metrics,
            "performance_summary": hybrid_engine.get_performance_summary() if hybrid_engine else None,
            "components_status": {
                "elasticsearch_client": {
                    "available": elasticsearch_client is not None,
                    "type": str(type(elasticsearch_client)) if elasticsearch_client else None
                },
                "qdrant_client": {
                    "available": qdrant_client is not None,
                    "type": str(type(qdrant_client)) if qdrant_client else None
                },
                "embedding_service": {
                    "available": embedding_service is not None,
                    "type": str(type(embedding_service)) if embedding_service else None
                },
                "embedding_manager": {
                    "available": embedding_manager is not None,
                    "type": str(type(embedding_manager)) if embedding_manager else None
                },
                "lexical_engine": {
                    "available": lexical_engine is not None,
                    "type": str(type(lexical_engine)) if lexical_engine else None
                },
                "semantic_engine": {
                    "available": semantic_engine is not None,
                    "type": str(type(semantic_engine)) if semantic_engine else None
                },
                "hybrid_engine": {
                    "available": hybrid_engine is not None,
                    "type": str(type(hybrid_engine)) if hybrid_engine else None
                }
            }
        }
    
    # Gestionnaire d'erreurs global
    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        logger.error(f"Erreur non gérée: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "detail": str(exc) if settings.SEARCH_SERVICE_DEBUG else "Une erreur est survenue",
                "service": "search_service",
                "suggestion": "Vérifiez /debug/config si en mode debug"
            }
        )
    
    return app


# ==========================================
# 🔧 FONCTIONS UTILITAIRES
# ==========================================

def get_search_app() -> FastAPI:
    """Alias pour create_search_app."""
    return create_search_app()


def get_embedding_manager():
    """Retourne l'embedding manager actuel."""
    return embedding_manager


def get_semantic_engine():
    """Retourne le moteur sémantique actuel."""
    return semantic_engine


def get_initialization_results():
    """Retourne les résultats d'initialisation."""
    return initialization_results


def get_service_status():
    """Résumé du statut des services."""
    return {
        "elasticsearch_available": elasticsearch_client is not None,
        "qdrant_available": qdrant_client is not None,
        "embeddings_available": embedding_manager is not None,
        "semantic_search_available": semantic_engine is not None,
        "lexical_search_available": lexical_engine is not None,
        "hybrid_search_available": hybrid_engine is not None,
        "configuration_source": "config_service",
        "similarity_thresholds": {
            "default": settings.SIMILARITY_THRESHOLD_DEFAULT,
            "strict": settings.SIMILARITY_THRESHOLD_STRICT,
            "loose": settings.SIMILARITY_THRESHOLD_LOOSE
        },
        "initialization_results": initialization_results
    }


# ==========================================
# 🚀 POINT D'ENTRÉE PRINCIPAL
# ==========================================

if __name__ == "__main__":
    app = create_search_app()
    
    # Affichage des informations de démarrage
    print("=" * 60)
    print("🔍 HARENA SEARCH SERVICE")
    print("=" * 60)
    print(f"📊 Seuils de similarité:")
    print(f"   • Default: {settings.SIMILARITY_THRESHOLD_DEFAULT}")
    print(f"   • Strict:  {settings.SIMILARITY_THRESHOLD_STRICT}")
    print(f"   • Loose:   {settings.SIMILARITY_THRESHOLD_LOOSE}")
    print(f"⚖️ Poids hybride: lexical={settings.DEFAULT_LEXICAL_WEIGHT}, semantic={settings.DEFAULT_SEMANTIC_WEIGHT}")
    print(f"🎯 Limites: default={settings.DEFAULT_SEARCH_LIMIT}, max={settings.MAX_SEARCH_LIMIT}")
    print(f"💾 Cache: search={settings.SEARCH_CACHE_ENABLED}, embedding={settings.EMBEDDING_CACHE_ENABLED}")
    print(f"🚀 Debug: {settings.SEARCH_SERVICE_DEBUG}")
    print(f"🌍 Environment: {settings.ENVIRONMENT}")
    print("=" * 60)
    
    if settings.SEARCH_SERVICE_DEBUG:
        print("🔧 Mode DEBUG activé - Endpoints disponibles:")
        print("   • /health - Santé du service")
        print("   • /debug/config - Configuration complète")
        print("   • /debug/embedding - Debug embeddings")
        print("   • /debug/thresholds - Debug seuils similarité")
        print("   • /test/embedding - Test génération embedding")
        print("   • /test/semantic-search - Test recherche sémantique")
        print("   • /test/threshold-comparison - Compare différents seuils")
        print("   • /info - Informations détaillées")
        print("=" * 60)
    
    # Configuration de démarrage
    if settings.SEARCH_SERVICE_DEBUG:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8003,
            reload=True,
            log_level="info"
        )
    else:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=int(os.getenv("PORT", 8003)),
            log_level="warning"
        )