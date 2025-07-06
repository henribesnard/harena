"""
Moteurs de recherche et composants core - VERSION CENTRALISÉE.

Ce module expose tous les composants principaux du service de recherche hybride
avec gestion d'erreurs gracieuse et configuration centralisée via config_service.

CENTRALISÉ VIA CONFIG_SERVICE:
- Toutes les configurations viennent de config_service.config.settings
- Import sécurisé avec gestion d'erreurs pour tous les composants
- Exports conditionnels selon la disponibilité des modules
- Documentation des composants disponibles
- Fallbacks gracieux en cas d'échec d'import
"""

import logging
from typing import Dict, Any

# ✅ CONFIGURATION CENTRALISÉE - SEULE SOURCE DE VÉRITÉ
from config_service.config import settings

logger = logging.getLogger(__name__)

# ==================== IMPORTS SÉCURISÉS AVEC CONFIG CENTRALISÉE ====================

# 1. Embeddings (priorité absolue - toujours nécessaires)
try:
    from .embeddings import (
        EmbeddingModel,
        EmbeddingConfig,
        EmbeddingService, 
        EmbeddingManager,
        create_embedding_service,
        create_embedding_manager,
        get_global_embedding_service
    )
    EMBEDDINGS_AVAILABLE = True
    logger.debug("✅ Embeddings components imported successfully (centralized config)")
except ImportError as e:
    logger.error(f"❌ Failed to import embeddings: {e}")
    EMBEDDINGS_AVAILABLE = False

# 2. Query Processing
try:
    from .query_processor import (
        QueryProcessor, 
        QueryAnalysis, 
        QueryValidator,
        create_query_processor_with_config,
        get_query_processor_config
    )
    QUERY_PROCESSOR_AVAILABLE = True
    logger.debug("✅ Query processor components imported successfully (centralized config)")
except ImportError as e:
    logger.warning(f"⚠️ Query processor not available: {e}")
    QUERY_PROCESSOR_AVAILABLE = False

# 3. Lexical Search Engine
try:
    from .lexical_engine import (
        LexicalSearchEngine, 
        LexicalSearchConfig, 
        LexicalSearchResult
    )
    LEXICAL_ENGINE_AVAILABLE = True
    logger.debug("✅ Lexical engine components imported successfully (centralized config)")
except ImportError as e:
    logger.warning(f"⚠️ Lexical engine not available: {e}")
    LEXICAL_ENGINE_AVAILABLE = False

# 4. Semantic Search Engine
try:
    from .semantic_engine import (
        SemanticSearchEngine, 
        SemanticSearchConfig, 
        SemanticSearchResult
    )
    SEMANTIC_ENGINE_AVAILABLE = True
    logger.debug("✅ Semantic engine components imported successfully (centralized config)")
except ImportError as e:
    logger.warning(f"⚠️ Semantic engine not available: {e}")
    SEMANTIC_ENGINE_AVAILABLE = False

# 5. Hybrid Search Engine
try:
    from .search_engine import (
        HybridSearchEngine, 
        HybridSearchConfig, 
        HybridSearchResult
    )
    HYBRID_ENGINE_AVAILABLE = True
    logger.debug("✅ Hybrid engine components imported successfully (centralized config)")
except ImportError as e:
    logger.warning(f"⚠️ Hybrid engine not available: {e}")
    HYBRID_ENGINE_AVAILABLE = False

# 6. Result Merger (optionnel)
try:
    from .result_merger import (
        ResultMerger, 
        FusionConfig,
        FusionStrategy,
        FusionResult,
        create_result_merger,
        create_result_merger_with_preset,
        get_fusion_config_summary
    )
    RESULT_MERGER_AVAILABLE = True
    logger.debug("✅ Result merger components imported successfully (centralized config)")
except ImportError as e:
    logger.debug(f"ℹ️ Result merger not available: {e}")
    RESULT_MERGER_AVAILABLE = False

# 7. Reranker (optionnel)
try:
    from .reranker import (
        Reranker, 
        RerankerConfig,
        RerankerModel,
        create_reranker
    )
    RERANKER_AVAILABLE = True
    logger.debug("✅ Reranker components imported successfully")
except ImportError as e:
    logger.debug(f"ℹ️ Reranker not available: {e}")
    RERANKER_AVAILABLE = False

# ==================== CONSTRUCTION DYNAMIQUE DES EXPORTS ====================

# Liste des exports de base (toujours inclus si embeddings disponibles)
__all__ = []

# Ajout conditionnel des exports selon la disponibilité
if EMBEDDINGS_AVAILABLE:
    __all__.extend([
        # Classes et enums d'embeddings
        "EmbeddingModel",
        "EmbeddingConfig",
        "EmbeddingService",
        "EmbeddingManager",
        
        # Factory functions d'embeddings
        "create_embedding_service",
        "create_embedding_manager",
        "get_global_embedding_service",
    ])

if QUERY_PROCESSOR_AVAILABLE:
    __all__.extend([
        "QueryProcessor",
        "QueryAnalysis", 
        "QueryValidator",
        "create_query_processor_with_config",
        "get_query_processor_config",
    ])

if LEXICAL_ENGINE_AVAILABLE:
    __all__.extend([
        "LexicalSearchEngine",
        "LexicalSearchConfig",
        "LexicalSearchResult",
    ])

if SEMANTIC_ENGINE_AVAILABLE:
    __all__.extend([
        "SemanticSearchEngine", 
        "SemanticSearchConfig",
        "SemanticSearchResult",
    ])

if HYBRID_ENGINE_AVAILABLE:
    __all__.extend([
        "HybridSearchEngine",
        "HybridSearchConfig",
        "HybridSearchResult"
    ])

if RESULT_MERGER_AVAILABLE:
    __all__.extend([
        "ResultMerger",
        "FusionConfig",
        "FusionStrategy", 
        "FusionResult",
        "create_result_merger",
        "create_result_merger_with_preset",
        "get_fusion_config_summary"
    ])

if RERANKER_AVAILABLE:
    __all__.extend([
        "Reranker",
        "RerankerConfig",
        "RerankerModel",
        "create_reranker"
    ])

# ==================== INFORMATIONS SUR LA DISPONIBILITÉ ====================

COMPONENTS_STATUS = {
    "embeddings": EMBEDDINGS_AVAILABLE,
    "query_processor": QUERY_PROCESSOR_AVAILABLE,
    "lexical_engine": LEXICAL_ENGINE_AVAILABLE,
    "semantic_engine": SEMANTIC_ENGINE_AVAILABLE,
    "hybrid_engine": HYBRID_ENGINE_AVAILABLE,
    "result_merger": RESULT_MERGER_AVAILABLE,
    "reranker": RERANKER_AVAILABLE,
}

# Nombre de composants disponibles
AVAILABLE_COMPONENTS_COUNT = sum(COMPONENTS_STATUS.values())
TOTAL_COMPONENTS_COUNT = len(COMPONENTS_STATUS)

# ==================== FONCTIONS UTILITAIRES AVEC CONFIG CENTRALISÉE ====================

def get_available_components() -> dict:
    """Retourne la liste des composants disponibles."""
    return {
        component: available 
        for component, available in COMPONENTS_STATUS.items() 
        if available
    }

def get_unavailable_components() -> dict:
    """Retourne la liste des composants non disponibles."""
    return {
        component: available 
        for component, available in COMPONENTS_STATUS.items() 
        if not available
    }

def is_component_available(component_name: str) -> bool:
    """Vérifie si un composant spécifique est disponible."""
    return COMPONENTS_STATUS.get(component_name, False)

def get_components_summary() -> dict:
    """Retourne un résumé de l'état des composants avec info config centralisée."""
    available = get_available_components()
    unavailable = get_unavailable_components()
    
    return {
        "total_components": TOTAL_COMPONENTS_COUNT,
        "available_count": AVAILABLE_COMPONENTS_COUNT,
        "availability_rate": round(AVAILABLE_COMPONENTS_COUNT / TOTAL_COMPONENTS_COUNT * 100, 1),
        "available_components": list(available.keys()),
        "unavailable_components": list(unavailable.keys()),
        "critical_components_available": {
            "embeddings": EMBEDDINGS_AVAILABLE,
            "search_engines": LEXICAL_ENGINE_AVAILABLE or SEMANTIC_ENGINE_AVAILABLE or HYBRID_ENGINE_AVAILABLE
        },
        "service_operational": EMBEDDINGS_AVAILABLE and (
            LEXICAL_ENGINE_AVAILABLE or SEMANTIC_ENGINE_AVAILABLE or HYBRID_ENGINE_AVAILABLE
        ),
        "config_source": "centralized (config_service)",
        "centralized_settings": {
            "search_timeout": settings.SEARCH_TIMEOUT,
            "elasticsearch_timeout": settings.ELASTICSEARCH_TIMEOUT,
            "qdrant_timeout": settings.QDRANT_TIMEOUT,
            "cache_enabled": settings.SEARCH_CACHE_ENABLED,
            "cache_size": settings.SEARCH_CACHE_SIZE,
            "default_limit": settings.DEFAULT_LIMIT,
            "fusion_strategy": settings.DEFAULT_FUSION_STRATEGY,
            "adaptive_weighting": settings.ADAPTIVE_WEIGHTING
        }
    }

def get_centralized_config_summary() -> Dict[str, Any]:
    """Retourne un résumé complet de la configuration centralisée."""
    config_summary = {
        "config_source": "centralized (config_service)",
        "search_service": {
            "default_limit": settings.DEFAULT_LIMIT,
            "search_timeout": settings.SEARCH_TIMEOUT,
            "adaptive_weighting": settings.ADAPTIVE_WEIGHTING,
            "enable_fallback": settings.ENABLE_FALLBACK,
            "enable_parallel_search": settings.ENABLE_PARALLEL_SEARCH
        },
        "elasticsearch": {
            "timeout": settings.ELASTICSEARCH_TIMEOUT,
            "index": settings.ELASTICSEARCH_INDEX,
            "max_results": settings.LEXICAL_MAX_RESULTS,
            "min_score": settings.LEXICAL_MIN_SCORE
        },
        "qdrant": {
            "timeout": settings.QDRANT_TIMEOUT,
            "collection_name": settings.QDRANT_COLLECTION_NAME,
            "vector_size": settings.QDRANT_VECTOR_SIZE,
            "max_results": settings.SEMANTIC_MAX_RESULTS
        },
        "embeddings": {
            "model": settings.OPENAI_EMBEDDING_MODEL,
            "timeout": settings.OPENAI_TIMEOUT,
            "batch_size": settings.EMBEDDING_BATCH_SIZE,
            "cache_size": settings.EMBEDDING_CACHE_SIZE
        },
        "cache": {
            "search_cache_enabled": settings.SEARCH_CACHE_ENABLED,
            "search_cache_size": settings.SEARCH_CACHE_SIZE,
            "search_cache_ttl": settings.SEARCH_CACHE_TTL,
            "embedding_cache_enabled": settings.EMBEDDING_CACHE_ENABLED,
            "embedding_cache_size": settings.EMBEDDING_CACHE_SIZE
        },
        "fusion": {
            "default_strategy": settings.DEFAULT_FUSION_STRATEGY,
            "default_lexical_weight": settings.DEFAULT_LEXICAL_WEIGHT,
            "default_semantic_weight": settings.DEFAULT_SEMANTIC_WEIGHT,
            "rrf_k": settings.RRF_K,
            "quality_boost_factor": settings.QUALITY_BOOST_FACTOR,
            "enable_deduplication": settings.ENABLE_DEDUPLICATION,
            "enable_diversification": settings.ENABLE_DIVERSIFICATION
        },
        "quality": {
            "excellent_threshold": settings.QUALITY_EXCELLENT_THRESHOLD,
            "good_threshold": settings.QUALITY_GOOD_THRESHOLD,
            "medium_threshold": settings.QUALITY_MEDIUM_THRESHOLD,
            "poor_threshold": settings.QUALITY_POOR_THRESHOLD,
            "min_results_for_good_quality": settings.MIN_RESULTS_FOR_GOOD_QUALITY
        }
    }
    
    # Ajouter les configs spécifiques des composants disponibles
    if QUERY_PROCESSOR_AVAILABLE:
        config_summary["query_processor"] = get_query_processor_config()
    
    if RESULT_MERGER_AVAILABLE:
        config_summary["result_merger"] = get_fusion_config_summary()
    
    return config_summary

# ==================== VALIDATION ET DIAGNOSTICS ====================

def validate_core_setup() -> dict:
    """Valide la configuration des composants core avec info config centralisée."""
    validation_result = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "recommendations": [],
        "config_source": "centralized (config_service)"
    }
    
    # Vérifications critiques
    if not EMBEDDINGS_AVAILABLE:
        validation_result["valid"] = False
        validation_result["errors"].append("Embeddings components are missing - critical for search functionality")
    
    # Vérifications importantes
    search_engines_available = LEXICAL_ENGINE_AVAILABLE or SEMANTIC_ENGINE_AVAILABLE or HYBRID_ENGINE_AVAILABLE
    if not search_engines_available:
        validation_result["valid"] = False
        validation_result["errors"].append("No search engines available - service cannot function")
    
    # Avertissements
    if not QUERY_PROCESSOR_AVAILABLE:
        validation_result["warnings"].append("Query processor not available - basic query processing only")
    
    if not HYBRID_ENGINE_AVAILABLE and LEXICAL_ENGINE_AVAILABLE and SEMANTIC_ENGINE_AVAILABLE:
        validation_result["warnings"].append("Hybrid engine not available but both lexical and semantic engines are - consider fixing hybrid engine")
    
    # Vérifications de configuration centralisée
    try:
        # Vérifier les variables critiques
        if not hasattr(settings, 'OPENAI_API_KEY') or not settings.OPENAI_API_KEY:
            validation_result["warnings"].append("OPENAI_API_KEY not configured - embeddings may fail")
        
        if not hasattr(settings, 'ELASTICSEARCH_URL') or not settings.ELASTICSEARCH_URL:
            validation_result["warnings"].append("ELASTICSEARCH_URL not configured - lexical search may fail")
        
        if not hasattr(settings, 'QDRANT_URL') or not settings.QDRANT_URL:
            validation_result["warnings"].append("QDRANT_URL not configured - semantic search may fail")
        
        # Vérifier la cohérence des timeouts
        if settings.SEARCH_TIMEOUT <= max(settings.ELASTICSEARCH_TIMEOUT, settings.QDRANT_TIMEOUT):
            validation_result["warnings"].append("SEARCH_TIMEOUT should be higher than individual engine timeouts")
        
        # Vérifier les poids de fusion
        total_weight = settings.DEFAULT_LEXICAL_WEIGHT + settings.DEFAULT_SEMANTIC_WEIGHT
        if abs(total_weight - 1.0) > 0.01:
            validation_result["warnings"].append(f"Fusion weights don't sum to 1.0 (current: {total_weight})")
        
    except Exception as e:
        validation_result["warnings"].append(f"Error checking centralized configuration: {e}")
    
    # Recommandations
    if AVAILABLE_COMPONENTS_COUNT < TOTAL_COMPONENTS_COUNT:
        missing_count = TOTAL_COMPONENTS_COUNT - AVAILABLE_COMPONENTS_COUNT
        validation_result["recommendations"].append(
            f"Consider fixing {missing_count} missing components for full functionality"
        )
    
    if not RESULT_MERGER_AVAILABLE and LEXICAL_ENGINE_AVAILABLE and SEMANTIC_ENGINE_AVAILABLE:
        validation_result["recommendations"].append("Result merger would improve search quality when using multiple engines")
    
    if not RERANKER_AVAILABLE:
        validation_result["recommendations"].append("Reranker would further improve result relevance")
    
    # Recommandations de configuration
    validation_result["recommendations"].append("All configuration is centralized via config_service - update settings there")
    
    return validation_result

# ==================== LOGGING DE L'ÉTAT D'INITIALISATION ====================

# Log de l'état d'initialisation avec info config centralisée
summary = get_components_summary()
if summary["service_operational"]:
    logger.info(f"✅ Search service core initialized with centralized config: {summary['available_count']}/{summary['total_components']} components available ({summary['availability_rate']}%)")
    if summary["unavailable_components"]:
        logger.info(f"ℹ️ Missing optional components: {', '.join(summary['unavailable_components'])}")
    logger.info(f"🎛️ Configuration source: {summary['config_source']}")
else:
    logger.error(f"❌ Search service core initialization failed: critical components missing")
    validation = validate_core_setup()
    for error in validation["errors"]:
        logger.error(f"   - {error}")

# Ajout des fonctions utilitaires aux exports
__all__.extend([
    "COMPONENTS_STATUS",
    "get_available_components",
    "get_unavailable_components", 
    "is_component_available",
    "get_components_summary",
    "get_centralized_config_summary",
    "validate_core_setup"
])