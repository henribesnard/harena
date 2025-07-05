"""
Moteurs de recherche et composants core - VERSION CORRIGÉE.

Ce module expose tous les composants principaux du service de recherche hybride
avec gestion d'erreurs gracieuse et imports conditionnels pour éviter les crashes.

CORRECTIONS APPORTÉES:
- Import sécurisé avec gestion d'erreurs pour tous les composants
- Import correct de EmbeddingConfig depuis embeddings.py (maintenant disponible)
- Exports conditionnels selon la disponibilité des modules
- Documentation des composants disponibles
- Fallbacks gracieux en cas d'échec d'import
"""

import logging

logger = logging.getLogger(__name__)

# ==================== IMPORTS SÉCURISÉS ====================

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
    logger.debug("✅ Embeddings components imported successfully")
except ImportError as e:
    logger.error(f"❌ Failed to import embeddings: {e}")
    EMBEDDINGS_AVAILABLE = False

# 2. Query Processing
try:
    from .query_processor import QueryProcessor, QueryAnalysis, QueryValidator
    QUERY_PROCESSOR_AVAILABLE = True
    logger.debug("✅ Query processor components imported successfully")
except ImportError as e:
    logger.warning(f"⚠️ Query processor not available: {e}")
    QUERY_PROCESSOR_AVAILABLE = False

# 3. Lexical Search Engine
try:
    from .lexical_engine import LexicalSearchEngine, LexicalSearchConfig, LexicalSearchResult
    LEXICAL_ENGINE_AVAILABLE = True
    logger.debug("✅ Lexical engine components imported successfully")
except ImportError as e:
    logger.warning(f"⚠️ Lexical engine not available: {e}")
    LEXICAL_ENGINE_AVAILABLE = False

# 4. Semantic Search Engine
try:
    from .semantic_engine import SemanticSearchEngine, SemanticSearchConfig, SemanticSearchResult
    SEMANTIC_ENGINE_AVAILABLE = True
    logger.debug("✅ Semantic engine components imported successfully")
except ImportError as e:
    logger.warning(f"⚠️ Semantic engine not available: {e}")
    SEMANTIC_ENGINE_AVAILABLE = False

# 5. Hybrid Search Engine
try:
    from .search_engine import HybridSearchEngine, HybridSearchConfig, HybridSearchResult, FusionStrategy
    HYBRID_ENGINE_AVAILABLE = True
    logger.debug("✅ Hybrid engine components imported successfully")
except ImportError as e:
    logger.warning(f"⚠️ Hybrid engine not available: {e}")
    HYBRID_ENGINE_AVAILABLE = False

# 6. Result Merger (optionnel)
try:
    from .result_merger import ResultMerger, MergerConfig
    RESULT_MERGER_AVAILABLE = True
    logger.debug("✅ Result merger components imported successfully")
except ImportError as e:
    logger.debug(f"ℹ️ Result merger not available: {e}")
    RESULT_MERGER_AVAILABLE = False

# 7. Reranker (optionnel)
try:
    from .reranker import Reranker, RerankerConfig
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
        "HybridSearchResult",
        "FusionStrategy"
    ])

if RESULT_MERGER_AVAILABLE:
    __all__.extend([
        "ResultMerger",
        "MergerConfig"
    ])

if RERANKER_AVAILABLE:
    __all__.extend([
        "Reranker",
        "RerankerConfig"
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

# ==================== FONCTIONS UTILITAIRES ====================

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
    """Retourne un résumé de l'état des composants."""
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
        )
    }

# ==================== VALIDATION ET DIAGNOSTICS ====================

def validate_core_setup() -> dict:
    """Valide la configuration des composants core."""
    validation_result = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "recommendations": []
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
    
    # Recommandations
    if AVAILABLE_COMPONENTS_COUNT < TOTAL_COMPONENTS_COUNT:
        missing_count = TOTAL_COMPONENTS_COUNT - AVAILABLE_COMPONENTS_COUNT
        validation_result["recommendations"].append(
            f"Consider fixing {missing_count} missing components for full functionality"
        )
    
    if not RESULT_MERGER_AVAILABLE and LEXICAL_ENGINE_AVAILABLE and SEMANTIC_ENGINE_AVAILABLE:
        validation_result["recommendations"].append("Result merger would improve search quality when using multiple engines")
    
    return validation_result

# ==================== LOGGING DE L'ÉTAT D'INITIALISATION ====================

# Log de l'état d'initialisation
summary = get_components_summary()
if summary["service_operational"]:
    logger.info(f"✅ Search service core initialized: {summary['available_count']}/{summary['total_components']} components available ({summary['availability_rate']}%)")
    if summary["unavailable_components"]:
        logger.info(f"ℹ️ Missing optional components: {', '.join(summary['unavailable_components'])}")
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
    "validate_core_setup"
])