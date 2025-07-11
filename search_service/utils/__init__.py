"""
Utilitaires pour le service de recherche - VERSION CENTRALISÉE.

Ce module expose les utilitaires communs utilisés
par le service de recherche hybride avec configuration centralisée.

AMÉLIORATION:
- Configuration entièrement centralisée via config_service
- Plus de duplication de paramètres
- Contrôle total via .env
"""

# Stratégies de fusion avec configuration centralisée
from .fusion_strategies import (
    FusionStrategy,
    FusionConfig,
    ScoreNormalizer,
    FusionStrategyExecutor,
    create_simple_executor,
    create_default_executor
)

# Utilitaires de fusion (inchangés - pas de config hardcodée)
from .fusion_utils import (
    create_fused_item,
    create_transaction_signature,
    calculate_signature_similarity,
    deduplicate_results,
    diversify_results,
    apply_quality_boost
)

# Évaluation de qualité (inchangés - pas de config hardcodée)
from .quality_assessment import (
    QualityAssessor,
    quality_to_score,
    calculate_quality_metrics
)

# Optimisation des poids avec configuration centralisée
from .weight_optimizer import (
    WeightOptimizer,
    AdaptiveWeightManager,
    create_weight_optimizer,
    create_adaptive_weight_manager,
    get_default_weights,
    validate_weights,
    get_weight_optimization_config
)

# Cache avec configuration centralisée
from .cache import (
    SearchCache,
    MultiLevelCache,
    global_cache,
    get_search_cache,
    get_embedding_cache,
    get_query_analysis_cache,
    get_suggestions_cache,
    generate_cache_key,
    cache_with_ttl,
    get_cache_metrics,
    is_cache_enabled,
    get_cache_config_summary
)

__all__ = [
    # Stratégies de fusion
    "FusionStrategy",
    "FusionConfig", 
    "ScoreNormalizer",
    "FusionStrategyExecutor",
    "create_simple_executor",
    "create_default_executor",
    
    # Utilitaires de fusion
    "create_fused_item",
    "create_transaction_signature",
    "calculate_signature_similarity",
    "deduplicate_results",
    "diversify_results",
    "apply_quality_boost",
    
    # Évaluation de qualité
    "QualityAssessor",
    "quality_to_score",
    "calculate_quality_metrics",
    
    # Optimisation des poids
    "WeightOptimizer",
    "AdaptiveWeightManager",
    "create_weight_optimizer",
    "create_adaptive_weight_manager",
    "get_default_weights",
    "validate_weights",
    "get_weight_optimization_config",
    
    # Cache
    "SearchCache",
    "MultiLevelCache",
    "global_cache",
    "get_search_cache",
    "get_embedding_cache",
    "get_query_analysis_cache",
    "get_suggestions_cache",
    "generate_cache_key",
    "cache_with_ttl",
    "get_cache_metrics",
    "is_cache_enabled",
    "get_cache_config_summary"
]

# ==========================================
# 🎯 FONCTIONS D'ACCÈS RAPIDE CENTRALISÉES
# ==========================================

def get_utils_config_summary():
    """Retourne un résumé de la configuration des utilitaires."""
    from config_service.config import settings
    
    return {
        "fusion": {
            "default_lexical_weight": settings.DEFAULT_LEXICAL_WEIGHT,
            "default_semantic_weight": settings.DEFAULT_SEMANTIC_WEIGHT,
            "score_normalization_method": settings.SCORE_NORMALIZATION_METHOD,
            "rrf_k": settings.RRF_K,
            "adaptive_threshold": settings.ADAPTIVE_THRESHOLD,
            "quality_boost_factor": settings.QUALITY_BOOST_FACTOR,
            "enable_deduplication": settings.ENABLE_DEDUPLICATION,
            "enable_diversification": settings.ENABLE_DIVERSIFICATION
        },
        "cache": get_cache_config_summary(),
        "quality": {
            "excellent_threshold": settings.QUALITY_EXCELLENT_THRESHOLD,
            "good_threshold": settings.QUALITY_GOOD_THRESHOLD,
            "medium_threshold": settings.QUALITY_MEDIUM_THRESHOLD,
            "poor_threshold": settings.QUALITY_POOR_THRESHOLD
        },
        "config_source": "centralized (config_service)"
    }