"""
Utilitaires pour le service de recherche.

Ce module expose les utilitaires communs utilisés
par le service de recherche hybride.
"""

# Stratégies de fusion
from .fusion_strategies import (
    FusionStrategy,
    FusionConfig,
    ScoreNormalizer,
    FusionStrategyExecutor
)

# Utilitaires de fusion
from .fusion_utils import (
    create_fused_item,
    create_transaction_signature,
    calculate_signature_similarity,
    deduplicate_results,
    diversify_results,
    apply_quality_boost
)

# Évaluation de qualité
from .quality_assessment import (
    QualityAssessor,
    quality_to_score,
    calculate_quality_metrics
)

# Optimisation des poids
from .weight_optimizer import (
    WeightOptimizer,
    AdaptiveWeightManager
)

__all__ = [
    # Stratégies de fusion
    "FusionStrategy",
    "FusionConfig", 
    "ScoreNormalizer",
    "FusionStrategyExecutor",
    
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
    "AdaptiveWeightManager"
]