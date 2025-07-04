"""
Configuration avancée pour la fusion de résultats.

Ce module fournit des configurations prédéfinies optimisées
pour différents scénarios d'usage.
"""
from typing import List
from search_service.utils.fusion_strategies import FusionStrategy, FusionConfig


class FusionConfigPresets:
    """Presets de configuration pour différents cas d'usage."""
    
    @staticmethod
    def default_config() -> FusionConfig:
        """Configuration par défaut équilibrée."""
        return FusionConfig(
            default_strategy=FusionStrategy.ADAPTIVE_FUSION,
            default_lexical_weight=0.6,
            default_semantic_weight=0.4,
            score_normalization_method="min_max",
            min_score_threshold=0.01,
            rrf_k=60,
            adaptive_threshold=0.1,
            quality_boost_factor=0.2,
            enable_deduplication=True,
            dedup_similarity_threshold=0.95,
            enable_diversification=True,
            diversity_factor=0.1,
            max_same_merchant=3
        )
    
    @staticmethod
    def precision_focused_config() -> FusionConfig:
        """Configuration optimisée pour la précision."""
        return FusionConfig(
            default_strategy=FusionStrategy.RECIPROCAL_RANK_FUSION,
            default_lexical_weight=0.7,
            default_semantic_weight=0.3,
            score_normalization_method="z_score",
            min_score_threshold=0.05,
            rrf_k=30,
            adaptive_threshold=0.05,
            quality_boost_factor=0.3,
            enable_deduplication=True,
            dedup_similarity_threshold=0.98,
            enable_diversification=True,
            diversity_factor=0.05,
            max_same_merchant=2
        )
    
    @staticmethod
    def recall_focused_config() -> FusionConfig:
        """Configuration optimisée pour le rappel."""
        return FusionConfig(
            default_strategy=FusionStrategy.COMBSUM,
            default_lexical_weight=0.5,
            default_semantic_weight=0.5,
            score_normalization_method="sigmoid",
            min_score_threshold=0.001,
            rrf_k=100,
            adaptive_threshold=0.2,
            quality_boost_factor=0.1,
            enable_deduplication=True,
            dedup_similarity_threshold=0.9,
            enable_diversification=False,
            diversity_factor=0.2,
            max_same_merchant=5
        )
    
    @staticmethod
    def semantic_heavy_config() -> FusionConfig:
        """Configuration favorisant la recherche sémantique."""
        return FusionConfig(
            default_strategy=FusionStrategy.WEIGHTED_AVERAGE,
            default_lexical_weight=0.3,
            default_semantic_weight=0.7,
            score_normalization_method="min_max",
            min_score_threshold=0.01,
            rrf_k=60,
            adaptive_threshold=0.15,
            quality_boost_factor=0.25,
            enable_deduplication=True,
            dedup_similarity_threshold=0.93,
            enable_diversification=True,
            diversity_factor=0.15,
            max_same_merchant=4
        )
    
    @staticmethod
    def lexical_heavy_config() -> FusionConfig:
        """Configuration favorisant la recherche lexicale."""
        return FusionConfig(
            default_strategy=FusionStrategy.RANK_FUSION,
            default_lexical_weight=0.8,
            default_semantic_weight=0.2,
            score_normalization_method="min_max",
            min_score_threshold=0.02,
            rrf_k=40,
            adaptive_threshold=0.08,
            quality_boost_factor=0.15,
            enable_deduplication=True,
            dedup_similarity_threshold=0.97,
            enable_diversification=True,
            diversity_factor=0.08,
            max_same_merchant=2
        )
    
    @staticmethod
    def speed_optimized_config() -> FusionConfig:
        """Configuration optimisée pour la vitesse."""
        return FusionConfig(
            default_strategy=FusionStrategy.WEIGHTED_AVERAGE,
            default_lexical_weight=0.6,
            default_semantic_weight=0.4,
            score_normalization_method="min_max",
            min_score_threshold=0.01,
            rrf_k=60,
            adaptive_threshold=0.1,
            quality_boost_factor=0.1,
            enable_deduplication=False,  # Désactivé pour la vitesse
            dedup_similarity_threshold=0.95,
            enable_diversification=False,  # Désactivé pour la vitesse
            diversity_factor=0.1,
            max_same_merchant=10
        )
    
    @staticmethod
    def get_config_by_name(config_name: str) -> FusionConfig:
        """
        Retourne une configuration par son nom.
        
        Args:
            config_name: Nom de la configuration
            
        Returns:
            Configuration correspondante
            
        Raises:
            ValueError: Si le nom n'est pas reconnu
        """
        configs = {
            "default": FusionConfigPresets.default_config,
            "precision": FusionConfigPresets.precision_focused_config,
            "recall": FusionConfigPresets.recall_focused_config,
            "semantic": FusionConfigPresets.semantic_heavy_config,
            "lexical": FusionConfigPresets.lexical_heavy_config,
            "speed": FusionConfigPresets.speed_optimized_config
        }
        
        if config_name not in configs:
            available = ", ".join(configs.keys())
            raise ValueError(
                f"Configuration '{config_name}' non reconnue. "
                f"Configurations disponibles: {available}"
            )
        
        return configs[config_name]()
    
    @staticmethod
    def get_available_configs() -> List[str]:
        """Retourne la liste des configurations disponibles."""
        return [
            "default",
            "precision", 
            "recall",
            "semantic",
            "lexical",
            "speed"
        ]


def create_custom_config(
    strategy: str = "adaptive_fusion",
    lexical_weight: float = 0.6,
    semantic_weight: float = 0.4,
    enable_dedup: bool = True,
    enable_diversification: bool = True,
    **kwargs
) -> FusionConfig:
    """
    Crée une configuration personnalisée.
    
    Args:
        strategy: Nom de la stratégie (par défaut: adaptive_fusion)
        lexical_weight: Poids lexical (par défaut: 0.6)
        semantic_weight: Poids sémantique (par défaut: 0.4)
        enable_dedup: Activer la déduplication (par défaut: True)
        enable_diversification: Activer la diversification (par défaut: True)
        **kwargs: Autres paramètres de configuration
        
    Returns:
        Configuration personnalisée
    """
    # Mapper les noms de stratégies aux enums
    strategy_map = {
        "weighted_average": FusionStrategy.WEIGHTED_AVERAGE,
        "rank_fusion": FusionStrategy.RANK_FUSION,
        "reciprocal_rank_fusion": FusionStrategy.RECIPROCAL_RANK_FUSION,
        "score_normalization": FusionStrategy.SCORE_NORMALIZATION,
        "adaptive_fusion": FusionStrategy.ADAPTIVE_FUSION,
        "borda_count": FusionStrategy.BORDA_COUNT,
        "combsum": FusionStrategy.COMBSUM,
        "combmnz": FusionStrategy.COMBMNZ
    }
    
    strategy_enum = strategy_map.get(strategy, FusionStrategy.ADAPTIVE_FUSION)
    
    # Normaliser les poids
    total_weight = lexical_weight + semantic_weight
    if total_weight > 0:
        lexical_weight /= total_weight
        semantic_weight /= total_weight
    else:
        lexical_weight = 0.6
        semantic_weight = 0.4
    
    # Configuration de base
    config_params = {
        "default_strategy": strategy_enum,
        "default_lexical_weight": lexical_weight,
        "default_semantic_weight": semantic_weight,
        "enable_deduplication": enable_dedup,
        "enable_diversification": enable_diversification,
        **kwargs
    }
    
    return FusionConfig(**config_params)