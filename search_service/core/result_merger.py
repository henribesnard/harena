"""
Fusion et classement des résultats de recherche hybride - VERSION CENTRALISÉE.

Ce module implémente la fusion intelligente des résultats
lexicaux et sémantiques de manière optimale.

CENTRALISÉ VIA CONFIG_SERVICE:
- Toutes les configurations viennent de config_service.config.settings
- Stratégies de fusion, poids, seuils configurables
- Compatible avec les autres moteurs centralisés
"""
import logging
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

# ✅ CONFIGURATION CENTRALISÉE - SEULE SOURCE DE VÉRITÉ
from config_service.config import settings

from search_service.core.lexical_engine import LexicalSearchResult
from search_service.core.semantic_engine import SemanticSearchResult
from search_service.core.query_processor import QueryAnalysis
from search_service.models.search_types import SearchQuality, SortOrder
from search_service.models.responses import SearchResultItem

logger = logging.getLogger(__name__)


class FusionStrategy(str, Enum):
    """Stratégies de fusion disponibles."""
    WEIGHTED_AVERAGE = "weighted_average"
    RANK_FUSION = "rank_fusion"
    RECIPROCAL_RANK_FUSION = "reciprocal_rank_fusion"
    SCORE_NORMALIZATION = "score_normalization"
    ADAPTIVE_FUSION = "adaptive_fusion"
    BORDA_COUNT = "borda_count"
    COMBSUM = "combsum"
    COMBMNZ = "combmnz"


@dataclass
class FusionConfig:
    """Configuration pour la fusion - Basé sur config_service."""
    # Stratégie par défaut depuis config centralisée
    default_strategy: FusionStrategy = FusionStrategy(settings.DEFAULT_FUSION_STRATEGY)
    
    # Poids par défaut depuis config centralisée
    default_lexical_weight: float = settings.DEFAULT_LEXICAL_WEIGHT
    default_semantic_weight: float = settings.DEFAULT_SEMANTIC_WEIGHT
    
    # Paramètres RRF depuis config centralisée
    rrf_k: float = settings.RRF_K
    
    # Normalisation des scores depuis config centralisée
    score_normalization_method: str = settings.SCORE_NORMALIZATION_METHOD
    
    # Seuils adaptatifs depuis config centralisée
    adaptive_threshold: float = settings.ADAPTIVE_THRESHOLD
    quality_boost_factor: float = settings.QUALITY_BOOST_FACTOR
    
    # Déduplication et diversification depuis config centralisée
    enable_deduplication: bool = settings.ENABLE_DEDUPLICATION
    dedup_similarity_threshold: float = settings.DEDUP_SIMILARITY_THRESHOLD
    enable_diversification: bool = settings.ENABLE_DIVERSIFICATION
    max_same_merchant: int = settings.MAX_SAME_MERCHANT
    
    # Performance depuis config centralisée
    min_results_for_quality: int = settings.MIN_RESULTS_FOR_GOOD_QUALITY


@dataclass
class FusionResult:
    """Résultat de la fusion."""
    results: List[SearchResultItem]
    fusion_strategy: str
    weights_used: Dict[str, float]
    processing_time_ms: float
    lexical_count: int
    semantic_count: int
    unique_count: int
    duplicates_removed: int
    quality: SearchQuality
    debug_info: Optional[Dict[str, Any]] = None


class ResultMerger:
    """
    Fusionneur de résultats de recherche hybride.
    
    Responsabilités:
    - Fusion intelligente des résultats lexicaux et sémantiques
    - Déduplication et diversification des résultats
    - Évaluation de la qualité de fusion
    
    CONFIGURATION CENTRALISÉE VIA CONFIG_SERVICE.
    """
    
    def __init__(self, config: Optional[FusionConfig] = None):
        # Configuration centralisée par défaut
        self.config = config or FusionConfig()
        
        # Métriques
        self.fusion_count = 0
        self.strategy_usage = {strategy.value: 0 for strategy in FusionStrategy}
        self.quality_distribution = {quality.value: 0 for quality in SearchQuality}
        self.avg_processing_time = 0.0
        
        logger.info(f"Result merger initialized with centralized config: {self.config.default_strategy.value}")
    
    def fuse_results(
        self,
        lexical_result: Optional[LexicalSearchResult],
        semantic_result: Optional[SemanticSearchResult],
        query_analysis: QueryAnalysis,
        strategy: Optional[FusionStrategy] = None,
        weights: Optional[Dict[str, float]] = None,
        limit: int = 20,
        offset: int = 0,
        sort_order: SortOrder = SortOrder.RELEVANCE,
        debug: bool = False
    ) -> FusionResult:
        """
        Fusionne les résultats lexicaux et sémantiques.
        
        Args:
            lexical_result: Résultats de recherche lexicale
            semantic_result: Résultats de recherche sémantique
            query_analysis: Analyse de la requête
            strategy: Stratégie de fusion (auto si None)
            weights: Poids pour la fusion (auto si None)
            limit: Limite de résultats finaux
            offset: Décalage pour pagination
            sort_order: Ordre de tri final
            debug: Inclure informations de debug
            
        Returns:
            Résultats fusionnés et optimisés
        """
        start_time = time.time()
        self.fusion_count += 1
        
        # 1. Gestion des cas où un seul moteur a des résultats
        if not lexical_result and not semantic_result:
            return self._create_empty_result(strategy, weights, debug)
        
        if not lexical_result:
            return self._create_single_engine_result(
                semantic_result.results, "semantic_only", weights, 
                offset, limit, sort_order, debug
            )
        
        if not semantic_result:
            return self._create_single_engine_result(
                lexical_result.results, "lexical_only", weights,
                offset, limit, sort_order, debug
            )
        
        # 2. Déterminer la stratégie optimale si non spécifiée (config centralisée)
        if strategy is None:
            strategy = self._determine_optimal_strategy(
                lexical_result, semantic_result, query_analysis
            )
        
        # 3. Déterminer les poids optimaux si non spécifiés (config centralisée)
        if weights is None:
            weights = self._determine_optimal_weights(
                lexical_result, semantic_result, query_analysis
            )
        
        # 4. Effectuer la fusion selon la stratégie choisie
        fused_results = self._execute_fusion_strategy(
            strategy, lexical_result.results, semantic_result.results, weights, debug
        )
        
        # 5. Post-traitement des résultats (config centralisée)
        processed_results = self._post_process_results(
            fused_results, query_analysis, debug
        )
        
        # 6. Appliquer pagination et tri
        final_results = self._apply_pagination_and_sorting(
            processed_results, offset, limit, sort_order
        )
        
        # 7. Calculer les métriques et qualité
        processing_time = (time.time() - start_time) * 1000
        self.avg_processing_time = (
            (self.avg_processing_time * (self.fusion_count - 1) + processing_time) 
            / self.fusion_count
        )
        
        quality = self._assess_fusion_quality(
            final_results, lexical_result, semantic_result, query_analysis
        )
        
        # 8. Mettre à jour les statistiques
        self.strategy_usage[strategy.value] += 1
        self.quality_distribution[quality.value] += 1
        
        # 9. Construire le résultat final
        return FusionResult(
            results=final_results,
            fusion_strategy=strategy.value,
            weights_used=weights,
            processing_time_ms=processing_time,
            lexical_count=len(lexical_result.results),
            semantic_count=len(semantic_result.results),
            unique_count=len(processed_results),
            duplicates_removed=len(fused_results) - len(processed_results),
            quality=quality,
            debug_info=self._create_debug_info(
                strategy, lexical_result, semantic_result, fused_results
            ) if debug else None
        )
    
    def _determine_optimal_strategy(
        self,
        lexical_result: LexicalSearchResult,
        semantic_result: SemanticSearchResult,
        query_analysis: QueryAnalysis
    ) -> FusionStrategy:
        """Détermine la stratégie optimale basée sur l'analyse et la config centralisée."""
        
        # Utiliser la stratégie par défaut de la config si pas d'adaptation
        default_strategy = self.config.default_strategy
        
        # Logique d'adaptation basée sur l'analyse de requête
        if getattr(query_analysis, 'has_exact_phrases', False):
            # Phrases exactes -> favoriser lexical
            return FusionStrategy.WEIGHTED_AVERAGE
        
        if getattr(query_analysis, 'is_question', False):
            # Questions -> favoriser sémantique
            return FusionStrategy.ADAPTIVE_FUSION
        
        # Analyse de la qualité des résultats
        lexical_quality = lexical_result.quality
        semantic_quality = semantic_result.quality
        
        if abs(lexical_quality.value - semantic_quality.value) > 1:
            # Grande différence de qualité -> fusion adaptative
            return FusionStrategy.ADAPTIVE_FUSION
        
        # Utiliser la stratégie par défaut configurée
        return default_strategy
    
    def _determine_optimal_weights(
        self,
        lexical_result: LexicalSearchResult,
        semantic_result: SemanticSearchResult,
        query_analysis: QueryAnalysis
    ) -> Dict[str, float]:
        """Détermine les poids optimaux basés sur la config centralisée."""
        
        # Poids par défaut de la config centralisée
        base_lexical = self.config.default_lexical_weight
        base_semantic = self.config.default_semantic_weight
        
        # Ajustements basés sur l'analyse de requête
        lexical_boost = 0.0
        semantic_boost = 0.0
        
        # Phrases exactes -> favoriser lexical
        if getattr(query_analysis, 'has_exact_phrases', False):
            lexical_boost += 0.2
        
        # Questions -> favoriser sémantique
        if getattr(query_analysis, 'is_question', False):
            semantic_boost += 0.2
        
        # Ajustements basés sur la qualité des résultats
        lexical_quality_score = lexical_result.quality.value
        semantic_quality_score = semantic_result.quality.value
        
        quality_diff = semantic_quality_score - lexical_quality_score
        if abs(quality_diff) > 1:
            if quality_diff > 0:
                semantic_boost += 0.15  # Sémantique meilleur
            else:
                lexical_boost += 0.15   # Lexical meilleur
        
        # Appliquer les ajustements
        final_lexical = max(0.1, min(0.9, base_lexical + lexical_boost - semantic_boost))
        final_semantic = 1.0 - final_lexical
        
        return {
            "lexical_weight": final_lexical,
            "semantic_weight": final_semantic
        }
    
    def _execute_fusion_strategy(
        self,
        strategy: FusionStrategy,
        lexical_results: List[SearchResultItem],
        semantic_results: List[SearchResultItem],
        weights: Dict[str, float],
        debug: bool
    ) -> List[SearchResultItem]:
        """Exécute la stratégie de fusion spécifiée."""
        
        if strategy == FusionStrategy.WEIGHTED_AVERAGE:
            return self._weighted_average_fusion(lexical_results, semantic_results, weights)
        
        elif strategy == FusionStrategy.RANK_FUSION:
            return self._rank_fusion(lexical_results, semantic_results, weights)
        
        elif strategy == FusionStrategy.RECIPROCAL_RANK_FUSION:
            return self._reciprocal_rank_fusion(lexical_results, semantic_results)
        
        elif strategy == FusionStrategy.SCORE_NORMALIZATION:
            return self._score_normalization_fusion(lexical_results, semantic_results, weights)
        
        elif strategy == FusionStrategy.ADAPTIVE_FUSION:
            return self._adaptive_fusion(lexical_results, semantic_results, weights)