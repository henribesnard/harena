"""
Fusion et classement des résultats de recherche hybride.

Ce module implémente la fusion intelligente des résultats
lexicaux et sémantiques de manière optimale.
"""
import logging
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from search_service.core.lexical_engine import LexicalSearchResult
from search_service.core.semantic_engine import SemanticSearchResult
from search_service.core.query_processor import QueryAnalysis
from search_service.models.search_types import SearchQuality, SortOrder
from search_service.models.responses import SearchResultItem

# Import des utilitaires
from search_service.utils.fusion_strategies import (
    FusionStrategy, FusionConfig, FusionStrategyExecutor
)
from search_service.config.fusion_config import FusionConfigPresets
from search_service.utils.fusion_utils import (
    deduplicate_results, diversify_results, apply_quality_boost
)
from search_service.utils.quality_assessment import QualityAssessor
from search_service.utils.weight_optimizer import WeightOptimizer

logger = logging.getLogger(__name__)


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
    """
    
    def __init__(self, config: Optional[FusionConfig] = None, config_preset: Optional[str] = None):
        # Priorité : config explicite > preset > défaut
        if config:
            self.config = config
        elif config_preset:
            self.config = FusionConfigPresets.get_config_by_name(config_preset)
        else:
            self.config = FusionConfigPresets.default_config()
            
        self.weight_optimizer = WeightOptimizer(
            self.config.default_lexical_weight,
            self.config.default_semantic_weight
        )
        self.strategy_executor = FusionStrategyExecutor(self.config)
        self.quality_assessor = QualityAssessor()
        
        # Métriques
        self.fusion_count = 0
        self.strategy_usage = {strategy.value: 0 for strategy in FusionStrategy}
        self.quality_distribution = {quality.value: 0 for quality in SearchQuality}
        self.avg_processing_time = 0.0
        
        logger.info(f"Result merger initialized with config: {self.config.default_strategy.value}")
    
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
        
        # 2. Déterminer la stratégie optimale si non spécifiée
        if strategy is None:
            strategy = self.weight_optimizer.determine_optimal_strategy(
                lexical_result, semantic_result, query_analysis
            )
        
        # 3. Déterminer les poids optimaux si non spécifiés
        if weights is None:
            weights = self.weight_optimizer.determine_optimal_weights(
                lexical_result, semantic_result, query_analysis
            )
        
        # 4. Effectuer la fusion selon la stratégie choisie
        fused_results = self.strategy_executor.execute(
            strategy, lexical_result.results, semantic_result.results, weights, debug
        )
        
        # 5. Post-traitement des résultats
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
        
        quality = self.quality_assessor.assess_fusion_quality(
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
    
    def _post_process_results(
        self,
        fused_results: List[SearchResultItem],
        query_analysis: QueryAnalysis,
        debug: bool
    ) -> List[SearchResultItem]:
        """Post-traite les résultats fusionnés."""
        
        # 1. Déduplication
        if self.config.enable_deduplication:
            fused_results = deduplicate_results(
                fused_results, self.config.dedup_similarity_threshold, debug
            )
        
        # 2. Diversification
        if self.config.enable_diversification:
            fused_results = diversify_results(
                fused_results, self.config.max_same_merchant, debug
            )
        
        # 3. Boost de qualité pour certains résultats
        fused_results = apply_quality_boost(
            fused_results, query_analysis.key_terms, 
            self.config.quality_boost_factor, debug
        )
        
        return fused_results
    
    def _apply_pagination_and_sorting(
        self,
        results: List[SearchResultItem],
        offset: int,
        limit: int,
        sort_order: SortOrder
    ) -> List[SearchResultItem]:
        """Applique la pagination et le tri aux résultats."""
        
        # Trier selon l'ordre demandé
        if sort_order == SortOrder.DATE_DESC:
            results.sort(key=lambda x: (x.transaction_date or "", -(x.score or 0)), reverse=True)
        elif sort_order == SortOrder.DATE_ASC:
            results.sort(key=lambda x: (x.transaction_date or "", -(x.score or 0)))
        elif sort_order == SortOrder.AMOUNT_DESC:
            results.sort(key=lambda x: (x.amount or 0, -(x.score or 0)), reverse=True)
        elif sort_order == SortOrder.AMOUNT_ASC:
            results.sort(key=lambda x: (x.amount or 0, -(x.score or 0)))
        else:  # RELEVANCE (défaut)
            results.sort(key=lambda x: x.score or 0, reverse=True)
        
        # Appliquer la pagination
        return results[offset:offset + limit]
    
    def _create_empty_result(
        self,
        strategy: Optional[FusionStrategy],
        weights: Optional[Dict[str, float]],
        debug: bool
    ) -> FusionResult:
        """Crée un résultat vide."""
        return FusionResult(
            results=[],
            fusion_strategy=(strategy.value if strategy else "none"),
            weights_used=(weights or {}),
            processing_time_ms=0.0,
            lexical_count=0,
            semantic_count=0,
            unique_count=0,
            duplicates_removed=0,
            quality=SearchQuality.POOR,
            debug_info={"reason": "no_results"} if debug else None
        )
    
    def _create_single_engine_result(
        self,
        results: List[SearchResultItem],
        strategy: str,
        weights: Optional[Dict[str, float]],
        offset: int,
        limit: int,
        sort_order: SortOrder,
        debug: bool
    ) -> FusionResult:
        """Crée un résultat pour un seul moteur."""
        start_time = time.time()
        
        # Appliquer pagination et tri
        final_results = self._apply_pagination_and_sorting(results, offset, limit, sort_order)
        
        processing_time = (time.time() - start_time) * 1000
        
        # Évaluer la qualité basée sur les scores
        quality = SearchQuality.MEDIUM
        if results:
            valid_scores = [r.score for r in results if r.score is not None]
            if valid_scores:
                avg_score = sum(valid_scores) / len(valid_scores)
                if avg_score >= 0.8:
                    quality = SearchQuality.EXCELLENT
                elif avg_score >= 0.6:
                    quality = SearchQuality.GOOD
                elif avg_score >= 0.3:
                    quality = SearchQuality.MEDIUM
                else:
                    quality = SearchQuality.POOR
        
        return FusionResult(
            results=final_results,
            fusion_strategy=strategy,
            weights_used=(weights or {}),
            processing_time_ms=processing_time,
            lexical_count=len(results) if "lexical" in strategy else 0,
            semantic_count=len(results) if "semantic" in strategy else 0,
            unique_count=len(results),
            duplicates_removed=0,
            quality=quality,
            debug_info={"single_engine": True} if debug else None
        )
    
    def _create_debug_info(
        self,
        strategy: FusionStrategy,
        lexical_result: LexicalSearchResult,
        semantic_result: SemanticSearchResult,
        fused_results: List[SearchResultItem]
    ) -> Dict[str, Any]:
        """Crée les informations de debug."""
        return {
            "strategy_used": strategy.value,
            "input_counts": {
                "lexical": len(lexical_result.results),
                "semantic": len(semantic_result.results)
            },
            "fusion_stats": {
                "total_fused": len(fused_results),
                "score_range": {
                    "min": min((r.score for r in fused_results if r.score), default=0),
                    "max": max((r.score for r in fused_results if r.score), default=0)
                }
            },
            "quality_inputs": {
                "lexical_quality": lexical_result.quality.value,
                "semantic_quality": semantic_result.quality.value
            }
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Retourne les métriques du fusionneur."""
        return {
            "fusion_count": self.fusion_count,
            "strategy_usage": self.strategy_usage,
            "quality_distribution": self.quality_distribution,
            "average_processing_time_ms": self.avg_processing_time,
            "config": {
                "default_strategy": self.config.default_strategy.value,
                "deduplication_enabled": self.config.enable_deduplication,
                "diversification_enabled": self.config.enable_diversification
            }
        }
    
    def reset_metrics(self) -> None:
        """Remet à zéro les métriques."""
        self.fusion_count = 0
        self.strategy_usage = {strategy.value: 0 for strategy in FusionStrategy}
        self.quality_distribution = {quality.value: 0 for quality in SearchQuality}
        self.avg_processing_time = 0.0
        
        logger.info("Result merger metrics reset")
    
    def update_config(self, new_config: FusionConfig) -> None:
        """Met à jour la configuration du fusionneur."""
        self.config = new_config
        self.weight_optimizer = WeightOptimizer(
            new_config.default_lexical_weight,
            new_config.default_semantic_weight
        )
        self.strategy_executor = FusionStrategyExecutor(new_config)
        
        logger.info("Result merger configuration updated")
    
    def optimize_weights_from_feedback(
        self,
        weights: Dict[str, float],
        feedback_score: float,
        query_features: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Optimise les poids basés sur le feedback utilisateur.
        
        Args:
            weights: Poids utilisés
            feedback_score: Score de feedback (0-1)
            query_features: Caractéristiques de la requête
            
        Returns:
            Poids optimisés
        """
        # Cette méthode peut être étendue avec un gestionnaire adaptatif
        # Pour l'instant, retourne les poids inchangés
        return weights
    
    @classmethod
    def create_with_preset(cls, preset_name: str) -> 'ResultMerger':
        """
        Crée un ResultMerger avec un preset de configuration.
        
        Args:
            preset_name: Nom du preset (default, precision, recall, etc.)
            
        Returns:
            Instance configurée de ResultMerger
        """
        return cls(config_preset=preset_name)
    
    @classmethod 
    def create_optimized_for_query_type(cls, query_analysis: QueryAnalysis) -> 'ResultMerger':
        """
        Crée un ResultMerger optimisé pour un type de requête.
        
        Args:
            query_analysis: Analyse de la requête
            
        Returns:
            Instance optimisée de ResultMerger
        """
        # Logique de sélection automatique de preset
        if query_analysis.has_exact_phrases:
            return cls.create_with_preset("precision")
        elif len(query_analysis.key_terms) > 5:
            return cls.create_with_preset("semantic")
        elif query_analysis.has_financial_entities:
            return cls.create_with_preset("lexical")
        else:
            return cls.create_with_preset("default")
    
    def switch_to_preset(self, preset_name: str) -> None:
        """
        Change la configuration vers un preset.
        
        Args:
            preset_name: Nom du nouveau preset
        """
        new_config = FusionConfigPresets.get_config_by_name(preset_name)
        self.update_config(new_config)
        logger.info(f"Switched to preset: {preset_name}")
    
    def get_fusion_explanation(
        self,
        strategy: FusionStrategy,
        weights: Dict[str, float],
        query_analysis: QueryAnalysis
    ) -> str:
        """
        Génère une explication de la stratégie de fusion utilisée.
        
        Args:
            strategy: Stratégie utilisée
            weights: Poids appliqués
            query_analysis: Analyse de la requête
            
        Returns:
            Explication textuelle de la fusion
        """
        explanations = {
            FusionStrategy.WEIGHTED_AVERAGE: f"Moyenne pondérée (lexical: {weights.get('lexical_weight', 0):.1%}, sémantique: {weights.get('semantic_weight', 0):.1%})",
            FusionStrategy.RANK_FUSION: "Fusion basée sur les rangs dans chaque liste",
            FusionStrategy.RECIPROCAL_RANK_FUSION: f"Fusion RRF avec k={self.config.rrf_k}",
            FusionStrategy.SCORE_NORMALIZATION: f"Normalisation {self.config.score_normalization_method} puis fusion",
            FusionStrategy.ADAPTIVE_FUSION: "Fusion adaptative selon le contexte de chaque résultat",
            FusionStrategy.BORDA_COUNT: "Fusion Borda Count avec points de rang",
            FusionStrategy.COMBSUM: "Somme des scores normalisés",
            FusionStrategy.COMBMNZ: "Somme pondérée par le nombre de moteurs"
        }
        
        base_explanation = explanations.get(strategy, "Stratégie de fusion inconnue")
        
        # Ajouter des détails selon l'analyse de requête
        details = []
        if query_analysis.has_exact_phrases:
            details.append("optimisée pour phrases exactes")
        if query_analysis.has_financial_entities:
            details.append("adaptée aux entités financières")
        if len(query_analysis.key_terms) > 5:
            details.append("pour requête complexe")
        
        if details:
            base_explanation += f" ({', '.join(details)})"
        
        return base_explanation
    
    def get_available_presets(self) -> List[str]:
        """Retourne les presets disponibles."""
        return FusionConfigPresets.get_available_configs()