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
from typing import Dict, Any, List, Optional, Union
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
        
        elif strategy == FusionStrategy.BORDA_COUNT:
            return self._borda_count_fusion(lexical_results, semantic_results, weights)
        
        elif strategy == FusionStrategy.COMBSUM:
            return self._combsum_fusion(lexical_results, semantic_results, weights)
        
        elif strategy == FusionStrategy.COMBMNZ:
            return self._combmnz_fusion(lexical_results, semantic_results, weights)
        
        else:
            # Fallback vers weighted average
            logger.warning(f"Unknown fusion strategy: {strategy}, falling back to weighted average")
            return self._weighted_average_fusion(lexical_results, semantic_results, weights)
    
    def _weighted_average_fusion(
        self,
        lexical_results: List[SearchResultItem],
        semantic_results: List[SearchResultItem],
        weights: Dict[str, float]
    ) -> List[SearchResultItem]:
        """Fusion par moyenne pondérée des scores."""
        lexical_weight = weights.get("lexical_weight", self.config.default_lexical_weight)
        semantic_weight = weights.get("semantic_weight", self.config.default_semantic_weight)
        
        # Créer un index des résultats par ID
        lexical_dict = {item.id: item for item in lexical_results}
        semantic_dict = {item.id: item for item in semantic_results}
        
        # Fusionner les résultats
        merged_results = []
        all_ids = set(lexical_dict.keys()) | set(semantic_dict.keys())
        
        for item_id in all_ids:
            lexical_item = lexical_dict.get(item_id)
            semantic_item = semantic_dict.get(item_id)
            
            if lexical_item and semantic_item:
                # Présent dans les deux -> moyenne pondérée
                final_score = (
                    lexical_item.score * lexical_weight + 
                    semantic_item.score * semantic_weight
                )
                # Utiliser l'item lexical comme base avec le nouveau score
                merged_item = lexical_item
                merged_item.score = final_score
                merged_results.append(merged_item)
            
            elif lexical_item:
                # Uniquement lexical -> appliquer le poids
                merged_item = lexical_item
                merged_item.score = lexical_item.score * lexical_weight
                merged_results.append(merged_item)
            
            elif semantic_item:
                # Uniquement sémantique -> appliquer le poids
                merged_item = semantic_item
                merged_item.score = semantic_item.score * semantic_weight
                merged_results.append(merged_item)
        
        # Trier par score décroissant
        merged_results.sort(key=lambda x: x.score, reverse=True)
        return merged_results
    
    def _reciprocal_rank_fusion(
        self,
        lexical_results: List[SearchResultItem],
        semantic_results: List[SearchResultItem]
    ) -> List[SearchResultItem]:
        """Fusion par Reciprocal Rank Fusion (RRF)."""
        k = self.config.rrf_k
        
        # Créer les dictionnaires de rangs
        lexical_ranks = {item.id: rank + 1 for rank, item in enumerate(lexical_results)}
        semantic_ranks = {item.id: rank + 1 for rank, item in enumerate(semantic_results)}
        
        # Calculer les scores RRF
        rrf_scores = {}
        all_ids = set(lexical_ranks.keys()) | set(semantic_ranks.keys())
        
        for item_id in all_ids:
            score = 0.0
            if item_id in lexical_ranks:
                score += 1.0 / (k + lexical_ranks[item_id])
            if item_id in semantic_ranks:
                score += 1.0 / (k + semantic_ranks[item_id])
            rrf_scores[item_id] = score
        
        # Créer les résultats fusionnés
        merged_results = []
        lexical_dict = {item.id: item for item in lexical_results}
        semantic_dict = {item.id: item for item in semantic_results}
        
        for item_id, rrf_score in rrf_scores.items():
            # Prendre l'item lexical si disponible, sinon sémantique
            base_item = lexical_dict.get(item_id) or semantic_dict.get(item_id)
            if base_item:
                base_item.score = rrf_score
                merged_results.append(base_item)
        
        # Trier par score RRF décroissant
        merged_results.sort(key=lambda x: x.score, reverse=True)
        return merged_results
    
    def _rank_fusion(
        self,
        lexical_results: List[SearchResultItem],
        semantic_results: List[SearchResultItem],
        weights: Dict[str, float]
    ) -> List[SearchResultItem]:
        """Fusion basée sur les rangs pondérés."""
        lexical_weight = weights.get("lexical_weight", self.config.default_lexical_weight)
        semantic_weight = weights.get("semantic_weight", self.config.default_semantic_weight)
        
        # Créer les dictionnaires de rangs (rang inversé pour le score)
        lexical_ranks = {item.id: len(lexical_results) - rank for rank, item in enumerate(lexical_results)}
        semantic_ranks = {item.id: len(semantic_results) - rank for rank, item in enumerate(semantic_results)}
        
        # Calculer les scores de fusion
        fusion_scores = {}
        all_ids = set(lexical_ranks.keys()) | set(semantic_ranks.keys())
        
        for item_id in all_ids:
            score = 0.0
            if item_id in lexical_ranks:
                score += lexical_ranks[item_id] * lexical_weight
            if item_id in semantic_ranks:
                score += semantic_ranks[item_id] * semantic_weight
            fusion_scores[item_id] = score
        
        # Créer les résultats fusionnés
        merged_results = []
        lexical_dict = {item.id: item for item in lexical_results}
        semantic_dict = {item.id: item for item in semantic_results}
        
        for item_id, fusion_score in fusion_scores.items():
            base_item = lexical_dict.get(item_id) or semantic_dict.get(item_id)
            if base_item:
                base_item.score = fusion_score
                merged_results.append(base_item)
        
        # Trier par score de fusion décroissant
        merged_results.sort(key=lambda x: x.score, reverse=True)
        return merged_results
    
    def _score_normalization_fusion(
        self,
        lexical_results: List[SearchResultItem],
        semantic_results: List[SearchResultItem],
        weights: Dict[str, float]
    ) -> List[SearchResultItem]:
        """Fusion avec normalisation des scores."""
        # Normaliser les scores lexicaux
        if lexical_results:
            lexical_scores = [item.score for item in lexical_results]
            lexical_min, lexical_max = min(lexical_scores), max(lexical_scores)
            lexical_range = lexical_max - lexical_min if lexical_max != lexical_min else 1.0
            
            for item in lexical_results:
                item.score = (item.score - lexical_min) / lexical_range
        
        # Normaliser les scores sémantiques
        if semantic_results:
            semantic_scores = [item.score for item in semantic_results]
            semantic_min, semantic_max = min(semantic_scores), max(semantic_scores)
            semantic_range = semantic_max - semantic_min if semantic_max != semantic_min else 1.0
            
            for item in semantic_results:
                item.score = (item.score - semantic_min) / semantic_range
        
        # Appliquer la fusion weighted average sur les scores normalisés
        return self._weighted_average_fusion(lexical_results, semantic_results, weights)
    
    def _adaptive_fusion(
        self,
        lexical_results: List[SearchResultItem],
        semantic_results: List[SearchResultItem],
        weights: Dict[str, float]
    ) -> List[SearchResultItem]:
        """Fusion adaptative basée sur la qualité des résultats."""
        # Évaluer la qualité relative de chaque ensemble de résultats
        lexical_quality = self._evaluate_result_quality(lexical_results)
        semantic_quality = self._evaluate_result_quality(semantic_results)
        
        # Ajuster les poids selon la qualité
        total_quality = lexical_quality + semantic_quality
        if total_quality > 0:
            adaptive_lexical_weight = lexical_quality / total_quality
            adaptive_semantic_weight = semantic_quality / total_quality
        else:
            adaptive_lexical_weight = self.config.default_lexical_weight
            adaptive_semantic_weight = self.config.default_semantic_weight
        
        # Mélanger les poids adaptatifs avec les poids originaux
        final_weights = {
            "lexical_weight": (adaptive_lexical_weight + weights.get("lexical_weight", 0.5)) / 2,
            "semantic_weight": (adaptive_semantic_weight + weights.get("semantic_weight", 0.5)) / 2
        }
        
        return self._weighted_average_fusion(lexical_results, semantic_results, final_weights)
    
    def _borda_count_fusion(
        self,
        lexical_results: List[SearchResultItem],
        semantic_results: List[SearchResultItem],
        weights: Dict[str, float]
    ) -> List[SearchResultItem]:
        """Fusion par méthode Borda Count."""
        lexical_weight = weights.get("lexical_weight", self.config.default_lexical_weight)
        semantic_weight = weights.get("semantic_weight", self.config.default_semantic_weight)
        
        # Calculer les points Borda
        borda_scores = {}
        
        # Points des résultats lexicaux
        for rank, item in enumerate(lexical_results):
            points = (len(lexical_results) - rank - 1) * lexical_weight
            borda_scores[item.id] = borda_scores.get(item.id, 0) + points
        
        # Points des résultats sémantiques
        for rank, item in enumerate(semantic_results):
            points = (len(semantic_results) - rank - 1) * semantic_weight
            borda_scores[item.id] = borda_scores.get(item.id, 0) + points
        
        # Créer les résultats fusionnés
        merged_results = []
        lexical_dict = {item.id: item for item in lexical_results}
        semantic_dict = {item.id: item for item in semantic_results}
        
        for item_id, borda_score in borda_scores.items():
            base_item = lexical_dict.get(item_id) or semantic_dict.get(item_id)
            if base_item:
                base_item.score = borda_score
                merged_results.append(base_item)
        
        merged_results.sort(key=lambda x: x.score, reverse=True)
        return merged_results
    
    def _combsum_fusion(
        self,
        lexical_results: List[SearchResultItem],
        semantic_results: List[SearchResultItem],
        weights: Dict[str, float]
    ) -> List[SearchResultItem]:
        """Fusion CombSUM (somme des scores)."""
        lexical_weight = weights.get("lexical_weight", self.config.default_lexical_weight)
        semantic_weight = weights.get("semantic_weight", self.config.default_semantic_weight)
        
        # Calculer les scores combinés
        combined_scores = {}
        
        # Ajouter les scores lexicaux
        for item in lexical_results:
            combined_scores[item.id] = item.score * lexical_weight
        
        # Ajouter les scores sémantiques
        for item in semantic_results:
            current_score = combined_scores.get(item.id, 0)
            combined_scores[item.id] = current_score + (item.score * semantic_weight)
        
        # Créer les résultats fusionnés
        merged_results = []
        lexical_dict = {item.id: item for item in lexical_results}
        semantic_dict = {item.id: item for item in semantic_results}
        
        for item_id, combined_score in combined_scores.items():
            base_item = lexical_dict.get(item_id) or semantic_dict.get(item_id)
            if base_item:
                base_item.score = combined_score
                merged_results.append(base_item)
        
        merged_results.sort(key=lambda x: x.score, reverse=True)
        return merged_results
    
    def _combmnz_fusion(
        self,
        lexical_results: List[SearchResultItem],
        semantic_results: List[SearchResultItem],
        weights: Dict[str, float]
    ) -> List[SearchResultItem]:
        """Fusion CombMNZ (somme * nombre de sources non-nulles)."""
        # D'abord calculer CombSUM
        combsum_results = self._combsum_fusion(lexical_results, semantic_results, weights)
        
        # Calculer le nombre de sources pour chaque item
        lexical_ids = {item.id for item in lexical_results}
        semantic_ids = {item.id for item in semantic_results}
        
        # Multiplier par le nombre de sources
        for item in combsum_results:
            source_count = 0
            if item.id in lexical_ids:
                source_count += 1
            if item.id in semantic_ids:
                source_count += 1
            
            item.score = item.score * source_count
        
        combsum_results.sort(key=lambda x: x.score, reverse=True)
        return combsum_results
    
    def _evaluate_result_quality(self, results: List[SearchResultItem]) -> float:
        """Évalue la qualité d'un ensemble de résultats."""
        if not results:
            return 0.0
        
        # Facteurs de qualité
        count_factor = min(len(results) / self.config.min_results_for_quality, 1.0)
        
        # Diversité des scores
        scores = [item.score for item in results]
        if len(scores) > 1:
            score_variance = sum((s - sum(scores) / len(scores)) ** 2 for s in scores) / len(scores)
            diversity_factor = min(score_variance, 1.0)
        else:
            diversity_factor = 0.5
        
        # Score moyen
        avg_score = sum(scores) / len(scores) if scores else 0.0
        
        return (count_factor + diversity_factor + avg_score) / 3.0
    
    def _post_process_results(
        self,
        results: List[SearchResultItem],
        query_analysis: QueryAnalysis,
        debug: bool
    ) -> List[SearchResultItem]:
        """Post-traitement des résultats fusionnés."""
        processed_results = results.copy()
        
        # 1. Déduplication si activée
        if self.config.enable_deduplication:
            processed_results = self._deduplicate_results(processed_results)
        
        # 2. Diversification si activée
        if self.config.enable_diversification:
            processed_results = self._diversify_results(processed_results)
        
        # 3. Boost de qualité selon l'analyse de requête
        processed_results = self._apply_quality_boost(processed_results, query_analysis)
        
        return processed_results
    
    def _deduplicate_results(self, results: List[SearchResultItem]) -> List[SearchResultItem]:
        """Supprime les doublons basés sur la similarité."""
        if not results:
            return results
        
        unique_results = []
        seen_ids = set()
        
        for item in results:
            if item.id not in seen_ids:
                unique_results.append(item)
                seen_ids.add(item.id)
        
        return unique_results
    
    def _diversify_results(self, results: List[SearchResultItem]) -> List[SearchResultItem]:
        """Applique la diversification des résultats."""
        if not results:
            return results
        
        diversified_results = []
        merchant_counts = {}
        
        for item in results:
            # Obtenir le marchand de l'item (à adapter selon votre modèle)
            merchant = getattr(item, 'merchant', 'unknown')
            
            current_count = merchant_counts.get(merchant, 0)
            
            if current_count < self.config.max_same_merchant:
                diversified_results.append(item)
                merchant_counts[merchant] = current_count + 1
        
        return diversified_results
    
    def _apply_quality_boost(
        self,
        results: List[SearchResultItem],
        query_analysis: QueryAnalysis
    ) -> List[SearchResultItem]:
        """Applique des boost de qualité selon l'analyse de requête."""
        if not results:
            return results
        
        boosted_results = []
        boost_factor = self.config.quality_boost_factor
        
        for item in results:
            boosted_score = item.score
            
            # Boost pour correspondance exacte de termes
            if getattr(query_analysis, 'has_exact_phrases', False):
                # Vérifier si l'item contient des phrases exactes (à adapter selon votre modèle)
                if hasattr(item, 'title') and any(
                    phrase.lower() in item.title.lower() 
                    for phrase in getattr(query_analysis, 'exact_phrases', [])
                ):
                    boosted_score *= boost_factor
            
            # Boost pour correspondance de montants
            if getattr(query_analysis, 'extracted_amount', None):
                # Vérifier si l'item a un montant similaire (à adapter selon votre modèle)
                if hasattr(item, 'amount') and item.amount:
                    query_amount = query_analysis.extracted_amount
                    item_amount = item.amount
                    
                    # Tolérance de 10% pour considérer les montants similaires
                    tolerance = 0.1
                    if abs(item_amount - query_amount) / query_amount <= tolerance:
                        boosted_score *= boost_factor
            
            item.score = boosted_score
            boosted_results.append(item)
        
        # Re-trier après application des boosts
        boosted_results.sort(key=lambda x: x.score, reverse=True)
        return boosted_results
    
    def _apply_pagination_and_sorting(
        self,
        results: List[SearchResultItem],
        offset: int,
        limit: int,
        sort_order: SortOrder
    ) -> List[SearchResultItem]:
        """Applique la pagination et le tri final."""
        # Tri selon l'ordre demandé
        if sort_order == SortOrder.RELEVANCE:
            # Déjà trié par score de pertinence
            sorted_results = results
        elif sort_order == SortOrder.DATE_DESC:
            sorted_results = sorted(
                results, 
                key=lambda x: getattr(x, 'created_at', ''), 
                reverse=True
            )
        elif sort_order == SortOrder.DATE_ASC:
            sorted_results = sorted(
                results, 
                key=lambda x: getattr(x, 'created_at', ''), 
                reverse=False
            )
        elif sort_order == SortOrder.AMOUNT_DESC:
            sorted_results = sorted(
                results, 
                key=lambda x: getattr(x, 'amount', 0), 
                reverse=True
            )
        elif sort_order == SortOrder.AMOUNT_ASC:
            sorted_results = sorted(
                results, 
                key=lambda x: getattr(x, 'amount', 0), 
                reverse=False
            )
        else:
            # Fallback vers relevance
            sorted_results = results
        
        # Appliquer la pagination
        end_index = offset + limit
        paginated_results = sorted_results[offset:end_index]
        
        return paginated_results
    
    def _assess_fusion_quality(
        self,
        final_results: List[SearchResultItem],
        lexical_result: LexicalSearchResult,
        semantic_result: SemanticSearchResult,
        query_analysis: QueryAnalysis
    ) -> SearchQuality:
        """Évalue la qualité globale de la fusion."""
        if not final_results:
            return SearchQuality.POOR
        
        # Facteurs de qualité
        result_count = len(final_results)
        min_for_good = self.config.min_results_for_quality
        
        # Score basé sur le nombre de résultats
        if result_count >= min_for_good:
            count_quality = SearchQuality.EXCELLENT
        elif result_count >= min_for_good * 0.7:
            count_quality = SearchQuality.GOOD
        elif result_count >= min_for_good * 0.4:
            count_quality = SearchQuality.FAIR
        else:
            count_quality = SearchQuality.POOR
        
        # Score basé sur la diversité des sources
        lexical_count = len(lexical_result.results)
        semantic_count = len(semantic_result.results)
        
        if lexical_count > 0 and semantic_count > 0:
            source_diversity = SearchQuality.EXCELLENT
        elif lexical_count > 0 or semantic_count > 0:
            source_diversity = SearchQuality.GOOD
        else:
            source_diversity = SearchQuality.POOR
        
        # Score basé sur la qualité moyenne des scores
        avg_score = sum(item.score for item in final_results) / len(final_results)
        
        if avg_score >= 0.8:
            score_quality = SearchQuality.EXCELLENT
        elif avg_score >= 0.6:
            score_quality = SearchQuality.GOOD
        elif avg_score >= 0.4:
            score_quality = SearchQuality.FAIR
        else:
            score_quality = SearchQuality.POOR
        
        # Calcul de la qualité finale (moyenne pondérée)
        quality_scores = {
            SearchQuality.POOR: 1,
            SearchQuality.FAIR: 2,
            SearchQuality.GOOD: 3,
            SearchQuality.EXCELLENT: 4
        }
        
        weighted_score = (
            quality_scores[count_quality] * 0.4 +
            quality_scores[source_diversity] * 0.3 +
            quality_scores[score_quality] * 0.3
        )
        
        # Retourner la qualité correspondante
        if weighted_score >= 3.5:
            return SearchQuality.EXCELLENT
        elif weighted_score >= 2.5:
            return SearchQuality.GOOD
        elif weighted_score >= 1.5:
            return SearchQuality.FAIR
        else:
            return SearchQuality.POOR
    
    def _create_empty_result(
        self,
        strategy: Optional[FusionStrategy],
        weights: Optional[Dict[str, float]],
        debug: bool
    ) -> FusionResult:
        """Crée un résultat vide."""
        return FusionResult(
            results=[],
            fusion_strategy=strategy.value if strategy else "none",
            weights_used=weights or {},
            processing_time_ms=0.0,
            lexical_count=0,
            semantic_count=0,
            unique_count=0,
            duplicates_removed=0,
            quality=SearchQuality.POOR,
            debug_info={"message": "No results from either engine"} if debug else None
        )
    
    def _create_single_engine_result(
        self,
        results: List[SearchResultItem],
        engine_type: str,
        weights: Optional[Dict[str, float]],
        offset: int,
        limit: int,
        sort_order: SortOrder,
        debug: bool
    ) -> FusionResult:
        """Crée un résultat à partir d'un seul moteur."""
        # Appliquer pagination et tri
        final_results = self._apply_pagination_and_sorting(results, offset, limit, sort_order)
        
        # Évaluer la qualité
        if len(final_results) >= self.config.min_results_for_quality:
            quality = SearchQuality.GOOD
        elif len(final_results) >= self.config.min_results_for_quality * 0.5:
            quality = SearchQuality.FAIR
        else:
            quality = SearchQuality.POOR
        
        return FusionResult(
            results=final_results,
            fusion_strategy=engine_type,
            weights_used=weights or {},
            processing_time_ms=0.0,
            lexical_count=len(results) if "lexical" in engine_type else 0,
            semantic_count=len(results) if "semantic" in engine_type else 0,
            unique_count=len(final_results),
            duplicates_removed=0,
            quality=quality,
            debug_info={"engine_used": engine_type} if debug else None
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
            "fusion_strategy": strategy.value,
            "lexical_results_count": len(lexical_result.results),
            "semantic_results_count": len(semantic_result.results),
            "fused_results_count": len(fused_results),
            "lexical_quality": lexical_result.quality.value,
            "semantic_quality": semantic_result.quality.value,
            "top_lexical_scores": [
                item.score for item in lexical_result.results[:5]
            ],
            "top_semantic_scores": [
                item.score for item in semantic_result.results[:5]
            ],
            "top_fused_scores": [
                item.score for item in fused_results[:5]
            ],
            "config_used": {
                "default_strategy": self.config.default_strategy.value,
                "default_lexical_weight": self.config.default_lexical_weight,
                "default_semantic_weight": self.config.default_semantic_weight,
                "enable_deduplication": self.config.enable_deduplication,
                "enable_diversification": self.config.enable_diversification
            }
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Retourne les métriques du fusionneur."""
        return {
            "total_fusions": self.fusion_count,
            "average_processing_time_ms": round(self.avg_processing_time, 2),
            "strategy_usage": self.strategy_usage.copy(),
            "quality_distribution": self.quality_distribution.copy(),
            "config_source": "centralized (config_service)",
            "current_config": {
                "default_strategy": self.config.default_strategy.value,
                "default_lexical_weight": self.config.default_lexical_weight,
                "default_semantic_weight": self.config.default_semantic_weight,
                "rrf_k": self.config.rrf_k,
                "enable_deduplication": self.config.enable_deduplication,
                "enable_diversification": self.config.enable_diversification
            }
        }
    
    def reset_metrics(self) -> None:
        """Remet à zéro les métriques."""
        self.fusion_count = 0
        self.strategy_usage = {strategy.value: 0 for strategy in FusionStrategy}
        self.quality_distribution = {quality.value: 0 for quality in SearchQuality}
        self.avg_processing_time = 0.0
        logger.info("Result merger metrics reset")


# ==========================================
# 🏭 FACTORY FUNCTIONS AVEC CONFIG CENTRALISÉE
# ==========================================

def create_result_merger(config: Optional[FusionConfig] = None) -> ResultMerger:
    """
    Factory function pour créer un result merger avec config centralisée.
    
    Args:
        config: Configuration personnalisée (optionnelle)
        
    Returns:
        Instance de ResultMerger configurée
    """
    if config is None:
        config = FusionConfig()
    
    merger = ResultMerger(config)
    logger.info(f"Created result merger with centralized config - strategy: {config.default_strategy.value}")
    return merger


def create_result_merger_with_preset(preset: str) -> ResultMerger:
    """
    Factory function pour créer un result merger avec un preset prédéfini.
    
    Args:
        preset: Nom du preset à utiliser
        
    Returns:
        Instance de ResultMerger avec configuration preset
        
    Presets disponibles:
        - "balanced": Configuration équilibrée (par défaut)
        - "lexical_focused": Favorise la recherche lexicale
        - "semantic_focused": Favorise la recherche sémantique
        - "quality_first": Optimisé pour la qualité des résultats
        - "speed_first": Optimisé pour la vitesse
        - "diverse": Optimisé pour la diversité des résultats
    """
    
    if preset == "balanced":
        config = FusionConfig(
            default_strategy=FusionStrategy.WEIGHTED_AVERAGE,
            default_lexical_weight=0.5,
            default_semantic_weight=0.5,
            enable_deduplication=True,
            enable_diversification=True
        )
    
    elif preset == "lexical_focused":
        config = FusionConfig(
            default_strategy=FusionStrategy.WEIGHTED_AVERAGE,
            default_lexical_weight=0.7,
            default_semantic_weight=0.3,
            enable_deduplication=True,
            enable_diversification=False
        )
    
    elif preset == "semantic_focused":
        config = FusionConfig(
            default_strategy=FusionStrategy.ADAPTIVE_FUSION,
            default_lexical_weight=0.3,
            default_semantic_weight=0.7,
            enable_deduplication=True,
            enable_diversification=True
        )
    
    elif preset == "quality_first":
        config = FusionConfig(
            default_strategy=FusionStrategy.ADAPTIVE_FUSION,
            default_lexical_weight=0.5,
            default_semantic_weight=0.5,
            quality_boost_factor=1.3,
            enable_deduplication=True,
            enable_diversification=True,
            min_results_for_quality=15
        )
    
    elif preset == "speed_first":
        config = FusionConfig(
            default_strategy=FusionStrategy.RANK_FUSION,
            default_lexical_weight=0.5,
            default_semantic_weight=0.5,
            enable_deduplication=False,
            enable_diversification=False
        )
    
    elif preset == "diverse":
        config = FusionConfig(
            default_strategy=FusionStrategy.BORDA_COUNT,
            default_lexical_weight=0.5,
            default_semantic_weight=0.5,
            enable_deduplication=True,
            enable_diversification=True,
            max_same_merchant=2
        )
    
    else:
        logger.warning(f"Unknown preset '{preset}', using balanced preset")
        config = FusionConfig()  # Configuration par défaut
    
    merger = ResultMerger(config)
    logger.info(f"Created result merger with preset '{preset}' - strategy: {config.default_strategy.value}")
    return merger


def get_fusion_config_summary() -> Dict[str, Any]:
    """
    Retourne un résumé de la configuration de fusion centralisée.
    
    Returns:
        Dictionnaire contenant les paramètres de configuration actuels
    """
    config = FusionConfig()
    
    return {
        "config_source": "centralized (config_service)",
        "default_strategy": config.default_strategy.value,
        "available_strategies": [strategy.value for strategy in FusionStrategy],
        "default_weights": {
            "lexical_weight": config.default_lexical_weight,
            "semantic_weight": config.default_semantic_weight
        },
        "rrf_parameters": {
            "k": config.rrf_k
        },
        "normalization": {
            "method": config.score_normalization_method
        },
        "adaptive_parameters": {
            "threshold": config.adaptive_threshold,
            "quality_boost_factor": config.quality_boost_factor
        },
        "post_processing": {
            "enable_deduplication": config.enable_deduplication,
            "dedup_similarity_threshold": config.dedup_similarity_threshold,
            "enable_diversification": config.enable_diversification,
            "max_same_merchant": config.max_same_merchant
        },
        "quality_assessment": {
            "min_results_for_good_quality": config.min_results_for_quality
        },
        "available_presets": [
            "balanced", "lexical_focused", "semantic_focused", 
            "quality_first", "speed_first", "diverse"
        ]
    }


# ==========================================
# 🎯 EXPORTS PRINCIPAUX
# ==========================================

__all__ = [
    # Classes et enums
    "ResultMerger",
    "FusionConfig",
    "FusionStrategy",
    "FusionResult",
    
    # Factory functions
    "create_result_merger",
    "create_result_merger_with_preset",
    "get_fusion_config_summary"
]