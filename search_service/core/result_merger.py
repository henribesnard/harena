"""
Fusion et classement des résultats de recherche hybride.

Ce module implémente différentes stratégies de fusion pour combiner
les résultats lexicaux (Elasticsearch) et sémantiques (Qdrant) de manière optimale.
"""
import logging
import math
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum

from search_service.core.lexical_engine import LexicalSearchResult
from search_service.core.semantic_engine import SemanticSearchResult
from search_service.core.query_processor import QueryAnalysis
from search_service.models.search_types import SearchQuality, SortOrder
from search_service.models.responses import SearchResultItem

logger = logging.getLogger(__name__)


class FusionStrategy(Enum):
    """Stratégies de fusion des résultats."""
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
    """Configuration pour la fusion des résultats."""
    # Stratégie de fusion par défaut
    default_strategy: FusionStrategy = FusionStrategy.ADAPTIVE_FUSION
    
    # Poids par défaut
    default_lexical_weight: float = 0.6
    default_semantic_weight: float = 0.4
    
    # Paramètres de normalisation
    score_normalization_method: str = "min_max"  # "min_max", "z_score", "sigmoid"
    min_score_threshold: float = 0.01
    
    # Paramètres pour RRF (Reciprocal Rank Fusion)
    rrf_k: int = 60
    
    # Paramètres adaptatifs
    adaptive_threshold: float = 0.1
    quality_boost_factor: float = 0.2
    
    # Déduplication
    enable_deduplication: bool = True
    dedup_similarity_threshold: float = 0.95
    
    # Diversification
    enable_diversification: bool = True
    diversity_factor: float = 0.1
    max_same_merchant: int = 3


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
    - Normalisation des scores provenant de différents moteurs
    - Déduplication avancée des résultats
    - Diversification pour éviter la redondance
    - Évaluation de la qualité de fusion
    """
    
    def __init__(self, config: Optional[FusionConfig] = None):
        self.config = config or FusionConfig()
        
        # Métriques de fusion
        self.fusion_count = 0
        self.strategy_usage = {strategy.value: 0 for strategy in FusionStrategy}
        self.quality_distribution = {quality.value: 0 for quality in SearchQuality}
        self.avg_processing_time = 0.0
        
        logger.info("Result merger initialized")
    
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
        import time
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
            strategy = self._determine_optimal_strategy(
                lexical_result, semantic_result, query_analysis
            )
        
        # 3. Déterminer les poids optimaux si non spécifiés
        if weights is None:
            weights = self._determine_optimal_weights(
                lexical_result, semantic_result, query_analysis
            )
        
        # 4. Indexer les résultats par transaction_id
        lexical_by_id = {r.transaction_id: r for r in lexical_result.results}
        semantic_by_id = {r.transaction_id: r for r in semantic_result.results}
        
        # 5. Effectuer la fusion selon la stratégie choisie
        fused_results = self._execute_fusion_strategy(
            strategy, lexical_result.results, semantic_result.results,
            lexical_by_id, semantic_by_id, weights, query_analysis, debug
        )
        
        # 6. Post-traitement des résultats
        processed_results = self._post_process_results(
            fused_results, query_analysis, debug
        )
        
        # 7. Appliquer pagination et tri
        final_results = self._apply_pagination_and_sorting(
            processed_results, offset, limit, sort_order
        )
        
        # 8. Calculer les métriques et qualité
        processing_time = (time.time() - start_time) * 1000
        self.avg_processing_time = (
            (self.avg_processing_time * (self.fusion_count - 1) + processing_time) 
            / self.fusion_count
        )
        
        quality = self._assess_fusion_quality(
            final_results, lexical_result, semantic_result, query_analysis
        )
        
        # 9. Mettre à jour les statistiques
        self.strategy_usage[strategy.value] += 1
        self.quality_distribution[quality.value] += 1
        
        # 10. Construire le résultat final
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
        """Détermine la stratégie de fusion optimale."""
        # Analyser la qualité des résultats de chaque moteur
        lexical_quality = lexical_result.quality
        semantic_quality = semantic_result.quality
        
        # Si une recherche est de très mauvaise qualité
        if lexical_quality == SearchQuality.POOR and semantic_quality != SearchQuality.POOR:
            return FusionStrategy.WEIGHTED_AVERAGE  # Favoriser le sémantique
        
        if semantic_quality == SearchQuality.POOR and lexical_quality != SearchQuality.POOR:
            return FusionStrategy.WEIGHTED_AVERAGE  # Favoriser le lexical
        
        # CORRECTION: Utiliser les propriétés correctes de QueryAnalysis
        
        # Pour requêtes très spécifiques (phrases exactes)
        if query_analysis.has_exact_phrases:
            return FusionStrategy.RANK_FUSION
        
        # Pour requêtes avec entités financières
        if query_analysis.has_financial_entities:
            return FusionStrategy.RECIPROCAL_RANK_FUSION
        
        # Pour requêtes courtes et simples
        if len(query_analysis.key_terms) <= 2:
            return FusionStrategy.SCORE_NORMALIZATION
        
        # Stratégie adaptative par défaut
        return FusionStrategy.ADAPTIVE_FUSION
    
    def _determine_optimal_weights(
        self,
        lexical_result: LexicalSearchResult,
        semantic_result: SemanticSearchResult,
        query_analysis: QueryAnalysis
    ) -> Dict[str, float]:
        """Détermine les poids optimaux pour la fusion."""
        base_lexical = self.config.default_lexical_weight
        base_semantic = self.config.default_semantic_weight
        
        # Ajustements basés sur la qualité des résultats
        quality_diff = self._quality_to_score(lexical_result.quality) - \
                      self._quality_to_score(semantic_result.quality)
        
        # Ajuster les poids selon la différence de qualité
        weight_adjustment = quality_diff * 0.2
        
        lexical_weight = max(0.1, min(0.9, base_lexical + weight_adjustment))
        semantic_weight = 1.0 - lexical_weight
        
        # CORRECTION: Utiliser les propriétés correctes de QueryAnalysis
        
        # Ajustements spécifiques selon le type de requête
        if query_analysis.has_exact_phrases:
            # Favoriser le lexical pour les phrases exactes
            lexical_weight += 0.1
            semantic_weight -= 0.1
        
        if query_analysis.has_financial_entities:
            # Équilibrer pour les entités financières
            lexical_weight = 0.5
            semantic_weight = 0.5
        
        if len(query_analysis.key_terms) > 5:
            # Favoriser le sémantique pour les requêtes complexes
            semantic_weight += 0.1
            lexical_weight -= 0.1
        
        # Normaliser pour assurer que la somme = 1.0
        total = lexical_weight + semantic_weight
        lexical_weight /= total
        semantic_weight /= total
        
        return {
            "lexical_weight": lexical_weight,
            "semantic_weight": semantic_weight
        }
    
    def _quality_to_score(self, quality: SearchQuality) -> float:
        """Convertit une qualité en score numérique."""
        quality_scores = {
            SearchQuality.EXCELLENT: 1.0,
            SearchQuality.GOOD: 0.7,
            SearchQuality.MEDIUM: 0.5,
            SearchQuality.POOR: 0.2
        }
        return quality_scores.get(quality, 0.5)
    
    def _execute_fusion_strategy(
        self,
        strategy: FusionStrategy,
        lexical_results: List[SearchResultItem],
        semantic_results: List[SearchResultItem],
        lexical_by_id: Dict[int, SearchResultItem],
        semantic_by_id: Dict[int, SearchResultItem],
        weights: Dict[str, float],
        query_analysis: QueryAnalysis,
        debug: bool
    ) -> List[SearchResultItem]:
        """Exécute la stratégie de fusion spécifiée."""
        
        if strategy == FusionStrategy.WEIGHTED_AVERAGE:
            return self._fuse_weighted_average(
                lexical_by_id, semantic_by_id, weights, debug
            )
        
        elif strategy == FusionStrategy.RANK_FUSION:
            return self._fuse_rank_fusion(
                lexical_results, semantic_results, weights, debug
            )
        
        elif strategy == FusionStrategy.RECIPROCAL_RANK_FUSION:
            return self._fuse_reciprocal_rank_fusion(
                lexical_results, semantic_results, weights, debug
            )
        
        elif strategy == FusionStrategy.SCORE_NORMALIZATION:
            return self._fuse_score_normalization(
                lexical_by_id, semantic_by_id, weights, debug
            )
        
        elif strategy == FusionStrategy.ADAPTIVE_FUSION:
            return self._fuse_adaptive(
                lexical_results, semantic_results, lexical_by_id, 
                semantic_by_id, weights, query_analysis, debug
            )
        
        elif strategy == FusionStrategy.BORDA_COUNT:
            return self._fuse_borda_count(
                lexical_results, semantic_results, weights, debug
            )
        
        elif strategy == FusionStrategy.COMBSUM:
            return self._fuse_combsum(
                lexical_by_id, semantic_by_id, weights, debug
            )
        
        elif strategy == FusionStrategy.COMBMNZ:
            return self._fuse_combmnz(
                lexical_by_id, semantic_by_id, weights, debug
            )
        
        else:
            # Fallback vers weighted average
            return self._fuse_weighted_average(
                lexical_by_id, semantic_by_id, weights, debug
            )
    
    def _fuse_weighted_average(
        self,
        lexical_by_id: Dict[int, SearchResultItem],
        semantic_by_id: Dict[int, SearchResultItem],
        weights: Dict[str, float],
        debug: bool
    ) -> List[SearchResultItem]:
        """Fusion par moyenne pondérée des scores."""
        fused_results = []
        all_transaction_ids = set(lexical_by_id.keys()) | set(semantic_by_id.keys())
        
        lexical_weight = weights.get("lexical_weight", 0.6)
        semantic_weight = weights.get("semantic_weight", 0.4)
        
        for transaction_id in all_transaction_ids:
            lexical_item = lexical_by_id.get(transaction_id)
            semantic_item = semantic_by_id.get(transaction_id)
            
            # Utiliser l'item avec le plus d'informations comme base
            base_item = lexical_item or semantic_item
            
            # Normaliser les scores
            lexical_score = self._normalize_lexical_score(
                lexical_item.score if lexical_item else 0.0
            )
            semantic_score = semantic_item.score if semantic_item else 0.0
            
            # Calculer le score combiné
            combined_score = (
                lexical_score * lexical_weight + 
                semantic_score * semantic_weight
            )
            
            # Créer l'item fusionné
            fused_item = self._create_fused_item(
                base_item, lexical_item, semantic_item, combined_score,
                "weighted_average", {"lexical_weight": lexical_weight, "semantic_weight": semantic_weight}
            )
            
            fused_results.append(fused_item)
        
        return fused_results
    
    def _fuse_rank_fusion(
        self,
        lexical_results: List[SearchResultItem],
        semantic_results: List[SearchResultItem],
        weights: Dict[str, float],
        debug: bool
    ) -> List[SearchResultItem]:
        """Fusion basée sur les rangs dans chaque liste."""
        lexical_weight = weights.get("lexical_weight", 0.6)
        semantic_weight = weights.get("semantic_weight", 0.4)
        
        # Créer des mappings rang -> transaction_id
        lexical_ranks = {item.transaction_id: i + 1 for i, item in enumerate(lexical_results)}
        semantic_ranks = {item.transaction_id: i + 1 for i, item in enumerate(semantic_results)}
        
        # Indexer par transaction_id
        lexical_by_id = {r.transaction_id: r for r in lexical_results}
        semantic_by_id = {r.transaction_id: r for r in semantic_results}
        
        all_transaction_ids = set(lexical_ranks.keys()) | set(semantic_ranks.keys())
        fused_results = []
        
        for transaction_id in all_transaction_ids:
            lexical_item = lexical_by_id.get(transaction_id)
            semantic_item = semantic_by_id.get(transaction_id)
            base_item = lexical_item or semantic_item
            
            # Calculer le score basé sur les rangs (plus bas = meilleur)
            lexical_rank = lexical_ranks.get(transaction_id, len(lexical_results) + 1)
            semantic_rank = semantic_ranks.get(transaction_id, len(semantic_results) + 1)
            
            # Convertir les rangs en scores (1/rang)
            lexical_rank_score = 1.0 / lexical_rank
            semantic_rank_score = 1.0 / semantic_rank
            
            combined_rank_score = (
                lexical_rank_score * lexical_weight + 
                semantic_rank_score * semantic_weight
            )
            
            fused_item = self._create_fused_item(
                base_item, lexical_item, semantic_item, combined_rank_score,
                "rank_fusion", {
                    "lexical_rank": lexical_rank, 
                    "semantic_rank": semantic_rank,
                    "lexical_weight": lexical_weight,
                    "semantic_weight": semantic_weight
                }
            )
            
            fused_results.append(fused_item)
        
        return fused_results
    
    def _fuse_reciprocal_rank_fusion(
        self,
        lexical_results: List[SearchResultItem],
        semantic_results: List[SearchResultItem],
        weights: Dict[str, float],
        debug: bool
    ) -> List[SearchResultItem]:
        """Fusion RRF (Reciprocal Rank Fusion)."""
        k = self.config.rrf_k
        
        # Créer des mappings rang -> transaction_id
        lexical_ranks = {item.transaction_id: i + 1 for i, item in enumerate(lexical_results)}
        semantic_ranks = {item.transaction_id: i + 1 for i, item in enumerate(semantic_results)}
        
        # Indexer par transaction_id
        lexical_by_id = {r.transaction_id: r for r in lexical_results}
        semantic_by_id = {r.transaction_id: r for r in semantic_results}
        
        all_transaction_ids = set(lexical_ranks.keys()) | set(semantic_ranks.keys())
        fused_results = []
        
        for transaction_id in all_transaction_ids:
            lexical_item = lexical_by_id.get(transaction_id)
            semantic_item = semantic_by_id.get(transaction_id)
            base_item = lexical_item or semantic_item
            
            # Calculer RRF score
            lexical_rank = lexical_ranks.get(transaction_id, len(lexical_results) + 1)
            semantic_rank = semantic_ranks.get(transaction_id, len(semantic_results) + 1)
            
            rrf_score = 0.0
            if lexical_item:
                rrf_score += 1.0 / (k + lexical_rank)
            if semantic_item and semantic_item.score > 0:
                non_zero_engines += 1
                score_sum += semantic_item.score
            
            # CombMNZ = CombSUM * nombre de moteurs
            combmnz_score = score_sum * non_zero_engines
            
            fused_item = self._create_fused_item(
                base_item, lexical_item, semantic_item, combmnz_score,
                "combmnz", {"non_zero_engines": non_zero_engines}
            )
            
            fused_results.append(fused_item)
        
        return fused_results
    
    def _normalize_lexical_score(self, score: float) -> float:
        """Normalise un score Elasticsearch (typiquement 1-20) vers 0-1."""
        # Elasticsearch scores sont généralement entre 1 et 20
        return min(score / 15.0, 1.0)
    
    def _min_max_normalize(self, scores: List[float]) -> List[float]:
        """Normalisation Min-Max."""
        if not scores or len(scores) == 1:
            return scores
        
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return [0.5] * len(scores)
        
        return [(s - min_score) / (max_score - min_score) for s in scores]
    
    def _z_score_normalize(self, scores: List[float]) -> List[float]:
        """Normalisation Z-Score."""
        if not scores or len(scores) == 1:
            return scores
        
        mean_score = sum(scores) / len(scores)
        variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
        std_dev = math.sqrt(variance)
        
        if std_dev == 0:
            return [0.5] * len(scores)
        
        z_scores = [(s - mean_score) / std_dev for s in scores]
        
        # Convertir en scores positifs entre 0-1 avec sigmoid
        return [1 / (1 + math.exp(-z)) for z in z_scores]
    
    def _sigmoid_normalize(self, scores: List[float]) -> List[float]:
        """Normalisation Sigmoid."""
        if not scores:
            return scores
        
        mean_score = sum(scores) / len(scores)
        return [1 / (1 + math.exp(-(s - mean_score))) for s in scores]
    
    def _create_fused_item(
        self,
        base_item: SearchResultItem,
        lexical_item: Optional[SearchResultItem],
        semantic_item: Optional[SearchResultItem],
        combined_score: float,
        fusion_method: str,
        fusion_metadata: Dict[str, Any]
    ) -> SearchResultItem:
        """Crée un SearchResultItem fusionné."""
        
        # Fusionner les métadonnées
        metadata = base_item.metadata.copy() if base_item.metadata else {}
        metadata.update({
            "fusion_method": fusion_method,
            "found_in_lexical": lexical_item is not None,
            "found_in_semantic": semantic_item is not None,
            **fusion_metadata
        })
        
        # Privilégier les highlights du lexical s'ils existent
        highlights = None
        if lexical_item and lexical_item.highlights:
            highlights = lexical_item.highlights
        elif semantic_item and semantic_item.highlights:
            highlights = semantic_item.highlights
        
        return SearchResultItem(
            transaction_id=base_item.transaction_id,
            user_id=base_item.user_id,
            account_id=base_item.account_id,
            score=combined_score,
            lexical_score=lexical_item.score if lexical_item else None,
            semantic_score=semantic_item.score if semantic_item else None,
            combined_score=combined_score,
            primary_description=base_item.primary_description,
            searchable_text=base_item.searchable_text,
            merchant_name=base_item.merchant_name,
            amount=base_item.amount,
            currency_code=base_item.currency_code,
            transaction_type=base_item.transaction_type,
            transaction_date=base_item.transaction_date,
            created_at=base_item.created_at,
            category_id=base_item.category_id,
            operation_type=base_item.operation_type,
            highlights=highlights,
            metadata=metadata,
            explanation=base_item.explanation
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
            fused_results = self._deduplicate_results(fused_results, debug)
        
        # 2. Diversification
        if self.config.enable_diversification:
            fused_results = self._diversify_results(fused_results, debug)
        
        # 3. Boost de qualité pour certains résultats
        fused_results = self._apply_quality_boost(fused_results, query_analysis, debug)
        
        return fused_results
    
    def _deduplicate_results(
        self,
        results: List[SearchResultItem],
        debug: bool
    ) -> List[SearchResultItem]:
        """Déduplique les résultats basés sur la similarité."""
        if len(results) <= 1:
            return results
        
        deduplicated = []
        seen_signatures = set()
        
        for result in results:
            # Créer une signature pour la transaction
            signature = self._create_transaction_signature(result)
            
            # Vérifier la similarité avec les transactions déjà vues
            is_duplicate = False
            for seen_sig in seen_signatures:
                if self._calculate_signature_similarity(signature, seen_sig) > self.config.dedup_similarity_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                deduplicated.append(result)
                seen_signatures.add(signature)
            elif debug:
                # Marquer comme dupliqué dans les métadonnées pour debug
                if result.metadata:
                    result.metadata["duplicate_removed"] = True
        
        return deduplicated
    
    def _create_transaction_signature(self, result: SearchResultItem) -> str:
        """Crée une signature pour identifier les doublons."""
        # Utiliser plusieurs champs pour créer une signature
        components = [
            str(result.amount or 0),
            result.transaction_date or "",
            (result.merchant_name or "").lower()[:20],
            (result.primary_description or "").lower()[:30]
        ]
        
        return "|".join(components)
    
    def _calculate_signature_similarity(self, sig1: str, sig2: str) -> float:
        """Calcule la similarité entre deux signatures."""
        if sig1 == sig2:
            return 1.0
        
        # Similarité basée sur les composants
        parts1 = sig1.split("|")
        parts2 = sig2.split("|")
        
        if len(parts1) != len(parts2):
            return 0.0
        
        similarities = []
        for p1, p2 in zip(parts1, parts2):
            if p1 == p2:
                similarities.append(1.0)
            elif p1 and p2:
                # Similarité de chaînes simple
                longer = max(len(p1), len(p2))
                common = sum(c1 == c2 for c1, c2 in zip(p1, p2))
                similarities.append(common / longer)
            else:
                similarities.append(0.0)
        
        return sum(similarities) / len(similarities)
    
    def _diversify_results(
        self,
        results: List[SearchResultItem],
        debug: bool
    ) -> List[SearchResultItem]:
        """Diversifie les résultats pour éviter la redondance."""
        if len(results) <= self.config.max_same_merchant:
            return results
        
        diversified = []
        merchant_counts = {}
        
        # Trier par score décroissant d'abord
        sorted_results = sorted(results, key=lambda x: x.score or 0, reverse=True)
        
        for result in sorted_results:
            merchant = result.merchant_name or "unknown"
            current_count = merchant_counts.get(merchant, 0)
            
            # Ajouter si on n'a pas atteint la limite pour ce marchand
            if current_count < self.config.max_same_merchant:
                diversified.append(result)
                merchant_counts[merchant] = current_count + 1
            elif debug:
                # Marquer comme filtré pour diversité
                if result.metadata:
                    result.metadata["diversity_filtered"] = True
        
        return diversified
    
    def _apply_quality_boost(
        self,
        results: List[SearchResultItem],
        query_analysis: QueryAnalysis,
        debug: bool
    ) -> List[SearchResultItem]:
        """Applique un boost de qualité aux résultats pertinents."""
        
        for result in results:
            boost_factor = 1.0
            boost_reasons = []
            
            # CORRECTION: Utiliser la propriété key_terms correcte
            # Boost pour correspondance avec les termes de la requête
            if query_analysis.key_terms:
                text_content = " ".join(filter(None, [
                    result.primary_description,
                    result.merchant_name,
                    result.searchable_text
                ])).lower()
                
                matching_terms = sum(
                    1 for term in query_analysis.key_terms
                    if term.lower() in text_content
                )
                
                if matching_terms > 0:
                    term_boost = 1 + (matching_terms / len(query_analysis.key_terms)) * self.config.quality_boost_factor
                    boost_factor *= term_boost
                    boost_reasons.append(f"term_match_{matching_terms}")
            
            # Boost pour transactions récentes
            if result.transaction_date:
                # Simple boost pour les transactions récentes (implémentation basique)
                boost_factor *= 1.05
                boost_reasons.append("recent_transaction")
            
            # Boost pour montants ronds (souvent plus mémorables)
            if result.amount and result.amount == int(result.amount):
                boost_factor *= 1.02
                boost_reasons.append("round_amount")
            
            # Appliquer le boost
            if boost_factor > 1.0:
                original_score = result.score
                result.score *= boost_factor
                result.combined_score *= boost_factor
                
                if debug and result.metadata:
                    result.metadata.update({
                        "quality_boost_applied": boost_factor,
                        "boost_reasons": boost_reasons,
                        "original_score": original_score
                    })
        
        return results
    
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
    
    def _assess_fusion_quality(
        self,
        results: List[SearchResultItem],
        lexical_result: LexicalSearchResult,
        semantic_result: SemanticSearchResult,
        query_analysis: QueryAnalysis
    ) -> SearchQuality:
        """Évalue la qualité de la fusion."""
        if not results:
            return SearchQuality.POOR
        
        # Facteurs de qualité
        score_quality = self._assess_fused_score_quality(results)
        coverage_quality = self._assess_coverage_quality(results, lexical_result, semantic_result)
        consistency_quality = self._assess_fusion_consistency(results)
        relevance_quality = self._assess_fused_relevance(results, query_analysis)
        
        # Moyenne pondérée
        overall_quality = (
            score_quality * 0.3 +
            coverage_quality * 0.25 +
            consistency_quality * 0.25 +
            relevance_quality * 0.2
        )
        
        # Conversion en enum
        if overall_quality >= 0.8:
            return SearchQuality.EXCELLENT
        elif overall_quality >= 0.6:
            return SearchQuality.GOOD
        elif overall_quality >= 0.4:
            return SearchQuality.MEDIUM
        else:
            return SearchQuality.POOR
    
    def _assess_fused_score_quality(self, results: List[SearchResultItem]) -> float:
        """Évalue la qualité des scores fusionnés."""
        if not results:
            return 0.0
        
        scores = [r.score for r in results if r.score]
        if not scores:
            return 0.0
        
        # Vérifier la distribution des scores
        max_score = max(scores)
        min_score = min(scores)
        avg_score = sum(scores) / len(scores)
        
        # Qualité basée sur le score max et la distribution
        max_quality = min(max_score, 1.0)
        distribution_quality = (max_score - min_score) / max_score if max_score > 0 else 0
        avg_quality = avg_score
        
        return (max_quality * 0.4 + distribution_quality * 0.3 + avg_quality * 0.3)
    
    def _assess_coverage_quality(
        self,
        results: List[SearchResultItem],
        lexical_result: LexicalSearchResult,
        semantic_result: SemanticSearchResult
    ) -> float:
        """Évalue la couverture de la fusion."""
        if not results:
            return 0.0
        
        # Compter les résultats de chaque moteur
        both_engines = sum(
            1 for r in results 
            if r.metadata and r.metadata.get("found_in_lexical") and r.metadata.get("found_in_semantic")
        )
        
        lexical_only = sum(
            1 for r in results
            if r.metadata and r.metadata.get("found_in_lexical") and not r.metadata.get("found_in_semantic")
        )
        
        semantic_only = sum(
            1 for r in results
            if r.metadata and not r.metadata.get("found_in_lexical") and r.metadata.get("found_in_semantic")
        )
        
        total_results = len(results)
        
        # Qualité basée sur la diversité des sources
        if total_results == 0:
            return 0.0
        
        both_ratio = both_engines / total_results
        diversity_ratio = (lexical_only + semantic_only) / total_results
        
        # Bonus pour avoir des résultats des deux moteurs
        return both_ratio * 0.6 + diversity_ratio * 0.4
    
    def _assess_fusion_consistency(self, results: List[SearchResultItem]) -> float:
        """Évalue la cohérence de la fusion."""
        if len(results) <= 1:
            return 1.0
        
        # Vérifier la cohérence des scores
        scores = [r.score for r in results if r.score]
        if len(scores) < 2:
            return 0.5
        
        # Calculer la cohérence des écarts
        score_gaps = [scores[i] - scores[i+1] for i in range(len(scores)-1)]
        
        if not score_gaps:
            return 1.0
        
        avg_gap = sum(score_gaps) / len(score_gaps)
        max_gap = max(score_gaps)
        
        # Bonne cohérence = écarts réguliers
        if max_gap == 0:
            return 1.0
        
        consistency = 1 - abs(avg_gap - max_gap) / max_gap
        return max(0.0, consistency)
    
    def _assess_fused_relevance(
        self,
        results: List[SearchResultItem],
        query_analysis: QueryAnalysis
    ) -> float:
        """Évalue la pertinence des résultats fusionnés."""
        # CORRECTION: Utiliser la propriété key_terms correcte
        if not results or not query_analysis.key_terms:
            return 0.5
        
        relevance_scores = []
        
        for result in results:
            text_content = " ".join(filter(None, [
                result.primary_description,
                result.merchant_name,
                result.searchable_text
            ])).lower()
            
            # Compter les termes qui matchent
            matching_terms = sum(
                1 for term in query_analysis.key_terms
                if term.lower() in text_content
            )
            
            relevance = matching_terms / len(query_analysis.key_terms)
            relevance_scores.append(relevance)
        
        return sum(relevance_scores) / len(relevance_scores)
    
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
        import time
        start_time = time.time()
        
        # Appliquer pagination et tri
        final_results = self._apply_pagination_and_sorting(results, offset, limit, sort_order)
        
        processing_time = (time.time() - start_time) * 1000
        
        # Évaluer la qualité basée sur les scores
        quality = SearchQuality.MEDIUM
        if results:
            avg_score = sum(r.score for r in results if r.score) / len(results)
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
        logger.info("Result merger configuration updated")
    
    def _fuse_score_normalization(
        self,
        lexical_by_id: Dict[int, SearchResultItem],
        semantic_by_id: Dict[int, SearchResultItem],
        weights: Dict[str, float],
        debug: bool
    ) -> List[SearchResultItem]:
        """Fusion avec normalisation avancée des scores."""
        # Extraire tous les scores pour normalisation
        lexical_scores = [item.score for item in lexical_by_id.values() if item.score]
        semantic_scores = [item.score for item in semantic_by_id.values() if item.score]
        
        # Normaliser les scores selon la méthode configurée
        if self.config.score_normalization_method == "min_max":
            lexical_norm = self._min_max_normalize(lexical_scores)
            semantic_norm = self._min_max_normalize(semantic_scores)
        elif self.config.score_normalization_method == "z_score":
            lexical_norm = self._z_score_normalize(lexical_scores)
            semantic_norm = self._z_score_normalize(semantic_scores)
        else:  # sigmoid
            lexical_norm = self._sigmoid_normalize(lexical_scores)
            semantic_norm = self._sigmoid_normalize(semantic_scores)
        
        # Créer des mappings score -> score normalisé
        lexical_score_map = dict(zip(lexical_scores, lexical_norm))
        semantic_score_map = dict(zip(semantic_scores, semantic_norm))
        
        all_transaction_ids = set(lexical_by_id.keys()) | set(semantic_by_id.keys())
        fused_results = []
        
        lexical_weight = weights.get("lexical_weight", 0.6)
        semantic_weight = weights.get("semantic_weight", 0.4)
        
        for transaction_id in all_transaction_ids:
            lexical_item = lexical_by_id.get(transaction_id)
            semantic_item = semantic_by_id.get(transaction_id)
            base_item = lexical_item or semantic_item
            
            # Obtenir les scores normalisés
            norm_lexical = lexical_score_map.get(lexical_item.score, 0.0) if lexical_item else 0.0
            norm_semantic = semantic_score_map.get(semantic_item.score, 0.0) if semantic_item else 0.0
            
            # Calculer le score final
            combined_score = norm_lexical * lexical_weight + norm_semantic * semantic_weight
            
            fused_item = self._create_fused_item(
                base_item, lexical_item, semantic_item, combined_score,
                "score_normalization", {
                    "normalization_method": self.config.score_normalization_method,
                    "lexical_weight": lexical_weight,
                    "semantic_weight": semantic_weight
                }
            )
            
            fused_results.append(fused_item)
        
        return fused_results
    
    def _fuse_adaptive(
        self,
        lexical_results: List[SearchResultItem],
        semantic_results: List[SearchResultItem],
        lexical_by_id: Dict[int, SearchResultItem],
        semantic_by_id: Dict[int, SearchResultItem],
        weights: Dict[str, float],
        query_analysis: QueryAnalysis,
        debug: bool
    ) -> List[SearchResultItem]:
        """Fusion adaptative qui choisit la meilleure méthode par transaction."""
        fused_results = []
        all_transaction_ids = set(lexical_by_id.keys()) | set(semantic_by_id.keys())
        
        for transaction_id in all_transaction_ids:
            lexical_item = lexical_by_id.get(transaction_id)
            semantic_item = semantic_by_id.get(transaction_id)
            base_item = lexical_item or semantic_item
            
            # Déterminer la méthode optimale pour cette transaction
            if lexical_item and semantic_item:
                # Les deux moteurs ont trouvé cette transaction
                score_diff = abs(
                    self._normalize_lexical_score(lexical_item.score) - semantic_item.score
                )
                
                if score_diff < self.config.adaptive_threshold:
                    # Scores similaires -> moyenne pondérée
                    combined_score = (
                        self._normalize_lexical_score(lexical_item.score) * weights["lexical_weight"] +
                        semantic_item.score * weights["semantic_weight"]
                    )
                    method = "weighted_average"
                else:
                    # Scores différents -> favoriser le meilleur
                    if self._normalize_lexical_score(lexical_item.score) > semantic_item.score:
                        combined_score = self._normalize_lexical_score(lexical_item.score) * 1.1
                        method = "lexical_boosted"
                    else:
                        combined_score = semantic_item.score * 1.1
                        method = "semantic_boosted"
            
            elif lexical_item:
                # Seulement lexical
                combined_score = self._normalize_lexical_score(lexical_item.score)
                method = "lexical_only"
            
            else:
                # Seulement sémantique
                combined_score = semantic_item.score
                method = "semantic_only"
            
            fused_item = self._create_fused_item(
                base_item, lexical_item, semantic_item, combined_score,
                "adaptive_fusion", {"adaptive_method": method}
            )
            
            fused_results.append(fused_item)
        
        return fused_results
    
    def _fuse_borda_count(
        self,
        lexical_results: List[SearchResultItem],
        semantic_results: List[SearchResultItem],
        weights: Dict[str, float],
        debug: bool
    ) -> List[SearchResultItem]:
        """Fusion Borda Count."""
        lexical_count = len(lexical_results)
        semantic_count = len(semantic_results)
        
        # Calculer les points Borda pour chaque liste
        lexical_points = {}
        for i, item in enumerate(lexical_results):
            lexical_points[item.transaction_id] = lexical_count - i
        
        semantic_points = {}
        for i, item in enumerate(semantic_results):
            semantic_points[item.transaction_id] = semantic_count - i
        
        # Indexer par transaction_id
        lexical_by_id = {r.transaction_id: r for r in lexical_results}
        semantic_by_id = {r.transaction_id: r for r in semantic_results}
        
        all_transaction_ids = set(lexical_points.keys()) | set(semantic_points.keys())
        fused_results = []
        
        lexical_weight = weights.get("lexical_weight", 0.6)
        semantic_weight = weights.get("semantic_weight", 0.4)
        
        for transaction_id in all_transaction_ids:
            lexical_item = lexical_by_id.get(transaction_id)
            semantic_item = semantic_by_id.get(transaction_id)
            base_item = lexical_item or semantic_item
            
            # Calculer le score Borda
            lex_points = lexical_points.get(transaction_id, 0)
            sem_points = semantic_points.get(transaction_id, 0)
            
            borda_score = (lex_points * lexical_weight + sem_points * semantic_weight)
            
            fused_item = self._create_fused_item(
                base_item, lexical_item, semantic_item, borda_score,
                "borda_count", {
                    "lexical_points": lex_points,
                    "semantic_points": sem_points
                }
            )
            
            fused_results.append(fused_item)
        
        return fused_results
    
    def _fuse_combsum(
        self,
        lexical_by_id: Dict[int, SearchResultItem],
        semantic_by_id: Dict[int, SearchResultItem],
        weights: Dict[str, float],
        debug: bool
    ) -> List[SearchResultItem]:
        """Fusion CombSUM (somme des scores normalisés)."""
        return self._fuse_weighted_average(lexical_by_id, semantic_by_id, {
            "lexical_weight": 1.0, "semantic_weight": 1.0
        }, debug)
    
    def _fuse_combmnz(
        self,
        lexical_by_id: Dict[int, SearchResultItem],
        semantic_by_id: Dict[int, SearchResultItem],
        weights: Dict[str, float],
        debug: bool
    ) -> List[SearchResultItem]:
        """Fusion CombMNZ (CombSUM * nombre de moteurs non-zéros)."""
        fused_results = []
        all_transaction_ids = set(lexical_by_id.keys()) | set(semantic_by_id.keys())
        
        for transaction_id in all_transaction_ids:
            lexical_item = lexical_by_id.get(transaction_id)
            semantic_item = semantic_by_id.get(transaction_id)
            base_item = lexical_item or semantic_item
            
            # Compter les moteurs non-zéros
            non_zero_engines = 0
            score_sum = 0.0
            
            if lexical_item and lexical_item.score and lexical_item.score > 0:
                non_zero_engines += 1
                score_sum += self._normalize_lexical_score(lexical_item.score)
            
            if semantic_item and semantic_item.score and semantic_item.score > 0:
                non_zero_engines += 1
                score_sum += semantic_item.score
            
            # CombMNZ = CombSUM * nombre de moteurs
            combmnz_score = score_sum * non_zero_engines
            
            fused_item = self._create_fused_item(
                base_item, lexical_item, semantic_item, combmnz_score,
                "combmnz", {"non_zero_engines": non_zero_engines}
            )
            
            fused_results.append(fused_item)
        
        return fused_results
    
    def _normalize_lexical_score(self, score: float) -> float:
        """Normalise un score Elasticsearch (typiquement 1-20) vers 0-1."""
        if score is None:
            return 0.0
        # Elasticsearch scores sont généralement entre 1 et 20
        return min(score / 15.0, 1.0)
    
    def _min_max_normalize(self, scores: List[float]) -> List[float]:
        """Normalisation Min-Max."""
        if not scores or len(scores) == 1:
            return scores
        
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return [0.5] * len(scores)
        
        return [(s - min_score) / (max_score - min_score) for s in scores]
    
    def _z_score_normalize(self, scores: List[float]) -> List[float]:
        """Normalisation Z-Score."""
        if not scores or len(scores) == 1:
            return scores
        
        mean_score = sum(scores) / len(scores)
        variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
        std_dev = math.sqrt(variance)
        
        if std_dev == 0:
            return [0.5] * len(scores)
        
        z_scores = [(s - mean_score) / std_dev for s in scores]
        
        # Convertir en scores positifs entre 0-1 avec sigmoid
        return [1 / (1 + math.exp(-z)) for z in z_scores]
    
    def _sigmoid_normalize(self, scores: List[float]) -> List[float]:
        """Normalisation Sigmoid."""
        if not scores:
            return scores
        
        mean_score = sum(scores) / len(scores)
        return [1 / (1 + math.exp(-(s - mean_score))) for s in scores]
    
    def _create_fused_item(
        self,
        base_item: SearchResultItem,
        lexical_item: Optional[SearchResultItem],
        semantic_item: Optional[SearchResultItem],
        combined_score: float,
        fusion_method: str,
        fusion_metadata: Dict[str, Any]
    ) -> SearchResultItem:
        """Crée un SearchResultItem fusionné."""
        
        # Fusionner les métadonnées
        metadata = base_item.metadata.copy() if base_item.metadata else {}
        metadata.update({
            "fusion_method": fusion_method,
            "found_in_lexical": lexical_item is not None,
            "found_in_semantic": semantic_item is not None,
            **fusion_metadata
        })
        
        # Privilégier les highlights du lexical s'ils existent
        highlights = None
        if lexical_item and hasattr(lexical_item, 'highlights') and lexical_item.highlights:
            highlights = lexical_item.highlights
        elif semantic_item and hasattr(semantic_item, 'highlights') and semantic_item.highlights:
            highlights = semantic_item.highlights
        
        return SearchResultItem(
            transaction_id=base_item.transaction_id,
            user_id=base_item.user_id,
            account_id=base_item.account_id,
            score=combined_score,
            lexical_score=lexical_item.score if lexical_item else None,
            semantic_score=semantic_item.score if semantic_item else None,
            combined_score=combined_score,
            primary_description=base_item.primary_description,
            searchable_text=base_item.searchable_text,
            merchant_name=base_item.merchant_name,
            amount=base_item.amount,
            currency_code=base_item.currency_code,
            transaction_type=base_item.transaction_type,
            transaction_date=base_item.transaction_date,
            created_at=base_item.created_at,
            category_id=base_item.category_id,
            operation_type=base_item.operation_type,
            highlights=highlights,
            metadata=metadata,
            explanation=getattr(base_item, 'explanation', None)
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
            fused_results = self._deduplicate_results(fused_results, debug)
        
        # 2. Diversification
        if self.config.enable_diversification:
            fused_results = self._diversify_results(fused_results, debug)
        
        # 3. Boost de qualité pour certains résultats
        fused_results = self._apply_quality_boost(fused_results, query_analysis, debug)
        
        return fused_results
    
    def _deduplicate_results(
        self,
        results: List[SearchResultItem],
        debug: bool
    ) -> List[SearchResultItem]:
        """Déduplique les résultats basés sur la similarité."""
        if len(results) <= 1:
            return results
        
        deduplicated = []
        seen_signatures = set()
        
        for result in results:
            # Créer une signature pour la transaction
            signature = self._create_transaction_signature(result)
            
            # Vérifier la similarité avec les transactions déjà vues
            is_duplicate = False
            for seen_sig in seen_signatures:
                if self._calculate_signature_similarity(signature, seen_sig) > self.config.dedup_similarity_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                deduplicated.append(result)
                seen_signatures.add(signature)
            elif debug:
                # Marquer comme dupliqué dans les métadonnées pour debug
                if result.metadata:
                    result.metadata["duplicate_removed"] = True
        
        return deduplicated
    
    def _create_transaction_signature(self, result: SearchResultItem) -> str:
        """Crée une signature pour identifier les doublons."""
        # Utiliser plusieurs champs pour créer une signature
        components = [
            str(result.amount or 0),
            result.transaction_date or "",
            (result.merchant_name or "").lower()[:20],
            (result.primary_description or "").lower()[:30]
        ]
        
        return "|".join(components)
    
    def _calculate_signature_similarity(self, sig1: str, sig2: str) -> float:
        """Calcule la similarité entre deux signatures."""
        if sig1 == sig2:
            return 1.0
        
        # Similarité basée sur les composants
        parts1 = sig1.split("|")
        parts2 = sig2.split("|")
        
        if len(parts1) != len(parts2):
            return 0.0
        
        similarities = []
        for p1, p2 in zip(parts1, parts2):
            if p1 == p2:
                similarities.append(1.0)
            elif p1 and p2:
                # Similarité de chaînes simple
                longer = max(len(p1), len(p2))
                common = sum(c1 == c2 for c1, c2 in zip(p1, p2))
                similarities.append(common / longer)
            else:
                similarities.append(0.0)
        
        return sum(similarities) / len(similarities)
    
    def _diversify_results(
        self,
        results: List[SearchResultItem],
        debug: bool
    ) -> List[SearchResultItem]:
        """Diversifie les résultats pour éviter la redondance."""
        if len(results) <= self.config.max_same_merchant:
            return results
        
        diversified = []
        merchant_counts = {}
        
        # Trier par score décroissant d'abord
        sorted_results = sorted(results, key=lambda x: x.score or 0, reverse=True)
        
        for result in sorted_results:
            merchant = result.merchant_name or "unknown"
            current_count = merchant_counts.get(merchant, 0)
            
            # Ajouter si on n'a pas atteint la limite pour ce marchand
            if current_count < self.config.max_same_merchant:
                diversified.append(result)
                merchant_counts[merchant] = current_count + 1
            elif debug:
                # Marquer comme filtré pour diversité
                if result.metadata:
                    result.metadata["diversity_filtered"] = True
        
        return diversified
    
    def _apply_quality_boost(
        self,
        results: List[SearchResultItem],
        query_analysis: QueryAnalysis,
        debug: bool
    ) -> List[SearchResultItem]:
        """Applique un boost de qualité aux résultats pertinents."""
        
        for result in results:
            boost_factor = 1.0
            boost_reasons = []
            
            # CORRECTION: Utiliser la propriété key_terms correcte
            # Boost pour correspondance avec les termes de la requête
            if query_analysis.key_terms:
                text_content = " ".join(filter(None, [
                    result.primary_description,
                    result.merchant_name,
                    result.searchable_text
                ])).lower()
                
                matching_terms = sum(
                    1 for term in query_analysis.key_terms
                    if term.lower() in text_content
                )
                
                if matching_terms > 0:
                    term_boost = 1 + (matching_terms / len(query_analysis.key_terms)) * self.config.quality_boost_factor
                    boost_factor *= term_boost
                    boost_reasons.append(f"term_match_{matching_terms}")
            
            # Boost pour transactions récentes
            if result.transaction_date:
                # Simple boost pour les transactions récentes (implémentation basique)
                boost_factor *= 1.05
                boost_reasons.append("recent_transaction")
            
            # Boost pour montants ronds (souvent plus mémorables)
            if result.amount and result.amount == int(result.amount):
                boost_factor *= 1.02
                boost_reasons.append("round_amount")
            
            # Appliquer le boost
            if boost_factor > 1.0:
                original_score = result.score
                result.score *= boost_factor
                result.combined_score *= boost_factor
                
                if debug and result.metadata:
                    result.metadata.update({
                        "quality_boost_applied": boost_factor,
                        "boost_reasons": boost_reasons,
                        "original_score": original_score
                    })
        
        return results
    
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
    
    def _assess_fusion_quality(
        self,
        results: List[SearchResultItem],
        lexical_result: LexicalSearchResult,
        semantic_result: SemanticSearchResult,
        query_analysis: QueryAnalysis
    ) -> SearchQuality:
        """Évalue la qualité de la fusion."""
        if not results:
            return SearchQuality.POOR
        
        # Facteurs de qualité
        score_quality = self._assess_fused_score_quality(results)
        coverage_quality = self._assess_coverage_quality(results, lexical_result, semantic_result)
        consistency_quality = self._assess_fusion_consistency(results)
        relevance_quality = self._assess_fused_relevance(results, query_analysis)
        
        # Moyenne pondérée
        overall_quality = (
            score_quality * 0.3 +
            coverage_quality * 0.25 +
            consistency_quality * 0.25 +
            relevance_quality * 0.2
        )
        
        # Conversion en enum
        if overall_quality >= 0.8:
            return SearchQuality.EXCELLENT
        elif overall_quality >= 0.6:
            return SearchQuality.GOOD
        elif overall_quality >= 0.4:
            return SearchQuality.MEDIUM
        else:
            return SearchQuality.POOR
    
    def _assess_fused_score_quality(self, results: List[SearchResultItem]) -> float:
        """Évalue la qualité des scores fusionnés."""
        if not results:
            return 0.0
        
        scores = [r.score for r in results if r.score]
        if not scores:
            return 0.0
        
        # Vérifier la distribution des scores
        max_score = max(scores)
        min_score = min(scores)
        avg_score = sum(scores) / len(scores)
        
        # Qualité basée sur le score max et la distribution
        max_quality = min(max_score, 1.0)
        distribution_quality = (max_score - min_score) / max_score if max_score > 0 else 0
        avg_quality = avg_score
        
        return (max_quality * 0.4 + distribution_quality * 0.3 + avg_quality * 0.3)
    
    def _assess_coverage_quality(
        self,
        results: List[SearchResultItem],
        lexical_result: LexicalSearchResult,
        semantic_result: SemanticSearchResult
    ) -> float:
        """Évalue la couverture de la fusion."""
        if not results:
            return 0.0
        
        # Compter les résultats de chaque moteur
        both_engines = sum(
            1 for r in results 
            if r.metadata and r.metadata.get("found_in_lexical") and r.metadata.get("found_in_semantic")
        )
        
        lexical_only = sum(
            1 for r in results
            if r.metadata and r.metadata.get("found_in_lexical") and not r.metadata.get("found_in_semantic")
        )
        
        semantic_only = sum(
            1 for r in results
            if r.metadata and not r.metadata.get("found_in_lexical") and r.metadata.get("found_in_semantic")
        )
        
        total_results = len(results)
        
        # Qualité basée sur la diversité des sources
        if total_results == 0:
            return 0.0
        
        both_ratio = both_engines / total_results
        diversity_ratio = (lexical_only + semantic_only) / total_results
        
        # Bonus pour avoir des résultats des deux moteurs
        return both_ratio * 0.6 + diversity_ratio * 0.4
    
    def _assess_fusion_consistency(self, results: List[SearchResultItem]) -> float:
        """Évalue la cohérence de la fusion."""
        if len(results) <= 1:
            return 1.0
        
        # Vérifier la cohérence des scores
        scores = [r.score for r in results if r.score]
        if len(scores) < 2:
            return 0.5
        
        # Calculer la cohérence des écarts
        score_gaps = [scores[i] - scores[i+1] for i in range(len(scores)-1)]
        
        if not score_gaps:
            return 1.0
        
        avg_gap = sum(score_gaps) / len(score_gaps)
        max_gap = max(score_gaps)
        
        # Bonne cohérence = écarts réguliers
        if max_gap == 0:
            return 1.0
        
        consistency = 1 - abs(avg_gap - max_gap) / max_gap
        return max(0.0, consistency)
    
    def _assess_fused_relevance(
        self,
        results: List[SearchResultItem],
        query_analysis: QueryAnalysis
    ) -> float:
        """Évalue la pertinence des résultats fusionnés."""
        # CORRECTION: Utiliser la propriété key_terms correcte
        if not results or not query_analysis.key_terms:
            return 0.5
        
        relevance_scores = []
        
        for result in results:
            text_content = " ".join(filter(None, [
                result.primary_description,
                result.merchant_name,
                result.searchable_text
            ])).lower()
            
            # Compter les termes qui matchent
            matching_terms = sum(
                1 for term in query_analysis.key_terms
                if term.lower() in text_content
            )
            
            relevance = matching_terms / len(query_analysis.key_terms)
            relevance_scores.append(relevance)
        
        return sum(relevance_scores) / len(relevance_scores)
    
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
        import time
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
        logger.info("Result merger configuration updated")