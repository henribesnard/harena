"""
Stratégies de fusion pour la recherche hybride.

Ce module contient toutes les implémentations des stratégies
de fusion des résultats lexicaux et sémantiques.
"""
import math
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

from search_service.models.responses import SearchResultItem


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


class ScoreNormalizer:
    """Normaliseur de scores pour différents moteurs."""
    
    @staticmethod
    def normalize_lexical_score(score: float) -> float:
        """Normalise un score Elasticsearch (typiquement 1-20) vers 0-1."""
        if score is None:
            return 0.0
        return min(score / 15.0, 1.0)
    
    @staticmethod
    def min_max_normalize(scores: List[float]) -> List[float]:
        """Normalisation Min-Max."""
        if not scores or len(scores) == 1:
            return scores
        
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return [0.5] * len(scores)
        
        return [(s - min_score) / (max_score - min_score) for s in scores]
    
    @staticmethod
    def z_score_normalize(scores: List[float]) -> List[float]:
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
    
    @staticmethod
    def sigmoid_normalize(scores: List[float]) -> List[float]:
        """Normalisation Sigmoid."""
        if not scores:
            return scores
        
        mean_score = sum(scores) / len(scores)
        return [1 / (1 + math.exp(-(s - mean_score))) for s in scores]


class FusionStrategyExecutor:
    """Exécuteur des stratégies de fusion."""
    
    def __init__(self, config: FusionConfig):
        self.config = config
        self.normalizer = ScoreNormalizer()
    
    def execute(
        self,
        strategy: FusionStrategy,
        lexical_results: List[SearchResultItem],
        semantic_results: List[SearchResultItem],
        weights: Dict[str, float],
        debug: bool = False
    ) -> List[SearchResultItem]:
        """Exécute la stratégie de fusion spécifiée."""
        
        # Indexer les résultats par transaction_id
        lexical_by_id = {r.transaction_id: r for r in lexical_results}
        semantic_by_id = {r.transaction_id: r for r in semantic_results}
        
        if strategy == FusionStrategy.WEIGHTED_AVERAGE:
            return self._fuse_weighted_average(lexical_by_id, semantic_by_id, weights, debug)
        
        elif strategy == FusionStrategy.RANK_FUSION:
            return self._fuse_rank_fusion(lexical_results, semantic_results, weights, debug)
        
        elif strategy == FusionStrategy.RECIPROCAL_RANK_FUSION:
            return self._fuse_reciprocal_rank_fusion(lexical_results, semantic_results, weights, debug)
        
        elif strategy == FusionStrategy.SCORE_NORMALIZATION:
            return self._fuse_score_normalization(lexical_by_id, semantic_by_id, weights, debug)
        
        elif strategy == FusionStrategy.ADAPTIVE_FUSION:
            return self._fuse_adaptive(lexical_by_id, semantic_by_id, weights, debug)
        
        elif strategy == FusionStrategy.BORDA_COUNT:
            return self._fuse_borda_count(lexical_results, semantic_results, weights, debug)
        
        elif strategy == FusionStrategy.COMBSUM:
            return self._fuse_combsum(lexical_by_id, semantic_by_id, weights, debug)
        
        elif strategy == FusionStrategy.COMBMNZ:
            return self._fuse_combmnz(lexical_by_id, semantic_by_id, weights, debug)
        
        else:
            # Fallback vers weighted average
            return self._fuse_weighted_average(lexical_by_id, semantic_by_id, weights, debug)
    
    def _fuse_weighted_average(
        self,
        lexical_by_id: Dict[int, SearchResultItem],
        semantic_by_id: Dict[int, SearchResultItem],
        weights: Dict[str, float],
        debug: bool
    ) -> List[SearchResultItem]:
        """Fusion par moyenne pondérée des scores."""
        from .fusion_utils import create_fused_item
        
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
            lexical_score = self.normalizer.normalize_lexical_score(
                lexical_item.score if lexical_item else 0.0
            )
            semantic_score = semantic_item.score if semantic_item else 0.0
            
            # Calculer le score combiné
            combined_score = (
                lexical_score * lexical_weight + 
                semantic_score * semantic_weight
            )
            
            # Créer l'item fusionné
            fused_item = create_fused_item(
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
        from .fusion_utils import create_fused_item
        
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
            
            fused_item = create_fused_item(
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
        from .fusion_utils import create_fused_item
        
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
            if semantic_item:
                rrf_score += 1.0 / (k + semantic_rank)
            
            fused_item = create_fused_item(
                base_item, lexical_item, semantic_item, rrf_score,
                "reciprocal_rank_fusion", {
                    "rrf_k": k,
                    "lexical_rank": lexical_rank,
                    "semantic_rank": semantic_rank
                }
            )
            
            fused_results.append(fused_item)
        
        return fused_results
    
    def _fuse_score_normalization(
        self,
        lexical_by_id: Dict[int, SearchResultItem],
        semantic_by_id: Dict[int, SearchResultItem],
        weights: Dict[str, float],
        debug: bool
    ) -> List[SearchResultItem]:
        """Fusion avec normalisation avancée des scores."""
        from .fusion_utils import create_fused_item
        
        # Extraire tous les scores pour normalisation
        lexical_scores = [item.score for item in lexical_by_id.values() if item.score]
        semantic_scores = [item.score for item in semantic_by_id.values() if item.score]
        
        # Normaliser les scores selon la méthode configurée
        if self.config.score_normalization_method == "min_max":
            lexical_norm = self.normalizer.min_max_normalize(lexical_scores)
            semantic_norm = self.normalizer.min_max_normalize(semantic_scores)
        elif self.config.score_normalization_method == "z_score":
            lexical_norm = self.normalizer.z_score_normalize(lexical_scores)
            semantic_norm = self.normalizer.z_score_normalize(semantic_scores)
        else:  # sigmoid
            lexical_norm = self.normalizer.sigmoid_normalize(lexical_scores)
            semantic_norm = self.normalizer.sigmoid_normalize(semantic_scores)
        
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
            
            fused_item = create_fused_item(
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
        lexical_by_id: Dict[int, SearchResultItem],
        semantic_by_id: Dict[int, SearchResultItem],
        weights: Dict[str, float],
        debug: bool
    ) -> List[SearchResultItem]:
        """Fusion adaptative qui choisit la meilleure méthode par transaction."""
        from .fusion_utils import create_fused_item
        
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
                    self.normalizer.normalize_lexical_score(lexical_item.score) - semantic_item.score
                )
                
                if score_diff < self.config.adaptive_threshold:
                    # Scores similaires -> moyenne pondérée
                    combined_score = (
                        self.normalizer.normalize_lexical_score(lexical_item.score) * weights["lexical_weight"] +
                        semantic_item.score * weights["semantic_weight"]
                    )
                    method = "weighted_average"
                else:
                    # Scores différents -> favoriser le meilleur
                    if self.normalizer.normalize_lexical_score(lexical_item.score) > semantic_item.score:
                        combined_score = self.normalizer.normalize_lexical_score(lexical_item.score) * 1.1
                        method = "lexical_boosted"
                    else:
                        combined_score = semantic_item.score * 1.1
                        method = "semantic_boosted"
            
            elif lexical_item:
                # Seulement lexical
                combined_score = self.normalizer.normalize_lexical_score(lexical_item.score)
                method = "lexical_only"
            
            else:
                # Seulement sémantique
                combined_score = semantic_item.score
                method = "semantic_only"
            
            fused_item = create_fused_item(
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
        from .fusion_utils import create_fused_item
        
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
            
            fused_item = create_fused_item(
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
        from .fusion_utils import create_fused_item
        
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
                score_sum += self.normalizer.normalize_lexical_score(lexical_item.score)
            
            if semantic_item and semantic_item.score and semantic_item.score > 0:
                non_zero_engines += 1
                score_sum += semantic_item.score
            
            # CombMNZ = CombSUM * nombre de moteurs
            combmnz_score = score_sum * non_zero_engines
            
            fused_item = create_fused_item(
                base_item, lexical_item, semantic_item, combmnz_score,
                "combmnz", {"non_zero_engines": non_zero_engines}
            )
            
            fused_results.append(fused_item)
        
        return fused_results