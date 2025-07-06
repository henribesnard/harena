"""
Stratégies de fusion pour la recherche hybride - VERSION CENTRALISÉE.

Ce module contient toutes les implémentations des stratégies
de fusion des résultats lexicaux et sémantiques.

AMÉLIORATION:
- Configuration entièrement centralisée via config_service
- Plus de duplication de paramètres
- Contrôle total via .env
"""
import math
from typing import Dict, Any, List, Optional
from enum import Enum

# ✅ CONFIGURATION CENTRALISÉE
from config_service.config import settings
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


class ScoreNormalizer:
    """
    Normaliseur de scores pour différents moteurs.
    
    Classe simplifiée avec méthodes statiques robustes.
    """
    
    @staticmethod
    def normalize_lexical_score(score: float) -> float:
        """
        Normalise un score Elasticsearch (typiquement 1-20) vers 0-1.
        
        Args:
            score: Score brut d'Elasticsearch
            
        Returns:
            Score normalisé entre 0 et 1
        """
        if score is None or score <= 0:
            return 0.0
        # Elasticsearch donne typiquement des scores entre 1 et 15
        return min(score / 15.0, 1.0)
    
    @staticmethod
    def normalize_semantic_score(score: float) -> float:
        """
        Normalise un score sémantique (généralement déjà entre 0-1).
        
        Args:
            score: Score de similarité sémantique
            
        Returns:
            Score normalisé entre 0 et 1
        """
        if score is None or score < 0:
            return 0.0
        return min(score, 1.0)
    
    @staticmethod
    def min_max_normalize(scores: List[float]) -> List[float]:
        """
        Normalisation Min-Max robuste.
        
        Args:
            scores: Liste des scores à normaliser
            
        Returns:
            Scores normalisés entre 0 et 1
        """
        if not scores:
            return []
        
        if len(scores) == 1:
            return [1.0]
        
        valid_scores = [s for s in scores if s is not None and not math.isnan(s)]
        if not valid_scores:
            return [0.0] * len(scores)
        
        min_score = min(valid_scores)
        max_score = max(valid_scores)
        
        if max_score == min_score:
            return [0.5] * len(scores)
        
        normalized = []
        for score in scores:
            if score is None or math.isnan(score):
                normalized.append(0.0)
            else:
                normalized.append((score - min_score) / (max_score - min_score))
        
        return normalized
    
    @staticmethod
    def z_score_normalize(scores: List[float]) -> List[float]:
        """
        Normalisation Z-Score avec conversion sigmoid.
        
        Args:
            scores: Liste des scores à normaliser
            
        Returns:
            Scores normalisés entre 0 et 1
        """
        if not scores:
            return []
        
        if len(scores) == 1:
            return [0.5]
        
        valid_scores = [s for s in scores if s is not None and not math.isnan(s)]
        if not valid_scores:
            return [0.0] * len(scores)
        
        mean_score = sum(valid_scores) / len(valid_scores)
        variance = sum((s - mean_score) ** 2 for s in valid_scores) / len(valid_scores)
        std_dev = math.sqrt(variance)
        
        if std_dev == 0:
            return [0.5] * len(scores)
        
        normalized = []
        for score in scores:
            if score is None or math.isnan(score):
                normalized.append(0.0)
            else:
                z_score = (score - mean_score) / std_dev
                # Convertir en probabilité avec sigmoid
                sigmoid_score = 1 / (1 + math.exp(-z_score))
                normalized.append(sigmoid_score)
        
        return normalized
    
    @staticmethod
    def sigmoid_normalize(scores: List[float]) -> List[float]:
        """
        Normalisation Sigmoid directe.
        
        Args:
            scores: Liste des scores à normaliser
            
        Returns:
            Scores normalisés entre 0 et 1
        """
        if not scores:
            return []
        
        valid_scores = [s for s in scores if s is not None and not math.isnan(s)]
        if not valid_scores:
            return [0.0] * len(scores)
        
        mean_score = sum(valid_scores) / len(valid_scores)
        
        normalized = []
        for score in scores:
            if score is None or math.isnan(score):
                normalized.append(0.0)
            else:
                sigmoid_score = 1 / (1 + math.exp(-(score - mean_score)))
                normalized.append(sigmoid_score)
        
        return normalized


class FusionStrategyExecutor:
    """
    Exécuteur des stratégies de fusion.
    
    Version simplifiée utilisant la configuration centralisée.
    """
    
    def __init__(self):
        """Initialise l'exécuteur avec la configuration centralisée."""
        self.normalizer = ScoreNormalizer()
    
    def execute(
        self,
        strategy: FusionStrategy,
        lexical_results: List[SearchResultItem],
        semantic_results: List[SearchResultItem],
        weights: Optional[Dict[str, float]] = None,
        debug: bool = False
    ) -> List[SearchResultItem]:
        """
        Exécute la stratégie de fusion spécifiée.
        
        Args:
            strategy: Stratégie de fusion à utiliser
            lexical_results: Résultats de la recherche lexicale
            semantic_results: Résultats de la recherche sémantique
            weights: Poids personnalisés (utilise config centralisée si None)
            debug: Mode debug pour informations supplémentaires
            
        Returns:
            Liste des résultats fusionnés et triés
        """
        # ✅ Utiliser les poids de la configuration centralisée
        fusion_weights = weights or {
            "lexical_weight": settings.DEFAULT_LEXICAL_WEIGHT,
            "semantic_weight": settings.DEFAULT_SEMANTIC_WEIGHT
        }
        
        try:
            # Indexer les résultats par transaction_id pour faciliter la fusion
            lexical_by_id = {r.transaction_id: r for r in lexical_results if r.transaction_id}
            semantic_by_id = {r.transaction_id: r for r in semantic_results if r.transaction_id}
            
            # Exécuter la stratégie appropriée
            if strategy == FusionStrategy.WEIGHTED_AVERAGE:
                return self._fuse_weighted_average(lexical_by_id, semantic_by_id, fusion_weights)
            
            elif strategy == FusionStrategy.RANK_FUSION:
                return self._fuse_rank_fusion(lexical_results, semantic_results, fusion_weights)
            
            elif strategy == FusionStrategy.RECIPROCAL_RANK_FUSION:
                return self._fuse_reciprocal_rank_fusion(lexical_results, semantic_results, fusion_weights)
            
            elif strategy == FusionStrategy.SCORE_NORMALIZATION:
                return self._fuse_score_normalization(lexical_by_id, semantic_by_id, fusion_weights)
            
            elif strategy == FusionStrategy.ADAPTIVE_FUSION:
                return self._fuse_adaptive(lexical_by_id, semantic_by_id, fusion_weights)
            
            elif strategy == FusionStrategy.COMBSUM:
                return self._fuse_combsum(lexical_by_id, semantic_by_id, fusion_weights)
            
            else:
                # Fallback vers weighted average pour toutes les autres stratégies
                return self._fuse_weighted_average(lexical_by_id, semantic_by_id, fusion_weights)
                
        except Exception as e:
            # En cas d'erreur, retourner une fusion simple
            return self._simple_fallback_fusion(lexical_results, semantic_results, fusion_weights)
    
    def _create_fused_item(
        self,
        base_item: SearchResultItem,
        lexical_item: Optional[SearchResultItem],
        semantic_item: Optional[SearchResultItem],
        final_score: float,
        fusion_method: str
    ) -> SearchResultItem:
        """
        Crée un item fusionné à partir des résultats sources.
        
        Args:
            base_item: Item de base (avec le plus d'informations)
            lexical_item: Item lexical (peut être None)
            semantic_item: Item sémantique (peut être None)
            final_score: Score final calculé
            fusion_method: Méthode de fusion utilisée
            
        Returns:
            Nouvel item avec le score fusionné
        """
        # Créer une copie de l'item de base
        fused_item = SearchResultItem(
            transaction_id=base_item.transaction_id,
            description=base_item.description,
            amount=base_item.amount,
            date=base_item.date,
            merchant=base_item.merchant,
            category=base_item.category,
            score=final_score,
            highlights=base_item.highlights or [],
            metadata=base_item.metadata or {}
        )
        
        # Ajouter des métadonnées de fusion
        fused_item.metadata = fused_item.metadata.copy()
        fused_item.metadata.update({
            "fusion_method": fusion_method,
            "has_lexical": lexical_item is not None,
            "has_semantic": semantic_item is not None,
            "lexical_score": lexical_item.score if lexical_item else None,
            "semantic_score": semantic_item.score if semantic_item else None,
            "config_source": "centralized"
        })
        
        return fused_item
    
    def _fuse_weighted_average(
        self,
        lexical_by_id: Dict[int, SearchResultItem],
        semantic_by_id: Dict[int, SearchResultItem],
        weights: Dict[str, float]
    ) -> List[SearchResultItem]:
        """Fusion par moyenne pondérée des scores."""
        fused_results = []
        all_transaction_ids = set(lexical_by_id.keys()) | set(semantic_by_id.keys())
        
        lexical_weight = weights.get("lexical_weight", settings.DEFAULT_LEXICAL_WEIGHT)
        semantic_weight = weights.get("semantic_weight", settings.DEFAULT_SEMANTIC_WEIGHT)
        
        for transaction_id in all_transaction_ids:
            lexical_item = lexical_by_id.get(transaction_id)
            semantic_item = semantic_by_id.get(transaction_id)
            
            # Utiliser l'item avec le plus d'informations comme base
            base_item = lexical_item or semantic_item
            
            # Normaliser les scores
            lexical_score = self.normalizer.normalize_lexical_score(
                lexical_item.score if lexical_item else 0.0
            )
            semantic_score = self.normalizer.normalize_semantic_score(
                semantic_item.score if semantic_item else 0.0
            )
            
            # Calculer le score combiné
            combined_score = lexical_score * lexical_weight + semantic_score * semantic_weight
            
            # Créer l'item fusionné
            fused_item = self._create_fused_item(
                base_item, lexical_item, semantic_item, combined_score, "weighted_average"
            )
            
            fused_results.append(fused_item)
        
        # Trier par score décroissant et limiter selon la config
        fused_results.sort(key=lambda x: x.score, reverse=True)
        return fused_results[:settings.MAX_SEARCH_LIMIT]
    
    def _fuse_rank_fusion(
        self,
        lexical_results: List[SearchResultItem],
        semantic_results: List[SearchResultItem],
        weights: Dict[str, float]
    ) -> List[SearchResultItem]:
        """Fusion basée sur les rangs dans chaque liste."""
        lexical_weight = weights.get("lexical_weight", settings.DEFAULT_LEXICAL_WEIGHT)
        semantic_weight = weights.get("semantic_weight", settings.DEFAULT_SEMANTIC_WEIGHT)
        
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
            lexical_rank_score = 1.0 / lexical_rank if lexical_rank > 0 else 0.0
            semantic_rank_score = 1.0 / semantic_rank if semantic_rank > 0 else 0.0
            
            combined_rank_score = (
                lexical_rank_score * lexical_weight + 
                semantic_rank_score * semantic_weight
            )
            
            fused_item = self._create_fused_item(
                base_item, lexical_item, semantic_item, combined_rank_score, "rank_fusion"
            )
            
            fused_results.append(fused_item)
        
        # Trier par score décroissant
        fused_results.sort(key=lambda x: x.score, reverse=True)
        return fused_results[:settings.MAX_SEARCH_LIMIT]
    
    def _fuse_reciprocal_rank_fusion(
        self,
        lexical_results: List[SearchResultItem],
        semantic_results: List[SearchResultItem],
        weights: Dict[str, float]
    ) -> List[SearchResultItem]:
        """Fusion RRF (Reciprocal Rank Fusion)."""
        # ✅ Utiliser la configuration centralisée
        k = settings.RRF_K
        
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
            rrf_score = 0.0
            if lexical_item:
                lexical_rank = lexical_ranks[transaction_id]
                rrf_score += 1.0 / (k + lexical_rank)
            if semantic_item:
                semantic_rank = semantic_ranks[transaction_id]
                rrf_score += 1.0 / (k + semantic_rank)
            
            fused_item = self._create_fused_item(
                base_item, lexical_item, semantic_item, rrf_score, "reciprocal_rank_fusion"
            )
            
            fused_results.append(fused_item)
        
        # Trier par score décroissant
        fused_results.sort(key=lambda x: x.score, reverse=True)
        return fused_results[:settings.MAX_SEARCH_LIMIT]
    
    def _fuse_score_normalization(
        self,
        lexical_by_id: Dict[int, SearchResultItem],
        semantic_by_id: Dict[int, SearchResultItem],
        weights: Dict[str, float]
    ) -> List[SearchResultItem]:
        """Fusion avec normalisation avancée des scores."""
        # Extraire tous les scores pour normalisation
        lexical_scores = [item.score for item in lexical_by_id.values() if item.score is not None]
        semantic_scores = [item.score for item in semantic_by_id.values() if item.score is not None]
        
        # ✅ Utiliser la méthode de normalisation de la config centralisée
        normalization_method = settings.SCORE_NORMALIZATION_METHOD
        
        if normalization_method == "z_score":
            lexical_norm = self.normalizer.z_score_normalize(lexical_scores)
            semantic_norm = self.normalizer.z_score_normalize(semantic_scores)
        elif normalization_method == "sigmoid":
            lexical_norm = self.normalizer.sigmoid_normalize(lexical_scores)
            semantic_norm = self.normalizer.sigmoid_normalize(semantic_scores)
        else:  # min_max par défaut
            lexical_norm = self.normalizer.min_max_normalize(lexical_scores)
            semantic_norm = self.normalizer.min_max_normalize(semantic_scores)
        
        # Créer des mappings score -> score normalisé
        lexical_score_map = dict(zip(lexical_scores, lexical_norm)) if lexical_scores else {}
        semantic_score_map = dict(zip(semantic_scores, semantic_norm)) if semantic_scores else {}
        
        all_transaction_ids = set(lexical_by_id.keys()) | set(semantic_by_id.keys())
        fused_results = []
        
        lexical_weight = weights.get("lexical_weight", settings.DEFAULT_LEXICAL_WEIGHT)
        semantic_weight = weights.get("semantic_weight", settings.DEFAULT_SEMANTIC_WEIGHT)
        
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
                base_item, lexical_item, semantic_item, combined_score, "score_normalization"
            )
            
            fused_results.append(fused_item)
        
        # Trier par score décroissant
        fused_results.sort(key=lambda x: x.score, reverse=True)
        return fused_results[:settings.MAX_SEARCH_LIMIT]
    
    def _fuse_adaptive(
        self,
        lexical_by_id: Dict[int, SearchResultItem],
        semantic_by_id: Dict[int, SearchResultItem],
        weights: Dict[str, float]
    ) -> List[SearchResultItem]:
        """Fusion adaptative qui choisit la meilleure méthode par transaction."""
        fused_results = []
        all_transaction_ids = set(lexical_by_id.keys()) | set(semantic_by_id.keys())
        
        lexical_weight = weights.get("lexical_weight", settings.DEFAULT_LEXICAL_WEIGHT)
        semantic_weight = weights.get("semantic_weight", settings.DEFAULT_SEMANTIC_WEIGHT)
        
        # ✅ Utiliser les seuils de la configuration centralisée
        adaptive_threshold = settings.ADAPTIVE_THRESHOLD
        quality_boost_factor = settings.QUALITY_BOOST_FACTOR
        
        for transaction_id in all_transaction_ids:
            lexical_item = lexical_by_id.get(transaction_id)
            semantic_item = semantic_by_id.get(transaction_id)
            base_item = lexical_item or semantic_item
            
            # Déterminer la méthode optimale pour cette transaction
            if lexical_item and semantic_item:
                # Les deux moteurs ont trouvé cette transaction
                norm_lexical = self.normalizer.normalize_lexical_score(lexical_item.score)
                norm_semantic = self.normalizer.normalize_semantic_score(semantic_item.score)
                
                score_diff = abs(norm_lexical - norm_semantic)
                
                if score_diff < adaptive_threshold:
                    # Scores similaires -> moyenne pondérée
                    combined_score = norm_lexical * lexical_weight + norm_semantic * semantic_weight
                else:
                    # Scores différents -> favoriser le meilleur
                    if norm_lexical > norm_semantic:
                        combined_score = norm_lexical * (1 + quality_boost_factor)
                    else:
                        combined_score = norm_semantic * (1 + quality_boost_factor)
            
            elif lexical_item:
                # Seulement lexical
                combined_score = self.normalizer.normalize_lexical_score(lexical_item.score)
            
            else:
                # Seulement sémantique
                combined_score = self.normalizer.normalize_semantic_score(semantic_item.score)
            
            # S'assurer que le score reste dans [0, 1]
            combined_score = min(combined_score, 1.0)
            
            fused_item = self._create_fused_item(
                base_item, lexical_item, semantic_item, combined_score, "adaptive_fusion"
            )
            
            fused_results.append(fused_item)
        
        # Trier par score décroissant
        fused_results.sort(key=lambda x: x.score, reverse=True)
        return fused_results[:settings.MAX_SEARCH_LIMIT]
    
    def _fuse_combsum(
        self,
        lexical_by_id: Dict[int, SearchResultItem],
        semantic_by_id: Dict[int, SearchResultItem],
        weights: Dict[str, float]
    ) -> List[SearchResultItem]:
        """Fusion CombSUM (somme des scores normalisés)."""
        # CombSUM = somme simple sans pondération
        return self._fuse_weighted_average(lexical_by_id, semantic_by_id, {
            "lexical_weight": 0.5, "semantic_weight": 0.5
        })
    
    def _simple_fallback_fusion(
        self,
        lexical_results: List[SearchResultItem],
        semantic_results: List[SearchResultItem],
        weights: Dict[str, float]
    ) -> List[SearchResultItem]:
        """
        Fusion de fallback ultra-simple en cas d'erreur.
        
        Combine simplement les deux listes et trie par score.
        """
        all_results = []
        
        # Ajouter les résultats lexicaux avec pondération
        lexical_weight = weights.get("lexical_weight", settings.DEFAULT_LEXICAL_WEIGHT)
        for item in lexical_results[:50]:  # Limiter pour éviter les surcharges
            weighted_item = SearchResultItem(
                transaction_id=item.transaction_id,
                description=item.description,
                amount=item.amount,
                date=item.date,
                merchant=item.merchant,
                category=item.category,
                score=item.score * lexical_weight if item.score else 0.0,
                highlights=item.highlights,
                metadata=item.metadata
            )
            all_results.append(weighted_item)
        
        # Ajouter les résultats sémantiques avec pondération
        semantic_weight = weights.get("semantic_weight", settings.DEFAULT_SEMANTIC_WEIGHT)
        semantic_ids = {item.transaction_id for item in all_results}
        
        for item in semantic_results[:50]:  # Limiter pour éviter les surcharges
            if item.transaction_id not in semantic_ids:  # Éviter les doublons
                weighted_item = SearchResultItem(
                    transaction_id=item.transaction_id,
                    description=item.description,
                    amount=item.amount,
                    date=item.date,
                    merchant=item.merchant,
                    category=item.category,
                    score=item.score * semantic_weight if item.score else 0.0,
                    highlights=item.highlights,
                    metadata=item.metadata
                )
                all_results.append(weighted_item)
        
        # Trier par score et limiter selon la configuration centralisée
        all_results.sort(key=lambda x: x.score or 0.0, reverse=True)
        return all_results[:settings.MAX_SEARCH_LIMIT]


# ==========================================
# 🔧 FONCTIONS UTILITAIRES SIMPLIFIÉES
# ==========================================

def create_simple_executor() -> FusionStrategyExecutor:
    """Crée un exécuteur simple utilisant la configuration centralisée."""
    return FusionStrategyExecutor()


def create_default_executor() -> FusionStrategyExecutor:
    """Crée un exécuteur par défaut avec configuration centralisée."""
    return FusionStrategyExecutor()


# ==========================================
# 🎯 CLASSE DE CONFIGURATION SIMPLIFIÉE
# ==========================================

class FusionConfig:
    """
    Configuration de fusion utilisant les paramètres centralisés.
    
    Cette classe sert de facade pour accéder à la configuration
    centralisée de manière compatible avec l'ancien code.
    """
    
    @property
    def default_strategy(self) -> FusionStrategy:
        """Stratégie par défaut depuis la config centralisée."""
        strategy_mapping = {
            "weighted_average": FusionStrategy.WEIGHTED_AVERAGE,
            "rank_fusion": FusionStrategy.RANK_FUSION,
            "reciprocal_rank_fusion": FusionStrategy.RECIPROCAL_RANK_FUSION,
            "score_normalization": FusionStrategy.SCORE_NORMALIZATION,
            "adaptive_fusion": FusionStrategy.ADAPTIVE_FUSION,
            "combsum": FusionStrategy.COMBSUM,
            "combmnz": FusionStrategy.COMBMNZ
        }
        return strategy_mapping.get(settings.DEFAULT_SEARCH_TYPE, FusionStrategy.ADAPTIVE_FUSION)
    
    @property
    def default_lexical_weight(self) -> float:
        """Poids lexical par défaut."""
        return settings.DEFAULT_LEXICAL_WEIGHT
    
    @property
    def default_semantic_weight(self) -> float:
        """Poids sémantique par défaut."""
        return settings.DEFAULT_SEMANTIC_WEIGHT
    
    @property
    def score_normalization_method(self) -> str:
        """Méthode de normalisation des scores."""
        return settings.SCORE_NORMALIZATION_METHOD
    
    @property
    def min_score_threshold(self) -> float:
        """Seuil minimum de score."""
        return settings.MIN_SCORE_THRESHOLD
    
    @property
    def rrf_k(self) -> int:
        """Paramètre K pour RRF."""
        return settings.RRF_K
    
    @property
    def adaptive_threshold(self) -> float:
        """Seuil pour la fusion adaptative."""
        return settings.ADAPTIVE_THRESHOLD
    
    @property
    def quality_boost_factor(self) -> float:
        """Facteur de boost de qualité."""
        return settings.QUALITY_BOOST_FACTOR
    
    @property
    def enable_deduplication(self) -> bool:
        """Déduplication activée."""
        return settings.ENABLE_DEDUPLICATION
    
    @property
    def dedup_similarity_threshold(self) -> float:
        """Seuil de similarité pour déduplication."""
        return settings.DEDUP_SIMILARITY_THRESHOLD
    
    @property
    def enable_diversification(self) -> bool:
        """Diversification activée."""
        return settings.ENABLE_DIVERSIFICATION
    
    @property
    def diversity_factor(self) -> float:
        """Facteur de diversité."""
        return settings.DIVERSITY_FACTOR
    
    @property
    def max_same_merchant(self) -> int:
        """Maximum de résultats par marchand."""
        return settings.MAX_SAME_MERCHANT
    
    @property
    def max_results(self) -> int:
        """Nombre maximum de résultats."""
        return settings.MAX_SEARCH_LIMIT