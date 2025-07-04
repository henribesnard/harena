"""
Optimiseur de poids pour la fusion hybride.

Ce module détermine les poids optimaux pour combiner
les résultats lexicaux et sémantiques selon le contexte.
"""
from typing import Dict, Any
from search_service.core.lexical_engine import LexicalSearchResult
from search_service.core.semantic_engine import SemanticSearchResult
from search_service.core.query_processor import QueryAnalysis
from search_service.models.search_types import SearchQuality
from search_service.utils.fusion_strategies import FusionStrategy


class WeightOptimizer:
    """Optimiseur de poids pour la fusion hybride."""
    
    def __init__(
        self,
        default_lexical_weight: float = 0.6,
        default_semantic_weight: float = 0.4
    ):
        self.default_lexical_weight = default_lexical_weight
        self.default_semantic_weight = default_semantic_weight
    
    def determine_optimal_weights(
        self,
        lexical_result: LexicalSearchResult,
        semantic_result: SemanticSearchResult,
        query_analysis: QueryAnalysis
    ) -> Dict[str, float]:
        """
        Détermine les poids optimaux pour la fusion.
        
        Args:
            lexical_result: Résultats de recherche lexicale
            semantic_result: Résultats de recherche sémantique
            query_analysis: Analyse de la requête
            
        Returns:
            Dictionnaire des poids optimaux
        """
        base_lexical = self.default_lexical_weight
        base_semantic = self.default_semantic_weight
        
        # Ajustements basés sur la qualité des résultats
        quality_diff = self._quality_to_score(lexical_result.quality) - \
                      self._quality_to_score(semantic_result.quality)
        
        # Ajuster les poids selon la différence de qualité
        weight_adjustment = quality_diff * 0.2
        
        lexical_weight = max(0.1, min(0.9, base_lexical + weight_adjustment))
        semantic_weight = 1.0 - lexical_weight
        
        # Ajustements spécifiques selon le type de requête
        lexical_weight, semantic_weight = self._apply_query_specific_adjustments(
            lexical_weight, semantic_weight, query_analysis
        )
        
        # Normaliser pour assurer que la somme = 1.0
        total = lexical_weight + semantic_weight
        lexical_weight /= total
        semantic_weight /= total
        
        return {
            "lexical_weight": lexical_weight,
            "semantic_weight": semantic_weight
        }
    
    def determine_optimal_strategy(
        self,
        lexical_result: LexicalSearchResult,
        semantic_result: SemanticSearchResult,
        query_analysis: QueryAnalysis
    ) -> FusionStrategy:
        """
        Détermine la stratégie de fusion optimale.
        
        Args:
            lexical_result: Résultats de recherche lexicale
            semantic_result: Résultats de recherche sémantique
            query_analysis: Analyse de la requête
            
        Returns:
            Stratégie de fusion recommandée
        """
        # Analyser la qualité des résultats de chaque moteur
        lexical_quality = lexical_result.quality
        semantic_quality = semantic_result.quality
        
        # Si une recherche est de très mauvaise qualité
        if lexical_quality == SearchQuality.POOR and semantic_quality != SearchQuality.POOR:
            return FusionStrategy.WEIGHTED_AVERAGE  # Favoriser le sémantique
        
        if semantic_quality == SearchQuality.POOR and lexical_quality != SearchQuality.POOR:
            return FusionStrategy.WEIGHTED_AVERAGE  # Favoriser le lexical
        
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
    
    def _quality_to_score(self, quality: SearchQuality) -> float:
        """Convertit une qualité en score numérique."""
        quality_scores = {
            SearchQuality.EXCELLENT: 1.0,
            SearchQuality.GOOD: 0.7,
            SearchQuality.MEDIUM: 0.5,
            SearchQuality.POOR: 0.2
        }
        return quality_scores.get(quality, 0.5)
    
    def _apply_query_specific_adjustments(
        self,
        lexical_weight: float,
        semantic_weight: float,
        query_analysis: QueryAnalysis
    ) -> tuple[float, float]:
        """
        Applique des ajustements spécifiques au type de requête.
        
        Args:
            lexical_weight: Poids lexical initial
            semantic_weight: Poids sémantique initial
            query_analysis: Analyse de la requête
            
        Returns:
            Tuple des poids ajustés (lexical, semantic)
        """
        # Ajustements pour phrases exactes
        if query_analysis.has_exact_phrases:
            # Favoriser le lexical pour les phrases exactes
            lexical_weight += 0.1
            semantic_weight -= 0.1
        
        # Ajustements pour entités financières
        if query_analysis.has_financial_entities:
            # Équilibrer pour les entités financières
            lexical_weight = 0.5
            semantic_weight = 0.5
        
        # Ajustements pour requêtes complexes
        if len(query_analysis.key_terms) > 5:
            # Favoriser le sémantique pour les requêtes complexes
            semantic_weight += 0.1
            lexical_weight -= 0.1
        
        # Ajustements pour requêtes avec montants
        if query_analysis.has_amounts:
            # Favoriser légèrement le lexical pour les montants
            lexical_weight += 0.05
            semantic_weight -= 0.05
        
        # Ajustements pour requêtes avec dates
        if query_analysis.has_dates:
            # Favoriser légèrement le lexical pour les dates
            lexical_weight += 0.05
            semantic_weight -= 0.05
        
        # S'assurer que les poids restent dans des limites raisonnables
        lexical_weight = max(0.1, min(0.9, lexical_weight))
        semantic_weight = max(0.1, min(0.9, semantic_weight))
        
        return lexical_weight, semantic_weight


class AdaptiveWeightManager:
    """Gestionnaire de poids adaptatifs avec historique."""
    
    def __init__(self):
        self.weight_history = []
        self.performance_history = []
        self.learning_rate = 0.1
    
    def update_weights_from_feedback(
        self,
        current_weights: Dict[str, float],
        feedback_score: float,
        query_features: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Met à jour les poids basés sur le feedback de performance.
        
        Args:
            current_weights: Poids actuels utilisés
            feedback_score: Score de feedback (0-1, plus haut = meilleur)
            query_features: Caractéristiques de la requête
            
        Returns:
            Poids mis à jour
        """
        # Enregistrer l'historique
        self.weight_history.append(current_weights.copy())
        self.performance_history.append(feedback_score)
        
        # Si on n'a pas assez d'historique, retourner les poids actuels
        if len(self.weight_history) < 2:
            return current_weights
        
        # Calculer la tendance de performance
        recent_performance = self.performance_history[-5:]  # 5 dernières performances
        avg_performance = sum(recent_performance) / len(recent_performance)
        
        # Si la performance est bonne, garder les poids actuels
        if avg_performance >= 0.7:
            return current_weights
        
        # Sinon, ajuster graduellement
        adjusted_weights = current_weights.copy()
        
        # Logique simple : si performance faible, équilibrer plus
        if avg_performance < 0.5:
            lexical_weight = adjusted_weights["lexical_weight"]
            semantic_weight = adjusted_weights["semantic_weight"]
            
            # Rapprocher vers 0.5/0.5
            target_lexical = 0.5
            target_semantic = 0.5
            
            adjusted_weights["lexical_weight"] = (
                lexical_weight + (target_lexical - lexical_weight) * self.learning_rate
            )
            adjusted_weights["semantic_weight"] = (
                semantic_weight + (target_semantic - semantic_weight) * self.learning_rate
            )
        
        return adjusted_weights
    
    def get_weight_recommendations(
        self,
        query_features: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Recommande des poids basés sur l'historique et les caractéristiques de requête.
        
        Args:
            query_features: Caractéristiques de la requête
            
        Returns:
            Poids recommandés
        """
        # Implémentation basique - peut être étendue avec ML
        base_weights = {"lexical_weight": 0.6, "semantic_weight": 0.4}
        
        # Ajustements basés sur les caractéristiques
        if query_features.get("has_exact_phrases", False):
            base_weights["lexical_weight"] += 0.1
            base_weights["semantic_weight"] -= 0.1
        
        if query_features.get("query_length", 0) > 5:
            base_weights["semantic_weight"] += 0.1
            base_weights["lexical_weight"] -= 0.1
        
        # Normaliser
        total = sum(base_weights.values())
        for key in base_weights:
            base_weights[key] /= total
        
        return base_weights