"""
Optimiseur de poids pour la fusion hybride - VERSION CENTRALISÉE.

Ce module détermine les poids optimaux pour combiner
les résultats lexicaux et sémantiques selon le contexte.

AMÉLIORATION:
- Utilise la configuration centralisée pour les valeurs par défaut
- Plus de valeurs hardcodées
"""
from typing import Dict, Any, Optional
from search_service.core.lexical_engine import LexicalSearchResult
from search_service.core.semantic_engine import SemanticSearchResult
from search_service.core.query_processor import QueryAnalysis
from search_service.models.search_types import SearchQuality
from search_service.utils.fusion_strategies import FusionStrategy

# ✅ CONFIGURATION CENTRALISÉE
from config_service.config import settings


class WeightOptimizer:
    """Optimiseur de poids pour la fusion hybride."""
    
    def __init__(
        self,
        default_lexical_weight: Optional[float] = None,
        default_semantic_weight: Optional[float] = None
    ):
        """
        Initialise l'optimiseur avec la configuration centralisée.
        
        Args:
            default_lexical_weight: Poids lexical (utilise config si None)
            default_semantic_weight: Poids sémantique (utilise config si None)
        """
        # ✅ Utiliser la configuration centralisée par défaut
        self.default_lexical_weight = default_lexical_weight or settings.DEFAULT_LEXICAL_WEIGHT
        self.default_semantic_weight = default_semantic_weight or settings.DEFAULT_SEMANTIC_WEIGHT
    
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
        # ✅ Utiliser les seuils de qualité de la configuration centralisée
        quality_scores = {
            SearchQuality.EXCELLENT: settings.QUALITY_EXCELLENT_THRESHOLD,
            SearchQuality.GOOD: settings.QUALITY_GOOD_THRESHOLD,
            SearchQuality.MEDIUM: settings.QUALITY_MEDIUM_THRESHOLD,
            SearchQuality.POOR: settings.QUALITY_POOR_THRESHOLD
        }
        return quality_scores.get(quality, settings.QUALITY_MEDIUM_THRESHOLD)
    
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
        # ✅ Paramètres configurables pourraient être ajoutés à settings plus tard
        self.learning_rate = 0.1
        self.min_history_for_adaptation = 2
        self.good_performance_threshold = 0.7
        self.poor_performance_threshold = 0.5
    
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
        if len(self.weight_history) < self.min_history_for_adaptation:
            return current_weights
        
        # Calculer la tendance de performance
        recent_performance = self.performance_history[-5:]  # 5 dernières performances
        avg_performance = sum(recent_performance) / len(recent_performance)
        
        # Si la performance est bonne, garder les poids actuels
        if avg_performance >= self.good_performance_threshold:
            return current_weights
        
        # Sinon, ajuster graduellement
        adjusted_weights = current_weights.copy()
        
        # Logique simple : si performance faible, équilibrer plus
        if avg_performance < self.poor_performance_threshold:
            lexical_weight = adjusted_weights["lexical_weight"]
            semantic_weight = adjusted_weights["semantic_weight"]
            
            # ✅ Rapprocher vers les valeurs par défaut de la configuration centralisée
            target_lexical = settings.DEFAULT_LEXICAL_WEIGHT
            target_semantic = settings.DEFAULT_SEMANTIC_WEIGHT
            
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
        # ✅ Utiliser la configuration centralisée comme base
        base_weights = {
            "lexical_weight": settings.DEFAULT_LEXICAL_WEIGHT,
            "semantic_weight": settings.DEFAULT_SEMANTIC_WEIGHT
        }
        
        # Ajustements basés sur les caractéristiques
        if query_features.get("has_exact_phrases", False):
            base_weights["lexical_weight"] += 0.1
            base_weights["semantic_weight"] -= 0.1
        
        if query_features.get("query_length", 0) > 5:
            base_weights["semantic_weight"] += 0.1
            base_weights["lexical_weight"] -= 0.1
        
        if query_features.get("has_financial_entities", False):
            # Équilibrer pour les entités financières
            base_weights["lexical_weight"] = 0.5
            base_weights["semantic_weight"] = 0.5
        
        if query_features.get("has_amounts", False):
            # Légèrement favoriser le lexical pour les montants
            base_weights["lexical_weight"] += 0.05
            base_weights["semantic_weight"] -= 0.05
        
        # Normaliser
        total = sum(base_weights.values())
        for key in base_weights:
            base_weights[key] /= total
        
        return base_weights
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Retourne un résumé des performances."""
        if not self.performance_history:
            return {
                "total_feedback_count": 0,
                "average_performance": 0.0,
                "recent_performance": 0.0,
                "performance_trend": "unknown"
            }
        
        total_count = len(self.performance_history)
        avg_performance = sum(self.performance_history) / total_count
        
        # Performance récente (5 derniers)
        recent_scores = self.performance_history[-5:]
        recent_avg = sum(recent_scores) / len(recent_scores)
        
        # Tendance
        if total_count >= 3:
            first_half = self.performance_history[:total_count//2]
            second_half = self.performance_history[total_count//2:]
            first_avg = sum(first_half) / len(first_half)
            second_avg = sum(second_half) / len(second_half)
            
            if second_avg > first_avg + 0.1:
                trend = "improving"
            elif second_avg < first_avg - 0.1:
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"
        
        return {
            "total_feedback_count": total_count,
            "average_performance": avg_performance,
            "recent_performance": recent_avg,
            "performance_trend": trend,
            "config_source": "centralized"
        }
    
    def reset_history(self) -> None:
        """Remet à zéro l'historique."""
        self.weight_history.clear()
        self.performance_history.clear()


# ==========================================
# 🔧 FONCTIONS UTILITAIRES
# ==========================================

def create_weight_optimizer() -> WeightOptimizer:
    """Crée un optimiseur de poids avec la configuration centralisée."""
    return WeightOptimizer()


def create_adaptive_weight_manager() -> AdaptiveWeightManager:
    """Crée un gestionnaire de poids adaptatifs."""
    return AdaptiveWeightManager()


def get_default_weights() -> Dict[str, float]:
    """Retourne les poids par défaut de la configuration centralisée."""
    return {
        "lexical_weight": settings.DEFAULT_LEXICAL_WEIGHT,
        "semantic_weight": settings.DEFAULT_SEMANTIC_WEIGHT
    }


def validate_weights(weights: Dict[str, float]) -> Dict[str, float]:
    """
    Valide et normalise les poids.
    
    Args:
        weights: Dictionnaire des poids à valider
        
    Returns:
        Poids validés et normalisés
    """
    lexical_weight = weights.get("lexical_weight", settings.DEFAULT_LEXICAL_WEIGHT)
    semantic_weight = weights.get("semantic_weight", settings.DEFAULT_SEMANTIC_WEIGHT)
    
    # S'assurer que les poids sont positifs
    lexical_weight = max(0.0, lexical_weight)
    semantic_weight = max(0.0, semantic_weight)
    
    # Normaliser pour que la somme = 1.0
    total = lexical_weight + semantic_weight
    if total > 0:
        lexical_weight /= total
        semantic_weight /= total
    else:
        # Fallback vers les valeurs par défaut
        lexical_weight = settings.DEFAULT_LEXICAL_WEIGHT
        semantic_weight = settings.DEFAULT_SEMANTIC_WEIGHT
    
    return {
        "lexical_weight": lexical_weight,
        "semantic_weight": semantic_weight
    }


def get_weight_optimization_config() -> Dict[str, Any]:
    """Retourne la configuration d'optimisation des poids."""
    return {
        "default_weights": get_default_weights(),
        "quality_thresholds": {
            "excellent": settings.QUALITY_EXCELLENT_THRESHOLD,
            "good": settings.QUALITY_GOOD_THRESHOLD,
            "medium": settings.QUALITY_MEDIUM_THRESHOLD,
            "poor": settings.QUALITY_POOR_THRESHOLD
        },
        "adjustment_factors": {
            "quality_adjustment_factor": 0.2,
            "exact_phrase_boost": 0.1,
            "complex_query_semantic_boost": 0.1,
            "amount_lexical_boost": 0.05,
            "date_lexical_boost": 0.05
        },
        "constraints": {
            "min_weight": 0.1,
            "max_weight": 0.9
        },
        "config_source": "centralized"
    }