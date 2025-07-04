"""
Évaluation de la qualité des résultats de fusion.

Ce module contient les fonctions d'évaluation de la qualité
des résultats fusionnés et des métriques associées.
"""
from typing import List, Dict, Any
from search_service.models.responses import SearchResultItem
from search_service.models.search_types import SearchQuality
from search_service.core.lexical_engine import LexicalSearchResult
from search_service.core.semantic_engine import SemanticSearchResult
from search_service.core.query_processor import QueryAnalysis


class QualityAssessor:
    """Évaluateur de qualité pour les résultats de fusion."""
    
    @staticmethod
    def assess_fusion_quality(
        results: List[SearchResultItem],
        lexical_result: LexicalSearchResult,
        semantic_result: SemanticSearchResult,
        query_analysis: QueryAnalysis
    ) -> SearchQuality:
        """
        Évalue la qualité globale de la fusion.
        
        Args:
            results: Résultats fusionnés
            lexical_result: Résultats du moteur lexical
            semantic_result: Résultats du moteur sémantique
            query_analysis: Analyse de la requête
            
        Returns:
            Niveau de qualité évalué
        """
        if not results:
            return SearchQuality.POOR
        
        # Facteurs de qualité
        score_quality = QualityAssessor._assess_fused_score_quality(results)
        coverage_quality = QualityAssessor._assess_coverage_quality(
            results, lexical_result, semantic_result
        )
        consistency_quality = QualityAssessor._assess_fusion_consistency(results)
        relevance_quality = QualityAssessor._assess_fused_relevance(results, query_analysis)
        
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
    
    @staticmethod
    def _assess_fused_score_quality(results: List[SearchResultItem]) -> float:
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
    
    @staticmethod
    def _assess_coverage_quality(
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
    
    @staticmethod
    def _assess_fusion_consistency(results: List[SearchResultItem]) -> float:
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
    
    @staticmethod
    def _assess_fused_relevance(
        results: List[SearchResultItem],
        query_analysis: QueryAnalysis
    ) -> float:
        """Évalue la pertinence des résultats fusionnés."""
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


def quality_to_score(quality: SearchQuality) -> float:
    """Convertit une qualité en score numérique."""
    quality_scores = {
        SearchQuality.EXCELLENT: 1.0,
        SearchQuality.GOOD: 0.7,
        SearchQuality.MEDIUM: 0.5,
        SearchQuality.POOR: 0.2
    }
    return quality_scores.get(quality, 0.5)


def calculate_quality_metrics(results: List[SearchResultItem]) -> Dict[str, float]:
    """
    Calcule des métriques de qualité détaillées.
    
    Args:
        results: Liste des résultats à analyser
        
    Returns:
        Dictionnaire des métriques de qualité
    """
    if not results:
        return {
            "total_results": 0,
            "avg_score": 0.0,
            "score_variance": 0.0,
            "score_distribution": 0.0,
            "has_highlights": 0.0,
            "metadata_richness": 0.0
        }
    
    scores = [r.score for r in results if r.score is not None]
    
    # Métriques de base
    total_results = len(results)
    avg_score = sum(scores) / len(scores) if scores else 0.0
    
    # Variance des scores
    score_variance = 0.0
    if len(scores) > 1:
        mean = avg_score
        score_variance = sum((s - mean) ** 2 for s in scores) / len(scores)
    
    # Distribution des scores
    score_distribution = 0.0
    if scores:
        max_score = max(scores)
        min_score = min(scores)
        score_distribution = (max_score - min_score) / max_score if max_score > 0 else 0.0
    
    # Pourcentage avec highlights
    has_highlights = sum(1 for r in results if r.highlights) / total_results
    
    # Richesse des métadonnées
    metadata_richness = sum(
        1 for r in results 
        if r.metadata and len(r.metadata) > 2
    ) / total_results
    
    return {
        "total_results": total_results,
        "avg_score": avg_score,
        "score_variance": score_variance,
        "score_distribution": score_distribution,
        "has_highlights": has_highlights,
        "metadata_richness": metadata_richness
    }