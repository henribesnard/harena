"""
Moteur de recherche hybride principal - Chef d'orchestre.

Ce module coordonne les recherches lexicale et sémantique pour fournir
des résultats optimaux via différentes stratégies de fusion.
"""
import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from search_service.core.lexical_engine import LexicalSearchEngine, LexicalSearchResult
from search_service.core.semantic_engine import SemanticSearchEngine, SemanticSearchResult
from search_service.core.query_processor import QueryProcessor, QueryAnalysis
from search_service.models.search_types import SearchType, SearchQuality, SortOrder
from search_service.models.responses import SearchResultItem, SearchResponse
from search_service.utils.cache import SearchCache

logger = logging.getLogger(__name__)


class FusionStrategy(Enum):
    """Stratégies de fusion des résultats."""
    WEIGHTED_AVERAGE = "weighted_average"
    RANK_FUSION = "rank_fusion"
    SCORE_NORMALIZATION = "score_normalization"
    RECIPROCAL_RANK_FUSION = "reciprocal_rank_fusion"


@dataclass
class HybridSearchConfig:
    """Configuration pour la recherche hybride."""
    default_lexical_weight: float = 0.6
    default_semantic_weight: float = 0.4
    fusion_strategy: FusionStrategy = FusionStrategy.WEIGHTED_AVERAGE
    min_results_for_fusion: int = 2
    max_results_per_engine: int = 50
    enable_cache: bool = True
    cache_ttl_seconds: int = 300
    adaptive_weighting: bool = True
    quality_boost_factor: float = 0.2


@dataclass
class HybridSearchResult:
    """Résultat d'une recherche hybride."""
    results: List[SearchResultItem]
    total_found: int
    search_type: SearchType
    lexical_results_count: int
    semantic_results_count: int
    fusion_strategy: str
    weights_used: Dict[str, float]
    processing_time_ms: float
    quality: SearchQuality
    cache_hit: bool = False
    debug_info: Optional[Dict[str, Any]] = None


class HybridSearchEngine:
    """
    Moteur de recherche hybride - Chef d'orchestre.
    
    Coordonne les moteurs lexical et sémantique pour fournir
    des résultats optimaux selon différentes stratégies.
    
    Responsabilités:
    - Orchestration des recherches parallèles
    - Fusion intelligente des résultats
    - Gestion du cache
    - Adaptation des poids selon la qualité
    - Métriques de performance globales
    """
    
    def __init__(
        self,
        lexical_engine: Optional[LexicalSearchEngine] = None,
        semantic_engine: Optional[SemanticSearchEngine] = None,
        query_processor: Optional[QueryProcessor] = None,
        config: Optional[HybridSearchConfig] = None
    ):
        self.lexical_engine = lexical_engine
        self.semantic_engine = semantic_engine
        self.query_processor = query_processor or QueryProcessor()
        self.config = config or HybridSearchConfig()
        
        # Cache de résultats
        self.cache = SearchCache(
            max_size=1000,
            ttl_seconds=self.config.cache_ttl_seconds
        ) if self.config.enable_cache else None
        
        # Métriques globales
        self.search_count = 0
        self.cache_hits = 0
        self.total_processing_time = 0.0
        self.fusion_stats = {strategy.value: 0 for strategy in FusionStrategy}
        self.quality_distribution = {quality.value: 0 for quality in SearchQuality}
        
        logger.info("Hybrid search engine initialized")
    
    async def search(
        self,
        query: str,
        user_id: int,
        search_type: SearchType = SearchType.HYBRID,
        limit: int = 20,
        offset: int = 0,
        lexical_weight: Optional[float] = None,
        semantic_weight: Optional[float] = None,
        similarity_threshold: Optional[float] = None,
        sort_order: SortOrder = SortOrder.RELEVANCE,
        filters: Optional[Dict[str, Any]] = None,
        use_cache: bool = True,
        debug: bool = False
    ) -> HybridSearchResult:
        """
        Effectue une recherche hybride intelligente.
        
        Args:
            query: Requête de recherche
            user_id: ID de l'utilisateur
            search_type: Type de recherche (LEXICAL, SEMANTIC, HYBRID)
            limit: Nombre de résultats à retourner
            offset: Décalage pour pagination
            lexical_weight: Poids pour recherche lexicale (0-1)
            semantic_weight: Poids pour recherche sémantique (0-1)
            similarity_threshold: Seuil de similarité sémantique
            sort_order: Ordre de tri des résultats
            filters: Filtres à appliquer
            use_cache: Utiliser le cache si disponible
            debug: Retourner informations de debug
            
        Returns:
            HybridSearchResult avec résultats fusionnés
        """
        start_time = time.time()
        self.search_count += 1
        
        # Normaliser et analyser la requête
        query_analysis = self.query_processor.process_query(query)
        processed_query = query_analysis.processed_query
        
        # Vérifier le cache d'abord
        cache_key = None
        if use_cache and self.cache:
            cache_key = self._generate_cache_key(
                processed_query, user_id, search_type, limit, offset,
                lexical_weight, semantic_weight, similarity_threshold,
                sort_order, filters
            )
            cached_result = self.cache.get(cache_key)
            if cached_result:
                self.cache_hits += 1
                cached_result.cache_hit = True
                return cached_result
        
        # Déterminer les poids adaptatifs
        weights = self._calculate_adaptive_weights(
            query_analysis, lexical_weight, semantic_weight
        )
        
        # Exécuter la stratégie de recherche appropriée
        search_results = await self._execute_search_strategy(
            processed_query, user_id, search_type, weights, limit, offset,
            similarity_threshold, sort_order, filters, debug
        )
        
        # Finaliser le résultat
        result = self._finalize_search_result(
            search_results, query_analysis, start_time, debug
        )
        
        # Mettre en cache si activé
        if use_cache and self.cache and cache_key:
            self.cache.set(cache_key, result)
        
        # Mettre à jour les métriques
        self._update_metrics(result)
        
        return result
    
    def _generate_cache_key(
        self,
        query: str,
        user_id: int,
        search_type: SearchType,
        limit: int,
        offset: int,
        lexical_weight: Optional[float],
        semantic_weight: Optional[float],
        similarity_threshold: Optional[float],
        sort_order: SortOrder,
        filters: Optional[Dict[str, Any]]
    ) -> str:
        """Génère une clé de cache unique pour la requête."""
        key_parts = [
            f"q:{hash(query)}",
            f"u:{user_id}",
            f"t:{search_type.value}",
            f"l:{limit}",
            f"o:{offset}",
            f"lw:{lexical_weight}",
            f"sw:{semantic_weight}",
            f"st:{similarity_threshold}",
            f"so:{sort_order.value}",
            f"f:{hash(str(filters)) if filters else 'none'}"
        ]
        return ":".join(key_parts)
    
    def _calculate_adaptive_weights(
        self,
        query_analysis: QueryAnalysis,
        lexical_weight: Optional[float],
        semantic_weight: Optional[float]
    ) -> Dict[str, float]:
        """Calcule les poids adaptatifs selon l'analyse de requête."""
        if not self.config.adaptive_weighting:
            # Utiliser poids fournis ou par défaut
            lw = lexical_weight or self.config.default_lexical_weight
            sw = semantic_weight or self.config.default_semantic_weight
            
            # Normaliser pour que la somme soit 1
            total = lw + sw
            if total > 0:
                return {"lexical": lw / total, "semantic": sw / total}
            else:
                return {"lexical": 0.5, "semantic": 0.5}
        
        # Adaptation basée sur le type et la confiance de la requête
        base_lexical = self.config.default_lexical_weight
        base_semantic = self.config.default_semantic_weight
        
        # Ajustements selon le type de requête
        if query_analysis.query_type == "exact_match":
            # Favoriser lexical pour correspondances exactes
            base_lexical += 0.2
            base_semantic -= 0.2
        elif query_analysis.query_type == "conceptual":
            # Favoriser sémantique pour requêtes conceptuelles
            base_lexical -= 0.15
            base_semantic += 0.15
        elif query_analysis.query_type == "amount_focused":
            # Équilibrer pour requêtes de montant
            base_lexical += 0.1
            base_semantic -= 0.1
        
        # Ajustement selon la confiance
        confidence_factor = (query_analysis.confidence - 0.5) * 0.1
        if query_analysis.query_type in ["exact_match", "amount_focused"]:
            base_lexical += confidence_factor
            base_semantic -= confidence_factor
        else:
            base_lexical -= confidence_factor
            base_semantic += confidence_factor
        
        # Assurer que les poids restent dans [0.1, 0.9]
        base_lexical = max(0.1, min(0.9, base_lexical))
        base_semantic = max(0.1, min(0.9, base_semantic))
        
        # Normaliser
        total = base_lexical + base_semantic
        return {
            "lexical": base_lexical / total,
            "semantic": base_semantic / total
        }
    
    async def _execute_search_strategy(
        self,
        query: str,
        user_id: int,
        strategy: SearchType,
        weights: Dict[str, float],
        limit: int,
        offset: int,
        similarity_threshold: Optional[float],
        sort_order: SortOrder,
        filters: Optional[Dict[str, Any]],
        debug: bool
    ) -> Dict[str, Any]:
        """Exécute la stratégie de recherche appropriée."""
        if strategy == SearchType.LEXICAL:
            return await self._execute_lexical_only(
                query, user_id, limit, offset, sort_order, filters, debug
            )
        elif strategy == SearchType.SEMANTIC:
            return await self._execute_semantic_only(
                query, user_id, limit, offset, similarity_threshold, sort_order, filters, debug
            )
        else:  # HYBRID
            return await self._execute_hybrid_search(
                query, user_id, weights, limit, offset, 
                similarity_threshold, sort_order, filters, debug
            )
    
    async def _execute_lexical_only(
        self,
        query: str,
        user_id: int,
        limit: int,
        offset: int,
        sort_order: SortOrder,
        filters: Optional[Dict[str, Any]],
        debug: bool
    ) -> Dict[str, Any]:
        """Exécute une recherche lexicale uniquement."""
        if not self.lexical_engine:
            raise Exception("Lexical engine not available")
        
        lexical_result = await self.lexical_engine.search(
            query=query,
            user_id=user_id,
            limit=limit,
            offset=offset,
            sort_order=sort_order,
            filters=filters,
            debug=debug
        )
        
        return {
            "search_type": SearchType.LEXICAL,
            "lexical_result": lexical_result,
            "semantic_result": None,
            "fusion_strategy": "lexical_only"
        }
    
    async def _execute_semantic_only(
        self,
        query: str,
        user_id: int,
        limit: int,
        offset: int,
        similarity_threshold: Optional[float],
        sort_order: SortOrder,
        filters: Optional[Dict[str, Any]],
        debug: bool
    ) -> Dict[str, Any]:
        """Exécute une recherche sémantique uniquement."""
        if not self.semantic_engine:
            raise Exception("Semantic engine not available")
        
        semantic_result = await self.semantic_engine.search(
            query=query,
            user_id=user_id,
            limit=limit,
            offset=offset,
            similarity_threshold=similarity_threshold,
            sort_order=sort_order,
            filters=filters,
            debug=debug
        )
        
        return {
            "search_type": SearchType.SEMANTIC,
            "lexical_result": None,
            "semantic_result": semantic_result,
            "fusion_strategy": "semantic_only"
        }
    
    async def _execute_hybrid_search(
        self,
        query: str,
        user_id: int,
        weights: Dict[str, float],
        limit: int,
        offset: int,
        similarity_threshold: Optional[float],
        sort_order: SortOrder,
        filters: Optional[Dict[str, Any]],
        debug: bool
    ) -> Dict[str, Any]:
        """Exécute une recherche hybride avec fusion."""
        if not self.lexical_engine or not self.semantic_engine:
            raise Exception("Both lexical and semantic engines required for hybrid search")
        
        # Recherches en parallèle avec limite élargie pour meilleure fusion
        expanded_limit = min(limit + offset + 10, self.config.max_results_per_engine)
        
        lexical_task = self.lexical_engine.search(
            query=query,
            user_id=user_id,
            limit=expanded_limit,
            offset=0,  # Pas d'offset pour la fusion
            sort_order=SortOrder.RELEVANCE,  # Toujours par pertinence pour fusion
            filters=filters,
            debug=debug
        )
        
        semantic_task = self.semantic_engine.search(
            query=query,
            user_id=user_id,
            limit=expanded_limit,
            offset=0,
            similarity_threshold=similarity_threshold,
            sort_order=SortOrder.RELEVANCE,
            filters=filters,
            debug=debug
        )
        
        # Attendre les deux résultats
        lexical_result, semantic_result = await asyncio.gather(
            lexical_task, semantic_task, return_exceptions=True
        )
        
        # Vérifier les exceptions
        if isinstance(lexical_result, Exception):
            logger.warning(f"Lexical search failed in hybrid: {lexical_result}")
            lexical_result = None
        
        if isinstance(semantic_result, Exception):
            logger.warning(f"Semantic search failed in hybrid: {semantic_result}")
            semantic_result = None
        
        if not lexical_result and not semantic_result:
            raise Exception("Both search engines failed")
        
        # Fusionner les résultats
        fused_results = self._fuse_search_results(
            lexical_result, semantic_result, weights, limit, offset, sort_order
        )
        
        return {
            "search_type": SearchType.HYBRID,
            "lexical_result": lexical_result,
            "semantic_result": semantic_result,
            "fused_results": fused_results,
            "fusion_strategy": self.config.fusion_strategy.value,
            "weights_used": weights
        }
    
    def _fuse_search_results(
        self,
        lexical_result: Optional[LexicalSearchResult],
        semantic_result: Optional[SemanticSearchResult],
        weights: Dict[str, float],
        limit: int,
        offset: int,
        sort_order: SortOrder
    ) -> List[SearchResultItem]:
        """Fusionne les résultats lexicaux et sémantiques."""
        if self.config.fusion_strategy == FusionStrategy.WEIGHTED_AVERAGE:
            return self._weighted_average_fusion(
                lexical_result, semantic_result, weights, limit, offset, sort_order
            )
        elif self.config.fusion_strategy == FusionStrategy.RANK_FUSION:
            return self._rank_fusion(
                lexical_result, semantic_result, weights, limit, offset, sort_order
            )
        elif self.config.fusion_strategy == FusionStrategy.RECIPROCAL_RANK_FUSION:
            return self._reciprocal_rank_fusion(
                lexical_result, semantic_result, weights, limit, offset, sort_order
            )
        else:  # SCORE_NORMALIZATION
            return self._score_normalization_fusion(
                lexical_result, semantic_result, weights, limit, offset, sort_order
            )
    
    def _weighted_average_fusion(
        self,
        lexical_result: Optional[LexicalSearchResult],
        semantic_result: Optional[SemanticSearchResult],
        weights: Dict[str, float],
        limit: int,
        offset: int,
        sort_order: SortOrder
    ) -> List[SearchResultItem]:
        """Fusion par moyenne pondérée des scores normalisés."""
        # Collecter tous les résultats avec scores normalisés
        all_results = {}
        
        # Ajouter résultats lexicaux
        if lexical_result and lexical_result.results:
            lexical_weight = weights.get("lexical", 0.5)
            for result in lexical_result.results:
                transaction_id = result.transaction_id
                
                # Normaliser le score lexical (0-1)
                normalized_score = self._normalize_lexical_score(result.score, lexical_result.max_score)
                weighted_score = normalized_score * lexical_weight
                
                all_results[transaction_id] = {
                    "item": result,
                    "lexical_score": normalized_score,
                    "semantic_score": 0.0,
                    "combined_score": weighted_score,
                    "sources": ["lexical"]
                }
        
        # Ajouter résultats sémantiques
        if semantic_result and semantic_result.results:
            semantic_weight = weights.get("semantic", 0.5)
            for result in semantic_result.results:
                transaction_id = result.transaction_id
                
                # Score sémantique déjà normalisé (0-1)
                normalized_score = result.score
                weighted_score = normalized_score * semantic_weight
                
                if transaction_id in all_results:
                    # Combiner avec résultat existant
                    existing = all_results[transaction_id]
                    existing["normalized_score"] += weighted_score
                    existing["sources"].append("semantic")
                    existing["item"].combined_score = existing["normalized_score"]
                else:
                    # Nouveau résultat
                    all_results[transaction_id] = {
                        "item": result,
                        "normalized_score": weighted_score,
                        "sources": ["semantic"]
                    }
        
        # Boost pour convergence
        for result_data in all_results.values():
            if len(result_data["sources"]) == 2:
                boost = self.config.quality_boost_factor
                result_data["normalized_score"] *= (1 + boost)
                result_data["item"].combined_score = result_data["normalized_score"]
        
        # Trier par score normalisé
        sorted_results = sorted(
            all_results.values(),
            key=lambda x: x["normalized_score"],
            reverse=True
        )
        
        result_items = [data["item"] for data in sorted_results]
        if sort_order != SortOrder.RELEVANCE:
            result_items = self._apply_final_sorting(result_items, sort_order)
        
        start_idx = offset
        end_idx = offset + limit
        return result_items[start_idx:end_idx]
    
    def _normalize_lexical_score(self, score: float, max_score: float) -> float:
        """Normalise un score lexical entre 0 et 1."""
        if max_score <= 0:
            return 0.0
        
        # Normalisation logarithmique pour les scores Elasticsearch
        normalized = min(score / max_score, 1.0)
        
        # Ajustement pour améliorer la distribution
        return min(normalized * 1.2, 1.0)
    
    def _apply_final_sorting(
        self, 
        results: List[SearchResultItem], 
        sort_order: SortOrder
    ) -> List[SearchResultItem]:
        """Applique le tri final aux résultats fusionnés."""
        if sort_order == SortOrder.DATE_DESC:
            return sorted(results, key=lambda x: (x.transaction_date, x.combined_score), reverse=True)
        elif sort_order == SortOrder.DATE_ASC:
            return sorted(results, key=lambda x: (x.transaction_date, x.combined_score))
        elif sort_order == SortOrder.AMOUNT_DESC:
            return sorted(results, key=lambda x: (abs(x.amount), x.combined_score), reverse=True)
        elif sort_order == SortOrder.AMOUNT_ASC:
            return sorted(results, key=lambda x: (abs(x.amount), x.combined_score))
        else:
            return results  # Déjà trié par pertinence
    
    def _finalize_search_result(
        self,
        search_results: Dict[str, Any],
        query_analysis: QueryAnalysis,
        start_time: float,
        debug: bool
    ) -> HybridSearchResult:
        """Finalise le résultat de recherche."""
        processing_time = (time.time() - start_time) * 1000
        self.total_processing_time += processing_time
        
        search_type = search_results["search_type"]
        lexical_result = search_results.get("lexical_result")
        semantic_result = search_results.get("semantic_result")
        
        # Déterminer les résultats finaux
        if search_type == SearchType.HYBRID:
            final_results = search_results.get("fused_results", [])
            fusion_strategy = search_results.get("fusion_strategy", "unknown")
            weights_used = search_results.get("weights_used", {})
        elif search_type == SearchType.LEXICAL and lexical_result:
            final_results = lexical_result.results
            fusion_strategy = "lexical_only"
            weights_used = {"lexical": 1.0, "semantic": 0.0}
        elif search_type == SearchType.SEMANTIC and semantic_result:
            final_results = semantic_result.results
            fusion_strategy = "semantic_only"
            weights_used = {"lexical": 0.0, "semantic": 1.0}
        else:
            final_results = []
            fusion_strategy = "failed"
            weights_used = {}
        
        # Calculer la qualité globale
        quality = self._calculate_overall_quality(
            lexical_result, semantic_result, final_results, search_type
        )
        
        # Informations de debug
        debug_info = None
        if debug:
            debug_info = {
                "query_analysis": {
                    "original_query": query_analysis.original_query,
                    "query_type": query_analysis.query_type,
                    "confidence": query_analysis.confidence,
                    "detected_entities": query_analysis.detected_entities
                },
                "lexical_debug": lexical_result.debug_info if lexical_result else None,
                "semantic_debug": semantic_result.debug_info if semantic_result else None,
                "fusion_details": {
                    "strategy": fusion_strategy,
                    "weights": weights_used,
                    "total_candidates": (
                        len(lexical_result.results) if lexical_result else 0
                    ) + (
                        len(semantic_result.results) if semantic_result else 0
                    )
                }
            }
        
        return HybridSearchResult(
            results=final_results,
            total_found=len(final_results),
            search_type=search_type,
            lexical_results_count=len(lexical_result.results) if lexical_result else 0,
            semantic_results_count=len(semantic_result.results) if semantic_result else 0,
            fusion_strategy=fusion_strategy,
            weights_used=weights_used,
            processing_time_ms=processing_time,
            quality=quality,
            debug_info=debug_info
        )
    
    def _calculate_overall_quality(
        self,
        lexical_result: Optional[LexicalSearchResult],
        semantic_result: Optional[SemanticSearchResult],
        final_results: List[SearchResultItem],
        search_type: SearchType
    ) -> SearchQuality:
        """Calcule la qualité globale de la recherche."""
        if not final_results:
            return SearchQuality.FAILED
        
        # Calculs de base
        total_results = len(final_results)
        avg_score = sum(getattr(r, 'combined_score', r.score) for r in final_results) / total_results
        
        # Facteurs de qualité selon le type de recherche
        if search_type == SearchType.HYBRID:
            # Qualité hybride basée sur convergence et scores
            lexical_count = len(lexical_result.results) if lexical_result else 0
            semantic_count = len(semantic_result.results) if semantic_result else 0
            
            # Bonus pour convergence des moteurs
            convergence_bonus = 0
            if lexical_count > 0 and semantic_count > 0:
                # Calculer taux de convergence (résultats communs)
                if lexical_result and semantic_result:
                    lexical_ids = {r.transaction_id for r in lexical_result.results}
                    semantic_ids = {r.transaction_id for r in semantic_result.results}
                    common_ids = lexical_ids.intersection(semantic_ids)
                    convergence_rate = len(common_ids) / max(len(lexical_ids), len(semantic_ids))
                    convergence_bonus = convergence_rate * 0.2
            
            # Score combiné avec bonus
            quality_score = avg_score + convergence_bonus
            
            if quality_score >= 0.8:
                return SearchQuality.EXCELLENT
            elif quality_score >= 0.6:
                return SearchQuality.GOOD
            elif quality_score >= 0.4:
                return SearchQuality.AVERAGE
            else:
                return SearchQuality.POOR
        
        elif search_type == SearchType.LEXICAL:
            # Qualité lexicale basée sur scores et nombre de résultats
            if avg_score >= 0.7 and total_results >= 5:
                return SearchQuality.EXCELLENT
            elif avg_score >= 0.5 and total_results >= 3:
                return SearchQuality.GOOD
            elif avg_score >= 0.3:
                return SearchQuality.AVERAGE
            else:
                return SearchQuality.POOR
        
        else:  # SEMANTIC
            # Qualité sémantique basée sur seuils de similarité
            if avg_score >= 0.85:
                return SearchQuality.EXCELLENT
            elif avg_score >= 0.75:
                return SearchQuality.GOOD
            elif avg_score >= 0.6:
                return SearchQuality.AVERAGE
            else:
                return SearchQuality.POOR
    
    def _update_metrics(self, result: HybridSearchResult):
        """Met à jour les métriques globales."""
        self.fusion_stats[result.fusion_strategy] += 1
        self.quality_distribution[result.quality.value] += 1
    
    async def suggest_query_improvements(
        self,
        query: str,
        user_id: int,
        search_result: HybridSearchResult
    ) -> List[str]:
        """Suggère des améliorations de requête basées sur les résultats hybrides."""
        suggestions = []
        
        # Suggestions basées sur la qualité
        if search_result.quality == SearchQuality.FAILED:
            suggestions.append("Essayez des termes plus généraux")
            suggestions.append("Vérifiez l'orthographe de votre requête")
        elif search_result.quality == SearchQuality.POOR:
            suggestions.append("Ajoutez des mots-clés plus spécifiques")
            if search_result.semantic_results_count > search_result.lexical_results_count:
                suggestions.append("Essayez des termes exacts plutôt que conceptuels")
        
        # Suggestions basées sur les résultats par moteur
        if search_result.lexical_results_count == 0 and search_result.semantic_results_count > 0:
            suggestions.append("Vos termes sont corrects conceptuellement, essayez des mots-clés plus précis")
        elif search_result.semantic_results_count == 0 and search_result.lexical_results_count > 0:
            suggestions.append("Essayez des termes plus généraux ou des synonymes")
        
        # Suggestions basées sur l'analyse de requête
        query_analysis = self.query_processor.process_query(query)
        if not query_analysis.detected_entities.get("categories") and len(query.split()) == 1:
            suggestions.append("Ajoutez du contexte (ex: 'restaurant italien' au lieu de 'restaurant')")
        
        return suggestions[:3]  # Maximum 3 suggestions
    
    def get_engine_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques complètes du moteur hybride."""
        avg_processing_time = (
            self.total_processing_time / self.search_count 
            if self.search_count > 0 else 0
        )
        
        cache_hit_rate = (
            self.cache_hits / self.search_count * 100 
            if self.search_count > 0 else 0
        )
        
        return {
            "total_searches": self.search_count,
            "cache_hits": self.cache_hits,
            "cache_hit_rate_percent": round(cache_hit_rate, 2),
            "avg_processing_time_ms": round(avg_processing_time, 2),
            "fusion_strategy_usage": dict(self.fusion_stats),
            "quality_distribution": dict(self.quality_distribution),
            "engines_status": {
                "lexical_available": self.lexical_engine is not None,
                "semantic_available": self.semantic_engine is not None,
                "query_processor_available": self.query_processor is not None
            },
            "config": {
                "fusion_strategy": self.config.fusion_strategy.value,
                "default_weights": {
                    "lexical": self.config.default_lexical_weight,
                    "semantic": self.config.default_semantic_weight
                },
                "adaptive_weighting": self.config.adaptive_weighting,
                "cache_enabled": self.config.enable_cache,
                "cache_ttl_seconds": self.config.cache_ttl_seconds
            }
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Vérifie la santé du moteur hybride et de ses composants."""
        health_status = {
            "status": "healthy",
            "timestamp": time.time(),
            "components": {}
        }
        
        # Vérifier moteur lexical
        if self.lexical_engine:
            try:
                lexical_health = await self.lexical_engine.health_check()
                health_status["components"]["lexical_engine"] = lexical_health
            except Exception as e:
                health_status["components"]["lexical_engine"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                health_status["status"] = "degraded"
        else:
            health_status["components"]["lexical_engine"] = {
                "status": "unavailable",
                "error": "Engine not initialized"
            }
        
        # Vérifier moteur sémantique
        if self.semantic_engine:
            try:
                semantic_health = await self.semantic_engine.health_check()
                health_status["components"]["semantic_engine"] = semantic_health
            except Exception as e:
                health_status["components"]["semantic_engine"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                health_status["status"] = "degraded"
        else:
            health_status["components"]["semantic_engine"] = {
                "status": "unavailable",
                "error": "Engine not initialized"
            }
        
        # Vérifier processeur de requêtes
        health_status["components"]["query_processor"] = {
            "status": "healthy" if self.query_processor else "unavailable"
        }
        
        # Vérifier cache
        if self.cache:
            cache_stats = self.cache.get_stats()
            health_status["components"]["cache"] = {
                "status": "healthy",
                "stats": cache_stats
            }
        else:
            health_status["components"]["cache"] = {
                "status": "disabled"
            }
        
        # Déterminer statut global
        component_statuses = [
            comp.get("status", "unknown") 
            for comp in health_status["components"].values()
        ]
        
        if any(status == "unhealthy" for status in component_statuses):
            health_status["status"] = "unhealthy"
        elif any(status in ["degraded", "unavailable"] for status in component_statuses):
            health_status["status"] = "degraded"
        
        return health_status
    
    async def cleanup(self):
        """Nettoie les ressources du moteur hybride."""
        logger.info("Cleaning up hybrid search engine...")
        
        # Nettoyer le cache
        if self.cache:
            self.cache.clear()
        
        # Nettoyer les moteurs
        if self.lexical_engine and hasattr(self.lexical_engine, 'cleanup'):
            await self.lexical_engine.cleanup()
        
        if self.semantic_engine and hasattr(self.semantic_engine, 'cleanup'):
            await self.semantic_engine.cleanup()
        
        # Réinitialiser les métriques
        self.search_count = 0
        self.cache_hits = 0
        self.total_processing_time = 0.0
        self.fusion_stats = {strategy.value: 0 for strategy in FusionStrategy}
        self.quality_distribution = {quality.value: 0 for quality in SearchQuality}
        
        logger.info("Hybrid search engine cleanup completed")
    
    def __str__(self) -> str:
        """Représentation string du moteur hybride."""
        return (
            f"HybridSearchEngine("
            f"fusion_strategy={self.config.fusion_strategy.value}, "
            f"searches={self.search_count}, "
            f"cache_hits={self.cache_hits}, "
            f"lexical={'✓' if self.lexical_engine else '✗'}, "
            f"semantic={'✓' if self.semantic_engine else '✗'}"
            f")"
        )
    
    def __repr__(self) -> str:
        """Représentation détaillée du moteur hybride."""
        return self.__str__()
        
        # Appliquer boost pour résultats trouvés par les deux moteurs
        for result_data in all_results.values():
            if len(result_data["sources"]) == 2:
                # Boost pour convergence des deux moteurs
                boost = self.config.quality_boost_factor
                result_data["combined_score"] *= (1 + boost)
                result_data["item"].combined_score = result_data["combined_score"]
        
        # Trier par score combiné
        sorted_results = sorted(
            all_results.values(),
            key=lambda x: x["combined_score"],
            reverse=True
        )
        
        # Extraire les items et appliquer pagination
        result_items = [data["item"] for data in sorted_results]
        
        # Appliquer tri final si différent de pertinence
        if sort_order != SortOrder.RELEVANCE:
            result_items = self._apply_final_sorting(result_items, sort_order)
        
        # Pagination
        start_idx = offset
        end_idx = offset + limit
        return result_items[start_idx:end_idx]
    
    def _rank_fusion(
        self,
        lexical_result: Optional[LexicalSearchResult],
        semantic_result: Optional[SemanticSearchResult],
        weights: Dict[str, float],
        limit: int,
        offset: int,
        sort_order: SortOrder
    ) -> List[SearchResultItem]:
        """Fusion basée sur les rangs avec poids."""
        all_results = {}
        
        # Ajouter résultats lexicaux avec scores basés sur le rang
        if lexical_result and lexical_result.results:
            lexical_weight = weights.get("lexical", 0.5)
            for rank, result in enumerate(lexical_result.results):
                transaction_id = result.transaction_id
                # Score basé sur le rang inverse (rang 0 = score 1.0)
                rank_score = 1.0 / (rank + 1)
                weighted_score = rank_score * lexical_weight
                
                all_results[transaction_id] = {
                    "item": result,
                    "lexical_rank": rank,
                    "semantic_rank": -1,
                    "combined_score": weighted_score,
                    "sources": ["lexical"]
                }
        
        # Ajouter résultats sémantiques
        if semantic_result and semantic_result.results:
            semantic_weight = weights.get("semantic", 0.5)
            for rank, result in enumerate(semantic_result.results):
                transaction_id = result.transaction_id
                rank_score = 1.0 / (rank + 1)
                weighted_score = rank_score * semantic_weight
                
                if transaction_id in all_results:
                    # Combiner avec résultat lexical
                    existing = all_results[transaction_id]
                    existing["semantic_rank"] = rank
                    existing["combined_score"] += weighted_score
                    existing["sources"].append("semantic")
                    existing["item"].combined_score = existing["combined_score"]
                else:
                    # Nouveau résultat sémantique
                    all_results[transaction_id] = {
                        "item": result,
                        "lexical_rank": -1,
                        "semantic_rank": rank,
                        "combined_score": weighted_score,
                        "sources": ["semantic"]
                    }
        
        # Boost pour apparition dans les deux listes
        for result_data in all_results.values():
            if len(result_data["sources"]) == 2:
                boost = self.config.quality_boost_factor
                result_data["combined_score"] *= (1 + boost)
                result_data["item"].combined_score = result_data["combined_score"]
        
        # Trier et paginer
        sorted_results = sorted(
            all_results.values(),
            key=lambda x: x["combined_score"],
            reverse=True
        )
        
        result_items = [data["item"] for data in sorted_results]
        if sort_order != SortOrder.RELEVANCE:
            result_items = self._apply_final_sorting(result_items, sort_order)
        
        start_idx = offset
        end_idx = offset + limit
        return result_items[start_idx:end_idx]
    
    def _reciprocal_rank_fusion(
        self,
        lexical_result: Optional[LexicalSearchResult],
        semantic_result: Optional[SemanticSearchResult],
        weights: Dict[str, float],
        limit: int,
        offset: int,
        sort_order: SortOrder
    ) -> List[SearchResultItem]:
        """Fusion par rang réciproque (RRF)."""
        k = 60  # Constante pour RRF
        all_results = {}
        
        # Traiter résultats lexicaux
        if lexical_result and lexical_result.results:
            lexical_weight = weights.get("lexical", 0.5)
            for rank, result in enumerate(lexical_result.results):
                transaction_id = result.transaction_id
                rrf_score = lexical_weight * (1.0 / (k + rank + 1))
                
                all_results[transaction_id] = {
                    "item": result,
                    "rrf_score": rrf_score,
                    "sources": ["lexical"]
                }
        
        # Traiter résultats sémantiques
        if semantic_result and semantic_result.results:
            semantic_weight = weights.get("semantic", 0.5)
            for rank, result in enumerate(semantic_result.results):
                transaction_id = result.transaction_id
                rrf_score = semantic_weight * (1.0 / (k + rank + 1))
                
                if transaction_id in all_results:
                    # Additionner les scores RRF
                    existing = all_results[transaction_id]
                    existing["rrf_score"] += rrf_score
                    existing["sources"].append("semantic")
                    existing["item"].combined_score = existing["rrf_score"]
                else:
                    # Nouveau résultat
                    all_results[transaction_id] = {
                        "item": result,
                        "rrf_score": rrf_score,
                        "sources": ["semantic"]
                    }
        
        # Bonus pour convergence
        for result_data in all_results.values():
            if len(result_data["sources"]) == 2:
                boost = self.config.quality_boost_factor
                result_data["rrf_score"] *= (1 + boost)
                result_data["item"].combined_score = result_data["rrf_score"]
        
        # Trier par score RRF
        sorted_results = sorted(
            all_results.values(),
            key=lambda x: x["rrf_score"],
            reverse=True
        )
        
        result_items = [data["item"] for data in sorted_results]
        if sort_order != SortOrder.RELEVANCE:
            result_items = self._apply_final_sorting(result_items, sort_order)
        
        start_idx = offset
        end_idx = offset + limit
        return result_items[start_idx:end_idx]
    
    def _score_normalization_fusion(
        self,
        lexical_result: Optional[LexicalSearchResult],
        semantic_result: Optional[SemanticSearchResult],
        weights: Dict[str, float],
        limit: int,
        offset: int,
        sort_order: SortOrder
    ) -> List[SearchResultItem]:
        """Fusion avec normalisation Z-score."""
        all_results = {}
        
        # Calculer statistiques pour normalisation
        lexical_scores = []
        semantic_scores = []
        
        if lexical_result and lexical_result.results:
            lexical_scores = [r.score for r in lexical_result.results]
        
        if semantic_result and semantic_result.results:
            semantic_scores = [r.score for r in semantic_result.results]
        
        # Moyennes et écarts-types
        lexical_mean = sum(lexical_scores) / len(lexical_scores) if lexical_scores else 0
        lexical_std = (sum((x - lexical_mean) ** 2 for x in lexical_scores) / len(lexical_scores)) ** 0.5 if len(lexical_scores) > 1 else 1
        
        semantic_mean = sum(semantic_scores) / len(semantic_scores) if semantic_scores else 0
        semantic_std = (sum((x - semantic_mean) ** 2 for x in semantic_scores) / len(semantic_scores)) ** 0.5 if len(semantic_scores) > 1 else 1
        
        # Normaliser et combiner
        if lexical_result and lexical_result.results:
            lexical_weight = weights.get("lexical", 0.5)
            for result in lexical_result.results:
                transaction_id = result.transaction_id
                z_score = (result.score - lexical_mean) / lexical_std if lexical_std > 0 else 0
                normalized_score = max(0, min(1, (z_score + 3) / 6))  # Normaliser Z-score à [0,1]
                weighted_score = normalized_score * lexical_weight
                
                all_results[transaction_id] = {
                    "item": result,
                    "normalized_score": weighted_score,
                    "sources": ["lexical"]
                }
        
        if semantic_result and semantic_result.results:
            semantic_weight = weights.get("semantic", 0.5)
            for result in semantic_result.results:
                transaction_id = result.transaction_id
                z_score = (result.score - semantic_mean) / semantic_std if semantic_std > 0 else 0
                normalized_score = max(0, min(1, (z_score + 3) / 6))
                weighted_score = normalized_score * semantic_weight