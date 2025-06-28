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
            query: Terme de recherche
            user_id: ID de l'utilisateur
            search_type: Type de recherche (lexical, semantic, hybrid)
            limit: Nombre de résultats
            offset: Décalage pour pagination
            lexical_weight: Poids de la recherche lexicale
            semantic_weight: Poids de la recherche sémantique
            similarity_threshold: Seuil de similarité pour sémantique
            sort_order: Ordre de tri
            filters: Filtres additionnels
            use_cache: Utiliser le cache
            debug: Inclure informations de debug
            
        Returns:
            Résultats de recherche hybride
        """
        start_time = time.time()
        self.search_count += 1
        
        # Vérifier le cache d'abord
        if use_cache and self.cache:
            cache_key = self._generate_cache_key(
                query, user_id, search_type, limit, offset, 
                lexical_weight, semantic_weight, similarity_threshold,
                sort_order, filters
            )
            
            cached_result = self.cache.get(cache_key)
            if cached_result:
                self.cache_hits += 1
                cached_result.cache_hit = True
                logger.debug(f"Cache hit for query: {query}")
                return cached_result
        
        try:
            # 1. Analyser la requête
            query_analysis = self.query_processor.process_query(query)
            
            # 2. Déterminer la stratégie optimale
            optimal_strategy = self._determine_optimal_strategy(
                query_analysis, search_type
            )
            
            # 3. Ajuster les poids si nécessaire
            final_weights = self._calculate_optimal_weights(
                query_analysis, lexical_weight, semantic_weight
            )
            
            # 4. Exécuter la recherche selon la stratégie
            search_results = await self._execute_search_strategy(
                query=query,
                user_id=user_id,
                strategy=optimal_strategy,
                weights=final_weights,
                limit=limit,
                offset=offset,
                similarity_threshold=similarity_threshold,
                sort_order=sort_order,
                filters=filters,
                debug=debug
            )
            
            # 5. Post-traitement et finalisation
            final_result = self._finalize_search_result(
                search_results, query_analysis, start_time, debug
            )
            
            # 6. Mettre en cache si activé
            if use_cache and self.cache and final_result.quality != SearchQuality.FAILED:
                self.cache.put(cache_key, final_result)
            
            # 7. Mettre à jour les métriques
            self._update_metrics(final_result)
            
            logger.debug(
                f"Hybrid search completed: {len(final_result.results)} results, "
                f"type: {final_result.search_type}, quality: {final_result.quality}, "
                f"time: {final_result.processing_time_ms:.2f}ms"
            )
            
            return final_result
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            logger.error(f"Hybrid search failed: {e}")
            
            return HybridSearchResult(
                results=[],
                total_found=0,
                search_type=search_type,
                lexical_results_count=0,
                semantic_results_count=0,
                fusion_strategy="failed",
                weights_used={},
                processing_time_ms=processing_time,
                quality=SearchQuality.FAILED,
                debug_info={"error": str(e)} if debug else None
            )
    
    def _determine_optimal_strategy(
        self,
        query_analysis: QueryAnalysis,
        requested_type: SearchType
    ) -> SearchType:
        """Détermine la stratégie de recherche optimale."""
        # Si type spécifique demandé et moteur disponible
        if requested_type == SearchType.LEXICAL and self.lexical_engine:
            return SearchType.LEXICAL
        elif requested_type == SearchType.SEMANTIC and self.semantic_engine:
            return SearchType.SEMANTIC
        elif requested_type == SearchType.HYBRID and self.lexical_engine and self.semantic_engine:
            return SearchType.HYBRID
        
        # Sinon, déterminer automatiquement
        if not self.lexical_engine and self.semantic_engine:
            return SearchType.SEMANTIC
        elif self.lexical_engine and not self.semantic_engine:
            return SearchType.LEXICAL
        elif not self.lexical_engine and not self.semantic_engine:
            raise Exception("No search engines available")
        
        # Les deux moteurs disponibles - choisir selon l'analyse
        query_type = query_analysis.query_type
        confidence = query_analysis.confidence
        
        # Préférences par type de requête
        if query_type in ["amount_search", "date_search"] and confidence > 0.7:
            # Entités spécifiques = lexical plus efficace
            return SearchType.LEXICAL
        elif query_type in ["category_search", "similarity_search"]:
            # Recherche conceptuelle = sémantique préférable
            return SearchType.SEMANTIC
        elif len(query_analysis.cleaned_query.split()) == 1:
            # Mot unique = hybride pour couverture maximale
            return SearchType.HYBRID
        else:
            # Par défaut hybride si les deux disponibles
            return SearchType.HYBRID
    
    def _calculate_optimal_weights(
        self,
        query_analysis: QueryAnalysis,
        lexical_weight: Optional[float] = None,
        semantic_weight: Optional[float] = None
    ) -> Dict[str, float]:
        """Calcule les poids optimaux pour la fusion."""
        # Poids par défaut
        final_lexical = lexical_weight or self.config.default_lexical_weight
        final_semantic = semantic_weight or self.config.default_semantic_weight
        
        # Adaptation intelligente si activée
        if self.config.adaptive_weighting and lexical_weight is None and semantic_weight is None:
            query_type = query_analysis.query_type
            entities = query_analysis.detected_entities
            
            # Ajustements basés sur le type de requête
            if query_type in ["amount_search", "date_search"]:
                # Entités spécifiques = favoriser lexical
                final_lexical = 0.8
                final_semantic = 0.2
            elif query_type in ["category_search", "free_text"]:
                # Recherche conceptuelle = favoriser sémantique
                final_lexical = 0.4
                final_semantic = 0.6
            elif query_type == "merchant_query":
                # Marchands = lexical prioritaire (noms exacts)
                final_lexical = 0.7
                final_semantic = 0.3
            
            # Ajustements basés sur les entités détectées
            if entities.get("amounts") or entities.get("dates"):
                # Présence d'entités = boost lexical
                final_lexical = min(final_lexical + 0.1, 0.9)
                final_semantic = 1.0 - final_lexical
            
            if entities.get("categories"):
                # Catégories = boost sémantique
                final_semantic = min(final_semantic + 0.1, 0.9)
                final_lexical = 1.0 - final_semantic
        
        # Normaliser pour que la somme = 1.0
        total = final_lexical + final_semantic
        if total > 0:
            final_lexical /= total
            final_semantic /= total
        else:
            final_lexical = 0.5
            final_semantic = 0.5
        
        return {
            "lexical": final_lexical,
            "semantic": final_semantic
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
        """Exécute la stratégie de recherche choisie."""
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
                    # Combiner avec résultat lexical existant
                    existing = all_results[transaction_id]
                    existing["semantic_score"] = normalized_score
                    existing["combined_score"] += weighted_score
                    existing["sources"].append("semantic")
                    
                    # Mettre à jour l'item avec les scores combinés
                    existing["item"].semantic_score = normalized_score
                    existing["item"].combined_score = existing["combined_score"]
                else:
                    # Nouveau résultat sémantique uniquement
                    all_results[transaction_id] = {
                        "item": result,
                        "lexical_score": 0.0,
                        "semantic_score": normalized_score,
                        "combined_score": weighted_score,
                        "sources": ["semantic"]
                    }
        
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
        
        if search_type == SearchType.LEXICAL and lexical_result:
            return lexical_result.quality
        elif search_type == SearchType.SEMANTIC and semantic_result:
            return semantic_result.quality
        elif search_type == SearchType.HYBRID:
            # Qualité hybride = meilleure des deux avec bonus si convergence
            qualities = []
            
            if lexical_result:
                qualities.append(lexical_result.quality)
            if semantic_result:
                qualities.append(semantic_result.quality)
            
            if not qualities:
                return SearchQuality.FAILED
            
            # Convertir en scores numériques
            quality_scores = {
                SearchQuality.EXCELLENT: 5,
                SearchQuality.GOOD: 4,
                SearchQuality.MEDIUM: 3,
                SearchQuality.POOR: 2,
                SearchQuality.FAILED: 1
            }
            
            numeric_qualities = [quality_scores[q] for q in qualities]
            avg_quality = sum(numeric_qualities) / len(numeric_qualities)
            
            # Bonus si les deux moteurs ont fourni des résultats
            if len(qualities) == 2 and all(q != SearchQuality.FAILED for q in qualities):
                avg_quality += 0.5  # Bonus convergence
            
            # Reconvertir en enum
            if avg_quality >= 4.5:
                return SearchQuality.EXCELLENT
            elif avg_quality >= 3.5:
                return SearchQuality.GOOD
            elif avg_quality >= 2.5:
                return SearchQuality.MEDIUM
            elif avg_quality >= 1.5:
                return SearchQuality.POOR
            else:
                return SearchQuality.FAILED
        
        return SearchQuality.FAILED
    
    def _generate_cache_key(self, *args) -> str:
        """Génère une clé de cache unique."""
        import hashlib
        key_string = "|".join(str(arg) for arg in args)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _update_metrics(self, result: HybridSearchResult):
        """Met à jour les métriques internes."""
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
        avg