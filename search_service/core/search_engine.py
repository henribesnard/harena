"""
Moteur de recherche hybride principal - Chef d'orchestre.

Ce module coordonne les recherches lexicale et sémantique pour fournir
des résultats optimaux via différentes stratégies de fusion intelligentes.
"""
import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

from search_service.core.lexical_engine import LexicalSearchEngine, LexicalSearchResult
from search_service.core.semantic_engine import SemanticSearchEngine, SemanticSearchResult
from search_service.core.result_merger import ResultMerger, FusionStrategy, FusionConfig, FusionResult
from search_service.core.query_processor import QueryProcessor, QueryAnalysis
from search_service.models.search_types import SearchType, SearchQuality, SortOrder
from search_service.models.responses import SearchResultItem, SearchResponse
from search_service.utils.cache import SearchCache

logger = logging.getLogger(__name__)


@dataclass
class HybridSearchConfig:
    """Configuration pour la recherche hybride."""
    # Poids par défaut pour la fusion
    default_lexical_weight: float = 0.6
    default_semantic_weight: float = 0.4
    
    # Stratégie de fusion par défaut
    fusion_strategy: FusionStrategy = FusionStrategy.ADAPTIVE_FUSION
    
    # Paramètres de qualité et performance
    min_results_for_fusion: int = 2
    max_results_per_engine: int = 50
    
    # Cache
    enable_cache: bool = True
    cache_ttl_seconds: int = 300
    
    # Adaptation automatique des poids
    adaptive_weighting: bool = True
    quality_boost_factor: float = 0.2
    
    # Timeouts
    search_timeout: float = 15.0
    lexical_timeout: float = 5.0
    semantic_timeout: float = 8.0
    
    # Fallback et resilience
    enable_fallback: bool = True
    min_engine_success: int = 1  # Au moins 1 moteur doit réussir
    
    # Optimisations
    enable_parallel_search: bool = True
    enable_early_termination: bool = True
    early_termination_threshold: float = 0.9


@dataclass
class HybridSearchResult:
    """Résultat d'une recherche hybride complète."""
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
    
    # Détails des moteurs
    lexical_result: Optional[LexicalSearchResult] = None
    semantic_result: Optional[SemanticSearchResult] = None
    fusion_result: Optional[FusionResult] = None
    
    # Debug et métriques
    debug_info: Optional[Dict[str, Any]] = None
    performance_metrics: Optional[Dict[str, Any]] = None


class HybridSearchEngine:
    """
    Moteur de recherche hybride - Chef d'orchestre.
    
    Coordonne les moteurs lexical et sémantique pour fournir
    des résultats optimaux selon différentes stratégies.
    
    Responsabilités:
    - Orchestration des recherches parallèles
    - Fusion intelligente des résultats
    - Gestion du cache global
    - Adaptation des poids selon la qualité
    - Métriques de performance globales
    - Fallback en cas d'erreur partielle
    """
    
    def __init__(
        self,
        lexical_engine: Optional[LexicalSearchEngine] = None,
        semantic_engine: Optional[SemanticSearchEngine] = None,
        query_processor: Optional[QueryProcessor] = None,
        result_merger: Optional[ResultMerger] = None,
        config: Optional[HybridSearchConfig] = None
    ):
        self.lexical_engine = lexical_engine
        self.semantic_engine = semantic_engine
        self.query_processor = query_processor or QueryProcessor()
        self.result_merger = result_merger or ResultMerger()
        self.config = config or HybridSearchConfig()
        
        # Cache de résultats hybrides
        self.cache = SearchCache(
            max_size=1000,
            ttl_seconds=self.config.cache_ttl_seconds
        ) if self.config.enable_cache else None
        
        # Métriques globales
        self.search_count = 0
        self.cache_hits = 0
        self.total_processing_time = 0.0
        self.engine_failures = {"lexical": 0, "semantic": 0, "both": 0}
        self.search_type_usage = {search_type.value: 0 for search_type in SearchType}
        self.fusion_stats = {strategy.value: 0 for strategy in FusionStrategy}
        self.quality_distribution = {quality.value: 0 for quality in SearchQuality}
        self.performance_stats = {
            "fast_searches": 0,  # < 2s
            "normal_searches": 0,  # 2-5s
            "slow_searches": 0,  # > 5s
            "timeout_searches": 0
        }
        
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
            search_type: Type de recherche (LEXICAL, SEMANTIC, HYBRID)
            limit: Nombre de résultats
            offset: Décalage pour pagination
            lexical_weight: Poids pour la recherche lexicale
            semantic_weight: Poids pour la recherche sémantique
            similarity_threshold: Seuil de similarité pour recherche sémantique
            sort_order: Ordre de tri des résultats
            filters: Filtres additionnels
            use_cache: Utiliser le cache si disponible
            debug: Inclure informations de debug
            
        Returns:
            Résultats de recherche hybride
            
        Raises:
            Exception: Si tous les moteurs échouent
        """
        start_time = time.time()
        self.search_count += 1
        
        # Validation des paramètres
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        if search_type not in SearchType:
            raise ValueError(f"Invalid search type: {search_type}")
        
        # Génération de la clé de cache
        cache_key = None
        if use_cache and self.cache:
            cache_key = self._generate_cache_key(
                query, user_id, search_type, limit, offset, 
                lexical_weight, semantic_weight, similarity_threshold,
                sort_order, filters
            )
            
            # Vérifier le cache
            cached_result = self.cache.get(cache_key)
            if cached_result:
                self.cache_hits += 1
                cached_result.cache_hit = True
                logger.debug(f"Cache hit for hybrid search: {cache_key[:16]}...")
                return cached_result
        
        try:
            # Analyser la requête
            query_analysis = self.query_processor.process_query(query)
            
            # Déterminer les poids optimaux
            weights = self._determine_search_weights(
                query_analysis, lexical_weight, semantic_weight, search_type
            )
            
            # Exécuter la recherche selon le type
            result = await asyncio.wait_for(
                self._execute_search_strategy(
                    query, user_id, search_type, query_analysis, weights,
                    limit, offset, similarity_threshold, sort_order, filters, debug
                ),
                timeout=self.config.search_timeout
            )
            
            # Calculer les métriques finales
            processing_time = (time.time() - start_time) * 1000
            self.total_processing_time += processing_time
            result.processing_time_ms = processing_time
            
            # Mettre à jour les statistiques
            self._update_search_statistics(result, processing_time)
            
            # Mettre en cache si activé
            if use_cache and self.cache and cache_key:
                self.cache.put(cache_key, result)
            
            return result
            
        except asyncio.TimeoutError:
            self.performance_stats["timeout_searches"] += 1
            logger.error(f"Search timeout after {self.config.search_timeout}s")
            raise Exception("Search timeout")
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}", exc_info=True)
            raise Exception(f"Search error: {str(e)}")
    
    async def _execute_search_strategy(
        self,
        query: str,
        user_id: int,
        search_type: SearchType,
        query_analysis: QueryAnalysis,
        weights: Dict[str, float],
        limit: int,
        offset: int,
        similarity_threshold: Optional[float],
        sort_order: SortOrder,
        filters: Optional[Dict[str, Any]],
        debug: bool
    ) -> HybridSearchResult:
        """Exécute la stratégie de recherche selon le type."""
        
        if search_type == SearchType.LEXICAL:
            return await self._execute_lexical_only(
                query, user_id, limit, offset, sort_order, filters, debug
            )
        elif search_type == SearchType.SEMANTIC:
            return await self._execute_semantic_only(
                query, user_id, limit, offset, similarity_threshold, sort_order, filters, debug
            )
        else:  # HYBRID
            return await self._execute_hybrid_search(
                query, user_id, query_analysis, weights, limit, offset, 
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
    ) -> HybridSearchResult:
        """Exécute une recherche lexicale uniquement."""
        if not self.lexical_engine:
            raise Exception("Lexical engine not available")
        
        try:
            lexical_result = await asyncio.wait_for(
                self.lexical_engine.search(
                    query=query,
                    user_id=user_id,
                    limit=limit,
                    offset=offset,
                    sort_order=sort_order,
                    filters=filters,
                    debug=debug
                ),
                timeout=self.config.lexical_timeout
            )
            
            return HybridSearchResult(
                results=lexical_result.results,
                total_found=lexical_result.total_found,
                search_type=SearchType.LEXICAL,
                lexical_results_count=len(lexical_result.results),
                semantic_results_count=0,
                fusion_strategy="lexical_only",
                weights_used={"lexical_weight": 1.0, "semantic_weight": 0.0},
                processing_time_ms=lexical_result.processing_time_ms,
                quality=lexical_result.quality,
                lexical_result=lexical_result,
                debug_info={"strategy": "lexical_only"} if debug else None
            )
            
        except Exception as e:
            self.engine_failures["lexical"] += 1
            logger.error(f"Lexical-only search failed: {e}")
            raise Exception("Lexical search failed")
    
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
    ) -> HybridSearchResult:
        """Exécute une recherche sémantique uniquement."""
        if not self.semantic_engine:
            raise Exception("Semantic engine not available")
        
        try:
            semantic_result = await asyncio.wait_for(
                self.semantic_engine.search(
                    query=query,
                    user_id=user_id,
                    limit=limit,
                    offset=offset,
                    similarity_threshold=similarity_threshold,
                    sort_order=sort_order,
                    filters=filters,
                    debug=debug
                ),
                timeout=self.config.semantic_timeout
            )
            
            return HybridSearchResult(
                results=semantic_result.results,
                total_found=semantic_result.total_found,
                search_type=SearchType.SEMANTIC,
                lexical_results_count=0,
                semantic_results_count=len(semantic_result.results),
                fusion_strategy="semantic_only",
                weights_used={"lexical_weight": 0.0, "semantic_weight": 1.0},
                processing_time_ms=semantic_result.processing_time_ms,
                quality=semantic_result.quality,
                semantic_result=semantic_result,
                debug_info={"strategy": "semantic_only"} if debug else None
            )
            
        except Exception as e:
            self.engine_failures["semantic"] += 1
            logger.error(f"Semantic-only search failed: {e}")
            raise Exception("Semantic search failed")
    
    async def _execute_hybrid_search(
        self,
        query: str,
        user_id: int,
        query_analysis: QueryAnalysis,
        weights: Dict[str, float],
        limit: int,
        offset: int,
        similarity_threshold: Optional[float],
        sort_order: SortOrder,
        filters: Optional[Dict[str, Any]],
        debug: bool
    ) -> HybridSearchResult:
        """Exécute une recherche hybride avec fusion."""
        if not self.lexical_engine or not self.semantic_engine:
            raise Exception("Both lexical and semantic engines required for hybrid search")
        
        # Limite élargie pour meilleure fusion
        expanded_limit = min(limit + offset + 10, self.config.max_results_per_engine)
        
        # Créer les tâches de recherche
        lexical_task = self._safe_lexical_search(
            query, user_id, expanded_limit, 0, SortOrder.RELEVANCE, filters, debug
        )
        
        semantic_task = self._safe_semantic_search(
            query, user_id, expanded_limit, 0, similarity_threshold, 
            SortOrder.RELEVANCE, filters, debug
        )
        
        # Exécuter les recherches
        if self.config.enable_parallel_search:
            # Recherches en parallèle
            lexical_result, semantic_result = await asyncio.gather(
                lexical_task, semantic_task, return_exceptions=True
            )
        else:
            # Recherches séquentielles
            lexical_result = await lexical_task
            semantic_result = await semantic_task
        
        # Traiter les résultats et exceptions
        lexical_success = not isinstance(lexical_result, Exception)
        semantic_success = not isinstance(semantic_result, Exception)
        
        if isinstance(lexical_result, Exception):
            logger.warning(f"Lexical search failed in hybrid: {lexical_result}")
            lexical_result = None
            self.engine_failures["lexical"] += 1
        
        if isinstance(semantic_result, Exception):
            logger.warning(f"Semantic search failed in hybrid: {semantic_result}")
            semantic_result = None
            self.engine_failures["semantic"] += 1
        
        # Vérifier qu'au moins un moteur a réussi
        successful_engines = sum([lexical_success, semantic_success])
        
        if successful_engines < self.config.min_engine_success:
            self.engine_failures["both"] += 1
            raise Exception("All search engines failed")
        
        # Fallback si un seul moteur a réussi
        if not lexical_result and semantic_result:
            return self._create_fallback_result(
                semantic_result, SearchType.SEMANTIC, "semantic_fallback", weights, debug
            )
        
        if not semantic_result and lexical_result:
            return self._create_fallback_result(
                lexical_result, SearchType.LEXICAL, "lexical_fallback", weights, debug
            )
        
        # Fusion des résultats
        try:
            fusion_result = self.result_merger.fuse_results(
                lexical_result=lexical_result,
                semantic_result=semantic_result,
                query_analysis=query_analysis,
                strategy=self.config.fusion_strategy,
                weights=weights,
                limit=limit,
                offset=offset,
                sort_order=sort_order,
                debug=debug
            )
            
            # Mise à jour des statistiques de fusion
            self.fusion_stats[fusion_result.fusion_strategy] += 1
            
            return HybridSearchResult(
                results=fusion_result.results,
                total_found=fusion_result.lexical_count + fusion_result.semantic_count,
                search_type=SearchType.HYBRID,
                lexical_results_count=fusion_result.lexical_count,
                semantic_results_count=fusion_result.semantic_count,
                fusion_strategy=fusion_result.fusion_strategy,
                weights_used=fusion_result.weights_used,
                processing_time_ms=0.0,  # Sera calculé plus tard
                quality=fusion_result.quality,
                lexical_result=lexical_result,
                semantic_result=semantic_result,
                fusion_result=fusion_result,
                debug_info=self._create_hybrid_debug_info(
                    lexical_result, semantic_result, fusion_result
                ) if debug else None
            )
            
        except Exception as e:
            logger.error(f"Result fusion failed: {e}")
            
            # Fallback vers le meilleur résultat individuel
            if lexical_result and semantic_result:
                if lexical_result.quality.value >= semantic_result.quality.value:
                    return self._create_fallback_result(
                        lexical_result, SearchType.LEXICAL, "fusion_failed_lexical", weights, debug
                    )
                else:
                    return self._create_fallback_result(
                        semantic_result, SearchType.SEMANTIC, "fusion_failed_semantic", weights, debug
                    )
            
            raise Exception("Hybrid search and fusion failed")
    
    async def _safe_lexical_search(
        self,
        query: str,
        user_id: int,
        limit: int,
        offset: int,
        sort_order: SortOrder,
        filters: Optional[Dict[str, Any]],
        debug: bool
    ) -> Optional[LexicalSearchResult]:
        """Recherche lexicale sécurisée avec timeout."""
        try:
            return await asyncio.wait_for(
                self.lexical_engine.search(
                    query=query,
                    user_id=user_id,
                    limit=limit,
                    offset=offset,
                    sort_order=sort_order,
                    filters=filters,
                    debug=debug
                ),
                timeout=self.config.lexical_timeout
            )
        except Exception as e:
            logger.warning(f"Safe lexical search failed: {e}")
            raise e
    
    async def _safe_semantic_search(
        self,
        query: str,
        user_id: int,
        limit: int,
        offset: int,
        similarity_threshold: Optional[float],
        sort_order: SortOrder,
        filters: Optional[Dict[str, Any]],
        debug: bool
    ) -> Optional[SemanticSearchResult]:
        """Recherche sémantique sécurisée avec timeout."""
        try:
            return await asyncio.wait_for(
                self.semantic_engine.search(
                    query=query,
                    user_id=user_id,
                    limit=limit,
                    offset=offset,
                    similarity_threshold=similarity_threshold,
                    sort_order=sort_order,
                    filters=filters,
                    debug=debug
                ),
                timeout=self.config.semantic_timeout
            )
        except Exception as e:
            logger.warning(f"Safe semantic search failed: {e}")
            raise e
    
    def _determine_search_weights(
        self,
        query_analysis: QueryAnalysis,
        lexical_weight: Optional[float],
        semantic_weight: Optional[float],
        search_type: SearchType
    ) -> Dict[str, float]:
        """Détermine les poids optimaux pour la recherche."""
        
        # Si les poids sont fournis explicitement
        if lexical_weight is not None and semantic_weight is not None:
            # Normaliser pour que la somme = 1.0
            total = lexical_weight + semantic_weight
            if total > 0:
                return {
                    "lexical_weight": lexical_weight / total,
                    "semantic_weight": semantic_weight / total
                }
        
        # Pour recherche non-hybride
        if search_type == SearchType.LEXICAL:
            return {"lexical_weight": 1.0, "semantic_weight": 0.0}
        elif search_type == SearchType.SEMANTIC:
            return {"lexical_weight": 0.0, "semantic_weight": 1.0}
        
        # Adaptation automatique pour recherche hybride
        if self.config.adaptive_weighting:
            return self._calculate_adaptive_weights(query_analysis)
        
        # Poids par défaut
        return {
            "lexical_weight": self.config.default_lexical_weight,
            "semantic_weight": self.config.default_semantic_weight
        }
    
    def _calculate_adaptive_weights(self, query_analysis: QueryAnalysis) -> Dict[str, float]:
        """Calcule les poids adaptatifs basés sur l'analyse de requête."""
        
        base_lexical = self.config.default_lexical_weight
        base_semantic = self.config.default_semantic_weight
        
        # Facteurs d'ajustement
        lexical_boost = 0.0
        semantic_boost = 0.0
        
        # CORRECTION: Utiliser les propriétés correctes de QueryAnalysis
        
        # Requêtes avec phrases exactes -> favoriser lexical
        if query_analysis.has_exact_phrases:
            lexical_boost += 0.2
        
        # Requêtes courtes et spécifiques -> favoriser lexical
        if len(query_analysis.key_terms) <= 2:
            lexical_boost += 0.1
        
        # Requêtes avec entités financières -> équilibrer
        if query_analysis.has_financial_entities:
            # Rééquilibrer vers 50/50
            target_lexical = 0.5
            target_semantic = 0.5
            base_lexical = (base_lexical + target_lexical) / 2
            base_semantic = (base_semantic + target_semantic) / 2
        
        # Requêtes longues et complexes -> favoriser sémantique
        if len(query_analysis.key_terms) > 5:
            semantic_boost += 0.15
        
        # Requêtes interrogatives -> favoriser sémantique
        if query_analysis.is_question:
            semantic_boost += 0.1
        
        # Appliquer les ajustements
        lexical_weight = max(0.1, min(0.9, base_lexical + lexical_boost - semantic_boost))
        semantic_weight = 1.0 - lexical_weight
        
        return {
            "lexical_weight": lexical_weight,
            "semantic_weight": semantic_weight
        }
    
    def _create_fallback_result(
        self,
        single_result: Union[LexicalSearchResult, SemanticSearchResult],
        search_type: SearchType,
        strategy: str,
        weights: Dict[str, float],
        debug: bool
    ) -> HybridSearchResult:
        """Crée un résultat de fallback à partir d'un seul moteur."""
        
        if isinstance(single_result, LexicalSearchResult):
            lexical_count = len(single_result.results)
            semantic_count = 0
            lexical_result = single_result
            semantic_result = None
        else:
            lexical_count = 0
            semantic_count = len(single_result.results)
            lexical_result = None
            semantic_result = single_result
        
        return HybridSearchResult(
            results=single_result.results,
            total_found=single_result.total_found,
            search_type=search_type,
            lexical_results_count=lexical_count,
            semantic_results_count=semantic_count,
            fusion_strategy=strategy,
            weights_used=weights,
            processing_time_ms=single_result.processing_time_ms,
            quality=single_result.quality,
            lexical_result=lexical_result,
            semantic_result=semantic_result,
            debug_info={"fallback": True, "strategy": strategy} if debug else None
        )
    
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
        import hashlib
        
        cache_data = {
            "query": query.lower().strip(),
            "user_id": user_id,
            "search_type": search_type.value,
            "limit": limit,
            "offset": offset,
            "lexical_weight": lexical_weight,
            "semantic_weight": semantic_weight,
            "similarity_threshold": similarity_threshold,
            "sort_order": sort_order.value,
            "filters": filters or {}
        }
        
        cache_str = str(sorted(cache_data.items()))
        return f"hybrid_{hashlib.md5(cache_str.encode()).hexdigest()}"
    
    def _update_search_statistics(self, result: HybridSearchResult, processing_time: float) -> None:
        """Met à jour les statistiques de recherche."""
        
        # Statistiques par type de recherche
        self.search_type_usage[result.search_type.value] += 1
        
        # Statistiques de qualité
        self.quality_distribution[result.quality.value] += 1
        
        # Statistiques de performance
        if processing_time < 2000:  # < 2s
            self.performance_stats["fast_searches"] += 1
        elif processing_time < 5000:  # 2-5s
            self.performance_stats["normal_searches"] += 1
        else:  # > 5s
            self.performance_stats["slow_searches"] += 1
    
    def _create_hybrid_debug_info(
        self,
        lexical_result: Optional[LexicalSearchResult],
        semantic_result: Optional[SemanticSearchResult],
        fusion_result: FusionResult
    ) -> Dict[str, Any]:
        """Crée les informations de debug pour recherche hybride."""
        
        debug_info = {
            "strategy": "hybrid",
            "fusion_strategy": fusion_result.fusion_strategy,
            "engines_used": {
                "lexical": lexical_result is not None,
                "semantic": semantic_result is not None
            },
            "result_counts": {
                "lexical": len(lexical_result.results) if lexical_result else 0,
                "semantic": len(semantic_result.results) if semantic_result else 0,
                "fused": len(fusion_result.results),
                "duplicates_removed": fusion_result.duplicates_removed
            },
            "quality_scores": {
                "lexical": lexical_result.quality.value if lexical_result else None,
                "semantic": semantic_result.quality.value if semantic_result else None,
                "fused": fusion_result.quality.value
            },
            "processing_times": {
                "lexical": lexical_result.processing_time_ms if lexical_result else None,
                "semantic": semantic_result.processing_time_ms if semantic_result else None,
                "fusion": fusion_result.processing_time_ms
            }
        }
        
        # Ajouter les infos de debug de la fusion si disponibles
        if fusion_result.debug_info:
            debug_info["fusion_debug"] = fusion_result.debug_info
        
        return debug_info
    
    async def health_check(self) -> Dict[str, Any]:
        """Vérifie la santé du moteur hybride."""
        health_status = {
            "status": "healthy",
            "engines": {},
            "metrics": self.get_metrics()
        }
        
        # Vérifier les moteurs individuels
        if self.lexical_engine:
            try:
                lexical_health = await self.lexical_engine.health_check()
                health_status["engines"]["lexical"] = lexical_health
            except Exception as e:
                health_status["engines"]["lexical"] = {"status": "unhealthy", "error": str(e)}
                health_status["status"] = "degraded"
        else:
            health_status["engines"]["lexical"] = {"status": "not_configured"}
        
        if self.semantic_engine:
            try:
                semantic_health = await self.semantic_engine.health_check()
                health_status["engines"]["semantic"] = semantic_health
            except Exception as e:
                health_status["engines"]["semantic"] = {"status": "unhealthy", "error": str(e)}
                health_status["status"] = "degraded"
        else:
            health_status["engines"]["semantic"] = {"status": "not_configured"}
        
        # Vérifier qu'au moins un moteur est disponible
        available_engines = sum(
            1 for engine_health in health_status["engines"].values()
            if engine_health.get("status") == "healthy"
        )
        
        if available_engines == 0:
            health_status["status"] = "unhealthy"
        
        return health_status
    
    def get_metrics(self) -> Dict[str, Any]:
        """Retourne les métriques du moteur hybride."""
        avg_processing_time = (
            self.total_processing_time / self.search_count
            if self.search_count > 0 else 0
        )
        
        cache_hit_rate = self.cache_hits / self.search_count if self.search_count > 0 else 0
        
        # Calculer les taux d'échec
        total_failures = sum(self.engine_failures.values())
        failure_rate = total_failures / self.search_count if self.search_count > 0 else 0
        
        return {
            "engine_type": "hybrid",
            "search_count": self.search_count,
            "total_processing_time_ms": self.total_processing_time,
            "average_processing_time_ms": avg_processing_time,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": cache_hit_rate,
            
            "engine_failures": self.engine_failures,
            "total_failures": total_failures,
            "failure_rate": failure_rate,
            
            "search_type_usage": self.search_type_usage,
            "fusion_strategy_usage": self.fusion_stats,
            "quality_distribution": self.quality_distribution,
            "performance_distribution": self.performance_stats,
            
            "cache_stats": self.cache.get_stats() if self.cache else None,
            
            "engine_metrics": {
                "lexical": self.lexical_engine.get_metrics() if self.lexical_engine else None,
                "semantic": self.semantic_engine.get_metrics() if self.semantic_engine else None,
                "result_merger": self.result_merger.get_metrics()
            }
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Retourne un résumé des performances."""
        total_searches = sum(self.performance_stats.values())
        
        if total_searches == 0:
            return {"message": "No searches performed yet"}
        
        return {
            "total_searches": total_searches,
            "performance_breakdown": {
                "fast_searches_pct": (self.performance_stats["fast_searches"] / total_searches) * 100,
                "normal_searches_pct": (self.performance_stats["normal_searches"] / total_searches) * 100,
                "slow_searches_pct": (self.performance_stats["slow_searches"] / total_searches) * 100,
                "timeout_searches_pct": (self.performance_stats["timeout_searches"] / total_searches) * 100
            },
            "average_response_time_ms": self.total_processing_time / self.search_count,
            "cache_efficiency": (self.cache_hits / self.search_count) * 100,
            "reliability": {
                "success_rate": ((self.search_count - sum(self.engine_failures.values())) / self.search_count) * 100,
                "lexical_failure_rate": (self.engine_failures["lexical"] / self.search_count) * 100,
                "semantic_failure_rate": (self.engine_failures["semantic"] / self.search_count) * 100,
                "both_engines_failure_rate": (self.engine_failures["both"] / self.search_count) * 100
            }
        }
    
    async def suggest_search_improvements(self, query: str, user_id: int) -> Dict[str, Any]:
        """Suggère des améliorations pour une requête de recherche."""
        
        # Analyser la requête
        query_analysis = self.query_processor.process_query(query)
        
        suggestions = {
            "original_query": query,
            "analysis": {
                "key_terms": query_analysis.key_terms,
                "has_exact_phrases": query_analysis.has_exact_phrases,
                "has_financial_entities": query_analysis.has_financial_entities,
                "is_question": query_analysis.is_question
            },
            "recommendations": []
        }
        
        # Suggestions basées sur l'analyse
        if len(query_analysis.key_terms) == 1:
            suggestions["recommendations"].append({
                "type": "expand_query",
                "message": "Votre requête est très courte. Ajoutez des mots-clés pour des résultats plus précis.",
                "example": f"{query} montant date"
            })
        
        if len(query_analysis.key_terms) > 8:
            suggestions["recommendations"].append({
                "type": "simplify_query",
                "message": "Votre requête est très longue. Simplifiez-la pour de meilleurs résultats.",
                "example": " ".join(query_analysis.key_terms[:4])
            })
        
        if not query_analysis.has_financial_entities and not query_analysis.has_exact_phrases:
            suggestions["recommendations"].append({
                "type": "add_context",
                "message": "Ajoutez du contexte financier pour améliorer la précision.",
                "examples": [
                    f"{query} transaction",
                    f"{query} paiement",
                    f"{query} virement"
                ]
            })
        
        # Suggestions de recherche alternatives
        alternative_searches = []
        
        # Recherche par montant si pas spécifié
        if not any(term.replace("€", "").replace(",", "").replace(".", "").isdigit() for term in query_analysis.key_terms):
            alternative_searches.append({
                "type": "amount_search",
                "suggestion": f"{query} montant supérieur 100",
                "description": "Rechercher par montant"
            })
        
        # Recherche par période si pas spécifiée
        if not any(word in query.lower() for word in ["mois", "semaine", "jour", "janvier", "février", "mars"]):
            alternative_searches.append({
                "type": "time_search",
                "suggestion": f"{query} ce mois",
                "description": "Limiter à une période"
            })
        
        suggestions["alternative_searches"] = alternative_searches
        
        # Recommandations de type de recherche
        recommended_search_type = self._recommend_search_type(query_analysis)
        suggestions["recommended_search_type"] = {
            "type": recommended_search_type.value,
            "reason": self._explain_search_type_recommendation(query_analysis, recommended_search_type)
        }
        
        return suggestions
    
    def _recommend_search_type(self, query_analysis: QueryAnalysis) -> SearchType:
        """Recommande le type de recherche optimal."""
        
        # Recherche lexicale pour phrases exactes
        if query_analysis.has_exact_phrases:
            return SearchType.LEXICAL
        
        # Recherche sémantique pour questions complexes
        if query_analysis.is_question and len(query_analysis.key_terms) > 3:
            return SearchType.SEMANTIC
        
        # Recherche hybride par défaut
        return SearchType.HYBRID
    
    def _explain_search_type_recommendation(
        self, 
        query_analysis: QueryAnalysis, 
        recommended_type: SearchType
    ) -> str:
        """Explique pourquoi ce type de recherche est recommandé."""
        
        if recommended_type == SearchType.LEXICAL:
            if query_analysis.has_exact_phrases:
                return "Votre requête contient des phrases exactes, la recherche lexicale sera plus précise."
            return "Recherche lexicale recommandée pour cette requête spécifique."
        
        elif recommended_type == SearchType.SEMANTIC:
            if query_analysis.is_question:
                return "Votre question nécessite une compréhension contextuelle, la recherche sémantique est optimale."
            return "Recherche sémantique recommandée pour capturer le sens de votre requête."
        
        else:  # HYBRID
            return "La recherche hybride combine les avantages lexical et sémantique pour des résultats optimaux."
    
    async def benchmark_search_types(
        self, 
        query: str, 
        user_id: int,
        iterations: int = 3
    ) -> Dict[str, Any]:
        """Compare les performances des différents types de recherche."""
        
        benchmark_results = {
            "query": query,
            "iterations": iterations,
            "results": {}
        }
        
        search_types = [SearchType.LEXICAL, SearchType.SEMANTIC, SearchType.HYBRID]
        
        for search_type in search_types:
            type_results = []
            
            for i in range(iterations):
                try:
                    start_time = time.time()
                    
                    result = await self.search(
                        query=query,
                        user_id=user_id,
                        search_type=search_type,
                        limit=10,
                        use_cache=False  # Désactiver le cache pour mesures précises
                    )
                    
                    processing_time = (time.time() - start_time) * 1000
                    
                    type_results.append({
                        "iteration": i + 1,
                        "processing_time_ms": processing_time,
                        "results_count": len(result.results),
                        "quality": result.quality.value,
                        "total_found": result.total_found
                    })
                    
                except Exception as e:
                    type_results.append({
                        "iteration": i + 1,
                        "error": str(e),
                        "processing_time_ms": None,
                        "results_count": 0,
                        "quality": "failed"
                    })
            
            # Calculer les statistiques moyennes
            successful_runs = [r for r in type_results if "error" not in r]
            
            if successful_runs:
                avg_time = sum(r["processing_time_ms"] for r in successful_runs) / len(successful_runs)
                avg_results = sum(r["results_count"] for r in successful_runs) / len(successful_runs)
                
                benchmark_results["results"][search_type.value] = {
                    "successful_iterations": len(successful_runs),
                    "failed_iterations": len(type_results) - len(successful_runs),
                    "average_processing_time_ms": avg_time,
                    "average_results_count": avg_results,
                    "success_rate": len(successful_runs) / iterations,
                    "detailed_results": type_results
                }
            else:
                benchmark_results["results"][search_type.value] = {
                    "successful_iterations": 0,
                    "failed_iterations": iterations,
                    "success_rate": 0.0,
                    "detailed_results": type_results
                }
        
        # Déterminer le meilleur type
        best_type = None
        best_score = 0
        
        for search_type, stats in benchmark_results["results"].items():
            if stats["successful_iterations"] > 0:
                # Score composite basé sur succès, vitesse et résultats
                speed_score = 1000 / stats.get("average_processing_time_ms", 1000)  # Plus rapide = meilleur
                results_score = stats.get("average_results_count", 0)
                success_score = stats["success_rate"] * 100
                
                composite_score = speed_score + results_score + success_score
                
                if composite_score > best_score:
                    best_score = composite_score
                    best_type = search_type
        
        benchmark_results["recommendation"] = {
            "best_search_type": best_type,
            "score": best_score,
            "reason": f"Meilleur équilibre performance/résultats/fiabilité" if best_type else "Aucun type n'a réussi"
        }
        
        return benchmark_results
    
    def clear_cache(self) -> None:
        """Vide le cache du moteur hybride."""
        if self.cache:
            self.cache.clear()
            logger.info("Hybrid engine cache cleared")
        
        # Vider aussi les caches des moteurs individuels
        if self.lexical_engine:
            self.lexical_engine.clear_cache()
        
        if self.semantic_engine:
            self.semantic_engine.clear_cache()
    
    def reset_metrics(self) -> None:
        """Remet à zéro toutes les métriques."""
        self.search_count = 0
        self.cache_hits = 0
        self.total_processing_time = 0.0
        self.engine_failures = {"lexical": 0, "semantic": 0, "both": 0}
        self.search_type_usage = {search_type.value: 0 for search_type in SearchType}
        self.fusion_stats = {strategy.value: 0 for strategy in FusionStrategy}
        self.quality_distribution = {quality.value: 0 for quality in SearchQuality}
        self.performance_stats = {
            "fast_searches": 0,
            "normal_searches": 0,
            "slow_searches": 0,
            "timeout_searches": 0
        }
        
        # Reset des métriques des moteurs individuels
        if self.lexical_engine:
            self.lexical_engine.reset_metrics()
        
        if self.semantic_engine:
            self.semantic_engine.reset_metrics()
        
        self.result_merger.reset_metrics()
        
        logger.info("All hybrid engine metrics reset")
    
    def update_config(self, new_config: HybridSearchConfig) -> None:
        """Met à jour la configuration du moteur hybride."""
        old_config = self.config
        self.config = new_config
        
        # Recréer le cache si les paramètres ont changé
        if (old_config.cache_ttl_seconds != new_config.cache_ttl_seconds or
            old_config.enable_cache != new_config.enable_cache):
            
            if new_config.enable_cache:
                self.cache = SearchCache(
                    max_size=1000,
                    ttl_seconds=new_config.cache_ttl_seconds
                )
            else:
                self.cache = None
        
        logger.info("Hybrid engine configuration updated")
    
    async def warmup(self, warmup_queries: Optional[List[str]] = None) -> Dict[str, Any]:
        """Réchauffe le moteur avec des requêtes prédéfinies."""
        
        default_warmup_queries = [
            "restaurant",
            "supermarché", 
            "essence",
            "virement bancaire",
            "carte bleue",
            "pharmacie",
            "transport",
            "salaire"
        ]
        
        queries = warmup_queries or default_warmup_queries
        
        warmup_results = {
            "queries_count": len(queries),
            "successful_warmups": 0,
            "failed_warmups": 0,
            "total_time_ms": 0.0,
            "details": []
        }
        
        logger.info(f"Starting warmup with {len(queries)} queries")
        
        for query in queries:
            try:
                start_time = time.time()
                
                # Test avec utilisateur par défaut
                result = await self.search(
                    query=query,
                    user_id=1,
                    search_type=SearchType.HYBRID,
                    limit=5,
                    use_cache=True
                )
                
                processing_time = (time.time() - start_time) * 1000
                warmup_results["total_time_ms"] += processing_time
                warmup_results["successful_warmups"] += 1
                
                warmup_results["details"].append({
                    "query": query,
                    "success": True,
                    "processing_time_ms": processing_time,
                    "results_count": len(result.results)
                })
                
            except Exception as e:
                warmup_results["failed_warmups"] += 1
                warmup_results["details"].append({
                    "query": query,
                    "success": False,
                    "error": str(e)
                })
        
        warmup_results["average_time_ms"] = (
            warmup_results["total_time_ms"] / warmup_results["successful_warmups"]
            if warmup_results["successful_warmups"] > 0 else 0
        )
        
        logger.info(f"Warmup completed: {warmup_results['successful_warmups']}/{len(queries)} successful")
        
        return warmup_results