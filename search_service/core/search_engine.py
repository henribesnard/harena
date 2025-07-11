"""
Moteur de recherche hybride principal - Chef d'orchestre - VERSION CENTRALISÉE.

Ce module coordonne les recherches lexicale et sémantique pour fournir
des résultats optimaux via différentes stratégies de fusion intelligentes.

CENTRALISÉ VIA CONFIG_SERVICE:
- Toutes les configurations viennent de config_service.config.settings
- Poids de fusion, timeouts, cache configurables
- Stratégies de fusion basées sur la config centralisée
"""
import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

# ✅ CONFIGURATION CENTRALISÉE - SEULE SOURCE DE VÉRITÉ
from config_service.config import settings

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
    """Configuration pour la recherche hybride - Basé sur config_service."""
    # Poids par défaut pour la fusion depuis config centralisée
    default_lexical_weight: float = settings.DEFAULT_LEXICAL_WEIGHT
    default_semantic_weight: float = settings.DEFAULT_SEMANTIC_WEIGHT
    
    # Stratégie de fusion par défaut depuis config centralisée
    fusion_strategy: FusionStrategy = FusionStrategy(settings.DEFAULT_FUSION_STRATEGY)
    
    # Paramètres de qualité et performance depuis config centralisée
    min_results_for_fusion: int = settings.MIN_RESULTS_FOR_FUSION
    max_results_per_engine: int = settings.MAX_RESULTS_PER_ENGINE
    
    # Cache depuis config centralisée
    enable_cache: bool = settings.SEARCH_CACHE_ENABLED
    cache_ttl_seconds: int = settings.SEARCH_CACHE_TTL
    
    # Adaptation automatique des poids depuis config centralisée
    adaptive_weighting: bool = settings.ADAPTIVE_WEIGHTING
    quality_boost_factor: float = settings.QUALITY_BOOST_FACTOR
    
    # Timeouts depuis config centralisée
    search_timeout: float = settings.SEARCH_TIMEOUT
    lexical_timeout: float = settings.ELASTICSEARCH_TIMEOUT
    semantic_timeout: float = settings.QDRANT_TIMEOUT
    
    # Fallback et resilience depuis config centralisée
    enable_fallback: bool = settings.ENABLE_FALLBACK
    min_engine_success: int = settings.MIN_ENGINE_SUCCESS
    
    # Optimisations depuis config centralisée
    enable_parallel_search: bool = settings.ENABLE_PARALLEL_SEARCH
    enable_early_termination: bool = settings.ENABLE_EARLY_TERMINATION
    early_termination_threshold: float = settings.EARLY_TERMINATION_THRESHOLD


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
    
    CONFIGURATION CENTRALISÉE VIA CONFIG_SERVICE.
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
        
        # Configuration centralisée par défaut
        self.config = config or HybridSearchConfig()
        
        # Cache de résultats hybrides avec config centralisée
        self.cache = SearchCache(
            max_size=settings.SEARCH_CACHE_SIZE,
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
        
        logger.info("Hybrid search engine initialized with centralized config")
    
    async def search(
        self,
        query: str,
        user_id: int,
        search_type: SearchType = SearchType.HYBRID,
        limit: int = None,
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
            limit: Nombre de résultats (utilise config si None)
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
        
        # Utiliser la limite par défaut de la config centralisée
        if limit is None:
            limit = settings.DEFAULT_LIMIT
        
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
            
            # Déterminer les poids optimaux avec config centralisée
            weights = self._determine_search_weights(
                query_analysis, lexical_weight, semantic_weight, search_type
            )
            
            # Exécuter la recherche selon le type avec timeout configuré
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
                debug_info={"strategy": "lexical_only", "config_source": "centralized"} if debug else None
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
                debug_info={"strategy": "semantic_only", "config_source": "centralized"} if debug else None
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
        
        # Limite élargie pour meilleure fusion (config centralisée)
        expanded_limit = min(limit + offset + 10, self.config.max_results_per_engine)
        
        # Créer les tâches de recherche
        lexical_task = self._safe_lexical_search(
            query, user_id, expanded_limit, 0, SortOrder.RELEVANCE, filters, debug
        )
        
        semantic_task = self._safe_semantic_search(
            query, user_id, expanded_limit, 0, similarity_threshold, 
            SortOrder.RELEVANCE, filters, debug
        )
        
        # Exécuter les recherches selon la config
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
        
        # Vérifier qu'au moins un moteur a réussi (config centralisée)
        successful_engines = sum([lexical_success, semantic_success])
        
        if successful_engines < self.config.min_engine_success:
            self.engine_failures["both"] += 1
            raise Exception("All search engines failed")
        
        # Fallback si un seul moteur a réussi et fallback activé
        if self.config.enable_fallback:
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
            
            # Fallback vers le meilleur résultat individuel si activé
            if self.config.enable_fallback and lexical_result and semantic_result:
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
        """Recherche lexicale sécurisée avec timeout configuré."""
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
        """Recherche sémantique sécurisée avec timeout configuré."""
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
        """Détermine les poids optimaux pour la recherche avec config centralisée."""
        
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
        
        # Adaptation automatique pour recherche hybride si activée
        if self.config.adaptive_weighting:
            return self._calculate_adaptive_weights(query_analysis)
        
        # Poids par défaut de la config centralisée
        return {
            "lexical_weight": self.config.default_lexical_weight,
            "semantic_weight": self.config.default_semantic_weight
        }
    
    def _calculate_adaptive_weights(self, query_analysis: QueryAnalysis) -> Dict[str, float]:
        """Calcule les poids adaptatifs basés sur l'analyse de requête et config centralisée."""
        
        base_lexical = self.config.default_lexical_weight
        base_semantic = self.config.default_semantic_weight
        
        # Facteurs d'ajustement
        lexical_boost = 0.0
        semantic_boost = 0.0
        
        # Utiliser les propriétés correctes de QueryAnalysis
        
        # Requêtes avec phrases exactes -> favoriser lexical
        if getattr(query_analysis, 'has_exact_phrases', False):
            lexical_boost += 0.2
        
        # Requêtes courtes et spécifiques -> favoriser lexical
        key_terms = getattr(query_analysis, 'key_terms', [])
        if len(key_terms) <= 2:
            lexical_boost += 0.1
        
        # Requêtes avec entités financières -> équilibrer
        if getattr(query_analysis, 'has_financial_entities', False):
            # Rééquilibrer vers 50/50
            target_lexical = 0.5
            target_semantic = 0.5
            base_lexical = (base_lexical + target_lexical) / 2
            base_semantic = (base_semantic + target_semantic) / 2
        
        # Requêtes longues et complexes -> favoriser sémantique
        if len(key_terms) > 5:
            semantic_boost += 0.15
        
        # Requêtes interrogatives -> favoriser sémantique
        if getattr(query_analysis, 'is_question', False):
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
            debug_info={"fallback": True, "strategy": strategy, "config_source": "centralized"} if debug else None
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
        
        # Statistiques de performance (seuils configurés)
        fast_threshold = settings.FAST_SEARCH_THRESHOLD_MS if hasattr(settings, 'FAST_SEARCH_THRESHOLD_MS') else 2000
        slow_threshold = settings.SLOW_SEARCH_THRESHOLD_MS if hasattr(settings, 'SLOW_SEARCH_THRESHOLD_MS') else 5000
        
        if processing_time < fast_threshold:
            self.performance_stats["fast_searches"] += 1
        elif processing_time < slow_threshold:
            self.performance_stats["normal_searches"] += 1
        else:
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
            },
            "config_source": "centralized (config_service)"
        }
        
        # Ajouter les infos de debug de la fusion si disponibles
        if fusion_result.debug_info:
            debug_info["fusion_debug"] = fusion_result.debug_info
        
        return debug_info
    
    async def health_check(self) -> Dict[str, Any]:
        """Vérifie la santé du moteur hybride avec info config centralisée."""
        health_status = {
            "status": "healthy",
            "engines": {},
            "metrics": self.get_metrics(),
            "config_source": "centralized (config_service)"
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
        """Retourne les métriques du moteur hybride avec info config centralisée."""
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
            },
            
            "config_source": "centralized (config_service)",
            "centralized_settings": {
                "search_timeout": settings.SEARCH_TIMEOUT,
                "lexical_timeout": settings.ELASTICSEARCH_TIMEOUT,
                "semantic_timeout": settings.QDRANT_TIMEOUT,
                "default_weights": {
                    "lexical": settings.DEFAULT_LEXICAL_WEIGHT,
                    "semantic": settings.DEFAULT_SEMANTIC_WEIGHT
                },
                "fusion_strategy": settings.DEFAULT_FUSION_STRATEGY,
                "cache_enabled": settings.SEARCH_CACHE_ENABLED,
                "cache_size": settings.SEARCH_CACHE_SIZE,
                "adaptive_weighting": settings.ADAPTIVE_WEIGHTING,
                "parallel_search": settings.ENABLE_PARALLEL_SEARCH,
                "fallback_enabled": settings.ENABLE_FALLBACK
            }
        }
    
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
                    max_size=settings.SEARCH_CACHE_SIZE,
                    ttl_seconds=new_config.cache_ttl_seconds
                )
            else:
                self.cache = None
        
        logger.info("Hybrid engine configuration updated (centralized config)")
    
    def get_current_config(self) -> Dict[str, Any]:
        """Retourne la configuration actuelle complète du moteur."""
        return {
            "hybrid_search": {
                "default_lexical_weight": self.config.default_lexical_weight,
                "default_semantic_weight": self.config.default_semantic_weight,
                "fusion_strategy": self.config.fusion_strategy.value,
                "min_results_for_fusion": self.config.min_results_for_fusion,
                "max_results_per_engine": self.config.max_results_per_engine,
                "adaptive_weighting": self.config.adaptive_weighting,
                "quality_boost_factor": self.config.quality_boost_factor,
                "enable_parallel_search": self.config.enable_parallel_search,
                "enable_fallback": self.config.enable_fallback,
                "min_engine_success": self.config.min_engine_success,
                "enable_early_termination": self.config.enable_early_termination,
                "early_termination_threshold": self.config.early_termination_threshold
            },
            "timeouts": {
                "search_timeout": self.config.search_timeout,
                "lexical_timeout": self.config.lexical_timeout,
                "semantic_timeout": self.config.semantic_timeout
            },
            "cache": {
                "enabled": self.config.enable_cache,
                "ttl": self.config.cache_ttl_seconds,
                "max_size": settings.SEARCH_CACHE_SIZE
            },
            "config_source": "centralized (config_service)"
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Retourne un résumé des performances avec config centralisée."""
        total_searches = sum(self.performance_stats.values())
        
        if total_searches == 0:
            return {"message": "No searches performed yet", "config_source": "centralized"}
        
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
            },
            "config_source": "centralized (config_service)"
        }