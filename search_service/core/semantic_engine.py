"""
Moteur de recherche sémantique pour Qdrant.

Ce module implémente la recherche sémantique complète avec génération d'embeddings,
recherche vectorielle par similarité et évaluation de qualité.
"""
import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from search_service.clients.qdrant_client import QdrantClient
from search_service.core.embeddings import EmbeddingManager
from search_service.core.query_processor import QueryProcessor, QueryAnalysis
from search_service.models.search_types import SearchType, SearchQuality, SortOrder
from search_service.models.responses import SearchResultItem
from search_service.utils.cache import SearchCache

logger = logging.getLogger(__name__)


@dataclass
class SemanticSearchConfig:
    """Configuration pour la recherche sémantique."""
    # Seuils de similarité par type de requête
    similarity_threshold_default: float = 0.5
    similarity_threshold_strict: float = 0.7
    similarity_threshold_loose: float = 0.3
    
    # Configuration des requêtes
    max_results: int = 50
    enable_filtering: bool = True
    fallback_to_unfiltered: bool = True
    
    # Configuration des recommandations
    recommendation_enabled: bool = True
    recommendation_threshold: float = 0.6
    
    # Performance
    timeout_seconds: float = 8.0
    enable_cache: bool = True
    cache_ttl_seconds: int = 600
    
    # Qdrant spécifique
    collection_name: str = "harena_transactions"
    vector_size: int = 1536
    distance_metric: str = "cosine"
    
    # Stratégies de recherche
    enable_hybrid_scoring: bool = True
    enable_query_expansion: bool = True
    min_results_for_quality: int = 3


@dataclass
class SemanticSearchResult:
    """Résultat d'une recherche sémantique."""
    results: List[SearchResultItem]
    total_found: int
    processing_time_ms: float
    embedding_time_ms: float
    query_used: str
    similarity_threshold_used: float
    quality: SearchQuality
    qdrant_query: Optional[Dict[str, Any]] = None
    debug_info: Optional[Dict[str, Any]] = None


class SemanticSearchEngine:
    """
    Moteur de recherche sémantique utilisant Qdrant.
    
    Responsabilités:
    - Génération d'embeddings pour requêtes
    - Recherche vectorielle par similarité
    - Gestion des seuils de similarité adaptatifs
    - Filtrage par métadonnées
    - Optimisation des performances vectorielles
    """
    
    def __init__(
        self,
        qdrant_client: QdrantClient,
        embedding_manager: EmbeddingManager,
        query_processor: Optional[QueryProcessor] = None,
        config: Optional[SemanticSearchConfig] = None
    ):
        self.qdrant_client = qdrant_client
        self.embedding_manager = embedding_manager
        self.query_processor = query_processor or QueryProcessor()
        self.config = config or SemanticSearchConfig()
        
        # Cache pour les résultats et embeddings
        self.cache = SearchCache(
            max_size=1000,
            ttl_seconds=self.config.cache_ttl_seconds
        ) if self.config.enable_cache else None
        
        # Métriques de performance
        self.search_count = 0
        self.total_processing_time = 0.0
        self.embedding_generation_time = 0.0
        self.cache_hits = 0
        self.failed_searches = 0
        self.quality_distribution = {quality.value: 0 for quality in SearchQuality}
        self.threshold_adjustments = 0
        
        logger.info("Semantic search engine initialized")
    
    async def search(
        self,
        query: str,
        user_id: int,
        limit: int = 20,
        offset: int = 0,
        similarity_threshold: Optional[float] = None,
        sort_order: SortOrder = SortOrder.RELEVANCE,
        filters: Optional[Dict[str, Any]] = None,
        debug: bool = False
    ) -> SemanticSearchResult:
        """
        Effectue une recherche sémantique dans Qdrant.
        
        Args:
            query: Terme de recherche
            user_id: ID de l'utilisateur
            limit: Nombre de résultats
            offset: Décalage pour pagination
            similarity_threshold: Seuil de similarité (auto si None)
            sort_order: Ordre de tri
            filters: Filtres additionnels
            debug: Inclure informations de debug
            
        Returns:
            Résultats de recherche sémantique
            
        Raises:
            Exception: Si la recherche échoue
        """
        start_time = time.time()
        self.search_count += 1
        
        # Génération de la clé de cache
        cache_key = None
        if self.cache:
            cache_key = self._generate_cache_key(
                query, user_id, limit, offset, similarity_threshold, sort_order, filters
            )
            
            # Vérifier le cache
            cached_result = self.cache.get(cache_key)
            if cached_result:
                self.cache_hits += 1
                logger.debug(f"Cache hit for semantic search: {cache_key}")
                return cached_result
        
        try:
            # 1. Analyser et traiter la requête
            query_analysis = self.query_processor.process_query(query)
            optimized_query = self._optimize_query_for_semantic_search(query_analysis)
            
            # 2. Générer l'embedding pour la requête
            embedding_start = time.time()
            query_embedding = await self.embedding_manager.generate_embedding(
                optimized_query, use_cache=True
            )
            embedding_time = (time.time() - embedding_start) * 1000
            self.embedding_generation_time += embedding_time
            
            if not query_embedding:
                raise Exception("Failed to generate query embedding")
            
            # 3. Déterminer le seuil de similarité optimal
            if similarity_threshold is None:
                similarity_threshold = self._determine_optimal_threshold(query_analysis)
            
            # 4. Effectuer la recherche vectorielle avec stratégie de fallback
            search_results = await self._search_with_fallback_strategy(
                query_embedding=query_embedding,
                user_id=user_id,
                limit=min(limit + offset + 10, self.config.max_results),  # Marge pour pagination
                similarity_threshold=similarity_threshold,
                filters=filters,
                debug=debug
            )
            
            # 5. Traiter et convertir les résultats
            processed_results = self._process_qdrant_results(
                search_results, query_analysis, debug
            )
            
            # 6. Appliquer pagination et tri
            final_results = self._apply_pagination_and_sorting(
                processed_results, offset, limit, sort_order
            )
            
            # 7. Calculer les métriques de qualité
            processing_time = (time.time() - start_time) * 1000
            self.total_processing_time += processing_time
            
            quality = self._assess_semantic_quality(final_results, query_analysis)
            self.quality_distribution[quality.value] += 1
            
            # 8. Construire le résultat
            result = SemanticSearchResult(
                results=final_results,
                total_found=len(search_results) if search_results else 0,
                processing_time_ms=processing_time,
                embedding_time_ms=embedding_time,
                query_used=optimized_query,
                similarity_threshold_used=similarity_threshold,
                quality=quality,
                qdrant_query=self._build_qdrant_query_info(
                    query_embedding, similarity_threshold, filters
                ) if debug else None,
                debug_info=self._extract_debug_info(search_results) if debug else None
            )
            
            # 9. Mettre en cache si activé
            if self.cache and cache_key:
                self.cache.put(cache_key, result)
            
            return result
            
        except asyncio.TimeoutError:
            self.failed_searches += 1
            logger.error(f"Semantic search timeout after {self.config.timeout_seconds}s")
            raise Exception("Search timeout")
            
        except Exception as e:
            self.failed_searches += 1
            logger.error(f"Semantic search failed: {e}", exc_info=True)
            raise Exception(f"Semantic search error: {str(e)}")
    
    def _generate_cache_key(
        self,
        query: str,
        user_id: int,
        limit: int,
        offset: int,
        similarity_threshold: Optional[float],
        sort_order: SortOrder,
        filters: Optional[Dict[str, Any]]
    ) -> str:
        """Génère une clé de cache pour la requête."""
        import hashlib
        
        cache_data = {
            "query": query.lower().strip(),
            "user_id": user_id,
            "limit": limit,
            "offset": offset,
            "threshold": similarity_threshold,
            "sort_order": sort_order.value,
            "filters": filters or {}
        }
        
        cache_str = str(sorted(cache_data.items()))
        return f"semantic_{hashlib.md5(cache_str.encode()).hexdigest()}"
    
    def _optimize_query_for_semantic_search(self, query_analysis: QueryAnalysis) -> str:
        """Optimise la requête pour la recherche sémantique."""
        # Pour la recherche sémantique, on privilégie le contexte et le sens
        if self.config.enable_query_expansion and query_analysis.enriched_query:
            return query_analysis.enriched_query
        
        # Utiliser la requête nettoyée avec expansion contextuelle
        base_query = query_analysis.cleaned_query or query_analysis.original_query
        
        # Ajouter du contexte financier si pertinent
        if query_analysis.has_financial_entities:
            financial_context = " transaction financière"
            if "transaction" not in base_query.lower():
                base_query += financial_context
        
        return base_query
    
    def _determine_optimal_threshold(self, query_analysis: QueryAnalysis) -> float:
        """Détermine le seuil de similarité optimal basé sur l'analyse de requête."""
        # Seuil strict pour requêtes très spécifiques
        if (query_analysis.has_exact_phrases or 
            query_analysis.has_financial_entities or
            len(query_analysis.key_terms) <= 2):
            return self.config.similarity_threshold_strict
        
        # Seuil relâché pour requêtes génériques ou exploratoires
        elif (len(query_analysis.key_terms) > 5 or
              query_analysis.is_question or
              any(word in query_analysis.original_query.lower() 
                  for word in ["quoi", "comment", "pourquoi", "où", "quand"])):
            return self.config.similarity_threshold_loose
        
        # Seuil par défaut
        return self.config.similarity_threshold_default
    
    async def _search_with_fallback_strategy(
        self,
        query_embedding: List[float],
        user_id: int,
        limit: int,
        similarity_threshold: float,
        filters: Optional[Dict[str, Any]],
        debug: bool
    ) -> Optional[List[Dict[str, Any]]]:
        """Effectue la recherche avec stratégie de fallback."""
        try:
            # 1. Recherche principale avec filtres
            if filters and self.config.enable_filtering:
                results = await self._execute_filtered_search(
                    query_embedding, user_id, limit, similarity_threshold, filters, debug
                )
                
                # Si pas assez de résultats et fallback activé
                if (results and len(results) < self.config.min_results_for_quality and 
                    self.config.fallback_to_unfiltered):
                    
                    logger.info(f"Fallback: only {len(results)} filtered results, trying unfiltered")
                    
                    # 2. Fallback sans filtres avec seuil réduit
                    fallback_threshold = similarity_threshold * 0.8
                    fallback_results = await self._execute_unfiltered_search(
                        query_embedding, user_id, limit, fallback_threshold, debug
                    )
                    
                    # Combiner et dédupliquer
                    if fallback_results:
                        results = self._merge_and_deduplicate_results(results, fallback_results)
                
                return results
            
            else:
                # 3. Recherche simple sans filtres
                return await self._execute_unfiltered_search(
                    query_embedding, user_id, limit, similarity_threshold, debug
                )
                
        except Exception as e:
            logger.error(f"Qdrant search failed: {e}")
            # Fallback final avec seuil très relâché
            if similarity_threshold > self.config.similarity_threshold_loose:
                logger.info("Final fallback with relaxed threshold")
                self.threshold_adjustments += 1
                return await self._execute_unfiltered_search(
                    query_embedding, user_id, limit, 
                    self.config.similarity_threshold_loose, debug
                )
            raise
    
    async def _execute_filtered_search(
        self,
        query_embedding: List[float],
        user_id: int,
        limit: int,
        similarity_threshold: float,
        filters: Dict[str, Any],
        debug: bool
    ) -> Optional[List[Dict[str, Any]]]:
        """Exécute une recherche avec filtres."""
        # Construire les filtres Qdrant
        qdrant_filters = self._build_qdrant_filters(user_id, filters)
        
        search_params = {
            "collection_name": self.config.collection_name,
            "query_vector": query_embedding,
            "query_filter": qdrant_filters,
            "limit": limit,
            "score_threshold": similarity_threshold,
            "with_payload": True,
            "with_vectors": debug
        }
        
        return await asyncio.wait_for(
            self.qdrant_client.search(**search_params),
            timeout=self.config.timeout_seconds
        )
    
    async def _execute_unfiltered_search(
        self,
        query_embedding: List[float],
        user_id: int,
        limit: int,
        similarity_threshold: float,
        debug: bool
    ) -> Optional[List[Dict[str, Any]]]:
        """Exécute une recherche sans filtres (seulement user_id)."""
        qdrant_filters = {"must": [{"key": "user_id", "match": {"value": user_id}}]}
        
        search_params = {
            "collection_name": self.config.collection_name,
            "query_vector": query_embedding,
            "query_filter": qdrant_filters,
            "limit": limit,
            "score_threshold": similarity_threshold,
            "with_payload": True,
            "with_vectors": debug
        }
        
        return await asyncio.wait_for(
            self.qdrant_client.search(**search_params),
            timeout=self.config.timeout_seconds
        )
    
    def _build_qdrant_filters(self, user_id: int, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Construit les filtres Qdrant."""
        must_conditions = [
            {"key": "user_id", "match": {"value": user_id}}
        ]
        
        # Filtre par type de transaction
        if filters.get("transaction_type"):
            must_conditions.append({
                "key": "transaction_type",
                "match": {"value": filters["transaction_type"]}
            })
        
        # Filtre par montant
        if filters.get("amount_range"):
            amount_range = filters["amount_range"]
            range_condition = {"key": "amount", "range": {}}
            
            if amount_range.get("min") is not None:
                range_condition["range"]["gte"] = amount_range["min"]
            if amount_range.get("max") is not None:
                range_condition["range"]["lte"] = amount_range["max"]
            
            if range_condition["range"]:
                must_conditions.append(range_condition)
        
        # Filtre par date
        if filters.get("date_range"):
            date_range = filters["date_range"]
            range_condition = {"key": "transaction_date", "range": {}}
            
            if date_range.get("start"):
                range_condition["range"]["gte"] = date_range["start"]
            if date_range.get("end"):
                range_condition["range"]["lte"] = date_range["end"]
            
            if range_condition["range"]:
                must_conditions.append(range_condition)
        
        # Filtre par comptes
        if filters.get("account_ids") and filters["account_ids"]:
            must_conditions.append({
                "key": "account_id",
                "match": {"any": filters["account_ids"]}
            })
        
        # Filtre par catégories
        if filters.get("category_ids") and filters["category_ids"]:
            must_conditions.append({
                "key": "category_id",
                "match": {"any": filters["category_ids"]}
            })
        
        return {"must": must_conditions}
    
    def _merge_and_deduplicate_results(
        self,
        primary_results: List[Dict[str, Any]],
        fallback_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Fusionne et déduplique les résultats de différentes recherches."""
        seen_ids = set()
        merged_results = []
        
        # Ajouter les résultats primaires (priorité)
        for result in primary_results:
            transaction_id = result.get("payload", {}).get("transaction_id")
            if transaction_id not in seen_ids:
                seen_ids.add(transaction_id)
                merged_results.append(result)
        
        # Ajouter les résultats de fallback non dupliqués
        for result in fallback_results:
            transaction_id = result.get("payload", {}).get("transaction_id")
            if transaction_id not in seen_ids:
                seen_ids.add(transaction_id)
                # Marquer comme résultat de fallback
                if "payload" in result:
                    result["payload"]["_fallback_result"] = True
                merged_results.append(result)
        
        return merged_results
    
    def _process_qdrant_results(
        self,
        search_results: Optional[List[Dict[str, Any]]],
        query_analysis: QueryAnalysis,
        debug: bool
    ) -> List[SearchResultItem]:
        """Traite les résultats Qdrant en SearchResultItem."""
        if not search_results:
            return []
        
        results = []
        
        for result in search_results:
            payload = result.get("payload", {})
            score = result.get("score", 0.0)
            
            # Construire les métadonnées
            metadata = {
                "search_engine": "semantic",
                "qdrant_score": score,
                "similarity_score": score,
                "is_fallback_result": payload.get("_fallback_result", False)
            }
            
            if debug:
                metadata["debug"] = {
                    "qdrant_id": result.get("id"),
                    "vector": result.get("vector") if debug else None
                }
            
            result_item = SearchResultItem(
                transaction_id=payload.get("transaction_id"),
                user_id=payload.get("user_id"),
                account_id=payload.get("account_id"),
                score=score,
                lexical_score=None,
                semantic_score=score,
                combined_score=score,
                primary_description=payload.get("primary_description", ""),
                searchable_text=payload.get("searchable_text"),
                merchant_name=payload.get("merchant_name"),
                amount=payload.get("amount"),
                currency_code=payload.get("currency_code", "EUR"),
                transaction_type=payload.get("transaction_type", ""),
                transaction_date=payload.get("transaction_date", ""),
                created_at=payload.get("created_at"),
                category_id=payload.get("category_id"),
                operation_type=payload.get("operation_type"),
                highlights=None,  # Pas de highlighting en sémantique
                metadata=metadata
            )
            
            results.append(result_item)
        
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
            results.sort(key=lambda x: x.transaction_date, reverse=True)
        elif sort_order == SortOrder.DATE_ASC:
            results.sort(key=lambda x: x.transaction_date)
        elif sort_order == SortOrder.AMOUNT_DESC:
            results.sort(key=lambda x: x.amount or 0, reverse=True)
        elif sort_order == SortOrder.AMOUNT_ASC:
            results.sort(key=lambda x: x.amount or 0)
        else:  # RELEVANCE (défaut)
            results.sort(key=lambda x: x.score or 0, reverse=True)
        
        # Appliquer la pagination
        return results[offset:offset + limit]
    
    def _build_qdrant_query_info(
        self,
        query_embedding: List[float],
        similarity_threshold: float,
        filters: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Construit les informations de requête Qdrant pour debug."""
        return {
            "vector_size": len(query_embedding),
            "similarity_threshold": similarity_threshold,
            "distance_metric": self.config.distance_metric,
            "collection": self.config.collection_name,
            "filters_applied": filters is not None,
            "filter_count": len(filters) if filters else 0
        }
    
    def _extract_debug_info(self, search_results: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Extrait les informations de debug."""
        if not search_results:
            return {"result_count": 0}
        
        scores = [r.get("score", 0) for r in search_results]
        
        return {
            "result_count": len(search_results),
            "max_score": max(scores) if scores else 0,
            "min_score": min(scores) if scores else 0,
            "avg_score": sum(scores) / len(scores) if scores else 0,
            "score_distribution": {
                "excellent": len([s for s in scores if s >= 0.8]),
                "good": len([s for s in scores if 0.6 <= s < 0.8]),
                "medium": len([s for s in scores if 0.4 <= s < 0.6]),
                "poor": len([s for s in scores if s < 0.4])
            }
        }
    
    def _assess_semantic_quality(
        self,
        results: List[SearchResultItem],
        query_analysis: QueryAnalysis
    ) -> SearchQuality:
        """Évalue la qualité des résultats sémantiques."""
        if not results:
            return SearchQuality.POOR
        
        # Calculer différents aspects de qualité
        score_quality = self._assess_similarity_scores(results)
        consistency_quality = self._assess_result_consistency(results)
        diversity_quality = self._assess_semantic_diversity(results)
        relevance_quality = self._assess_contextual_relevance(results, query_analysis)
        
        # Moyenne pondérée des qualités
        overall_quality = (
            score_quality * 0.40 +
            consistency_quality * 0.25 +
            relevance_quality * 0.25 +
            diversity_quality * 0.10
        )
        
        # Conversion en enum de qualité
        if overall_quality >= 0.8:
            return SearchQuality.EXCELLENT
        elif overall_quality >= 0.6:
            return SearchQuality.GOOD
        elif overall_quality >= 0.4:
            return SearchQuality.MEDIUM
        else:
            return SearchQuality.POOR
    
    def _assess_similarity_scores(self, results: List[SearchResultItem]) -> float:
        """Évalue la qualité basée sur les scores de similarité."""
        if not results:
            return 0.0
        
        scores = [r.semantic_score for r in results if r.semantic_score]
        if not scores:
            return 0.0
        
        max_score = max(scores)
        avg_score = sum(scores) / len(scores)
        
        # Qualité basée sur le score max et la moyenne
        max_quality = max_score  # Les scores Qdrant sont déjà entre 0-1
        avg_quality = avg_score
        
        # Bonus si les scores sont élevés et cohérents
        score_consistency = 1 - (max_score - min(scores)) if len(scores) > 1 else 1
        
        return (max_quality * 0.5 + avg_quality * 0.3 + score_consistency * 0.2)
    
    def _assess_result_consistency(self, results: List[SearchResultItem]) -> float:
        """Évalue la cohérence des résultats."""
        if len(results) <= 1:
            return 1.0
        
        # Cohérence des scores (éviter les chutes brutales)
        scores = [r.semantic_score for r in results if r.semantic_score]
        if len(scores) < 2:
            return 0.5
        
        # Calculer la cohérence des écarts entre scores consécutifs
        score_gaps = []
        for i in range(1, len(scores)):
            gap = scores[i-1] - scores[i]
            score_gaps.append(gap)
        
        if score_gaps:
            avg_gap = sum(score_gaps) / len(score_gaps)
            max_gap = max(score_gaps)
            
            # Bonne cohérence = écarts réguliers et pas trop grands
            gap_consistency = 1 - min(max_gap, 0.5) * 2  # Pénalité pour gros écarts
            gap_regularity = 1 - abs(avg_gap - max_gap) if max_gap > 0 else 1
            
            return (gap_consistency * 0.7 + gap_regularity * 0.3)
        
        return 0.5
    
    def _assess_semantic_diversity(self, results: List[SearchResultItem]) -> float:
        """Évalue la diversité sémantique des résultats."""
        if len(results) <= 1:
            return 1.0
        
        # Diversité par catégorie
        categories = {r.category_id for r in results if r.category_id}
        category_diversity = len(categories) / len(results) if categories else 0.5
        
        # Diversité par marchand
        merchants = {r.merchant_name for r in results if r.merchant_name}
        merchant_diversity = len(merchants) / len(results) if merchants else 0.5
        
        # Diversité temporelle (répartition dans le temps)
        dates = [r.transaction_date for r in results if r.transaction_date]
        if len(dates) > 1:
            unique_dates = len(set(dates))
            temporal_diversity = min(unique_dates / len(dates), 1.0)
        else:
            temporal_diversity = 0.5
        
        # Moyenne pondérée
        return (category_diversity * 0.4 + merchant_diversity * 0.4 + temporal_diversity * 0.2)
    
    def _assess_contextual_relevance(
        self,
        results: List[SearchResultItem],
        query_analysis: QueryAnalysis
    ) -> float:
        """Évalue la pertinence contextuelle des résultats."""
        if not results or not query_analysis.key_terms:
            return 0.5
        
        relevance_scores = []
        
        for result in results:
            # Construire le contexte textuel
            context = " ".join(filter(None, [
                result.primary_description,
                result.merchant_name,
                result.searchable_text
            ])).lower()
            
            # Vérifier la présence des termes clés
            matching_terms = sum(
                1 for term in query_analysis.key_terms
                if term.lower() in context
            )
            
            term_relevance = matching_terms / len(query_analysis.key_terms)
            
            # Bonus pour richesse du contexte
            context_richness = min(len(context.split()) / 10, 1.0)
            
            # Score de pertinence combiné
            result_relevance = (term_relevance * 0.7 + context_richness * 0.3)
            relevance_scores.append(result_relevance)
        
        return sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0
    
    def get_metrics(self) -> Dict[str, Any]:
        """Retourne les métriques du moteur sémantique."""
        avg_processing_time = (
            self.total_processing_time / self.search_count
            if self.search_count > 0 else 0
        )
        
        avg_embedding_time = (
            self.embedding_generation_time / self.search_count
            if self.search_count > 0 else 0
        )
        
        cache_hit_rate = self.cache_hits / self.search_count if self.search_count > 0 else 0
        failure_rate = self.failed_searches / self.search_count if self.search_count > 0 else 0
        
        return {
            "engine_type": "semantic",
            "search_count": self.search_count,
            "total_processing_time_ms": self.total_processing_time,
            "average_processing_time_ms": avg_processing_time,
            "embedding_generation_time_ms": self.embedding_generation_time,
            "average_embedding_time_ms": avg_embedding_time,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": cache_hit_rate,
            "failed_searches": self.failed_searches,
            "failure_rate": failure_rate,
            "threshold_adjustments": self.threshold_adjustments,
            "quality_distribution": self.quality_distribution,
            "cache_stats": self.cache.get_stats() if self.cache else None
        }
    
    def clear_cache(self) -> None:
        """Vide le cache du moteur sémantique."""
        if self.cache:
            self.cache.clear()
            logger.info("Semantic engine cache cleared")
    
    async def health_check(self) -> Dict[str, Any]:
        """Vérifie la santé du moteur sémantique."""
        try:
            # Test de connectivité Qdrant
            health = await self.qdrant_client.health_check()
            
            # Test de génération d'embedding
            test_embedding = await self.embedding_manager.generate_embedding(
                "test query", use_cache=False
            )
            
            return {
                "status": "healthy",
                "qdrant_status": health.get("status", "unknown"),
                "embedding_service": "healthy" if test_embedding else "unhealthy",
                "collection_name": self.config.collection_name,
                "metrics": self.get_metrics()
            }
        except Exception as e:
            logger.error(f"Semantic engine health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "metrics": self.get_metrics()
            }
    
    async def get_collection_info(self) -> Dict[str, Any]:
        """Retourne les informations sur la collection Qdrant."""
        try:
            collection_info = await self.qdrant_client.get_collection_info(
                self.config.collection_name
            )
            
            return {
                "collection_name": self.config.collection_name,
                "status": collection_info.get("status"),
                "vectors_count": collection_info.get("vectors_count", 0),
                "indexed_vectors_count": collection_info.get("indexed_vectors_count", 0),
                "points_count": collection_info.get("points_count", 0),
                "segments_count": collection_info.get("segments_count", 0),
                "config": {
                    "vector_size": self.config.vector_size,
                    "distance_metric": self.config.distance_metric
                }
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {
                "collection_name": self.config.collection_name,
                "status": "error",
                "error": str(e)
            }
    
    async def recommend_similar_transactions(
        self,
        transaction_id: int,
        user_id: int,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResultItem]:
        """
        Recommande des transactions similaires à une transaction donnée.
        
        Args:
            transaction_id: ID de la transaction de référence
            user_id: ID de l'utilisateur
            limit: Nombre de recommandations
            filters: Filtres additionnels
            
        Returns:
            Liste de transactions similaires
        """
        if not self.config.recommendation_enabled:
            return []
        
        try:
            # 1. Récupérer le vecteur de la transaction de référence
            reference_vector = await self.qdrant_client.get_vector(
                collection_name=self.config.collection_name,
                point_id=transaction_id
            )
            
            if not reference_vector:
                logger.warning(f"No vector found for transaction {transaction_id}")
                return []
            
            # 2. Effectuer la recherche de similarité
            search_results = await self._execute_filtered_search(
                query_embedding=reference_vector,
                user_id=user_id,
                limit=limit + 1,  # +1 pour exclure la transaction de référence
                similarity_threshold=self.config.recommendation_threshold,
                filters=filters or {},
                debug=False
            )
            
            # 3. Traiter les résultats et exclure la transaction de référence
            if search_results:
                processed_results = self._process_qdrant_results(
                    search_results, QueryAnalysis(), False
                )
                
                # Exclure la transaction de référence
                recommendations = [
                    r for r in processed_results 
                    if r.transaction_id != transaction_id
                ]
                
                return recommendations[:limit]
            
            return []
            
        except Exception as e:
            logger.error(f"Recommendation failed for transaction {transaction_id}: {e}")
            return []
    
    async def find_outliers(
        self,
        user_id: int,
        limit: int = 20,
        similarity_threshold: float = 0.3
    ) -> List[SearchResultItem]:
        """
        Trouve les transactions atypiques (outliers) pour un utilisateur.
        
        Args:
            user_id: ID de l'utilisateur
            limit: Nombre d'outliers à retourner
            similarity_threshold: Seuil en dessous duquel une transaction est considérée atypique
            
        Returns:
            Liste de transactions atypiques
        """
        try:
            # Recherche avec un seuil très bas pour trouver les transactions isolées
            search_results = await self._execute_unfiltered_search(
                query_embedding=[0.0] * self.config.vector_size,  # Vecteur neutre
                user_id=user_id,
                limit=limit * 2,  # Plus de résultats pour filtrer
                similarity_threshold=0.0,  # Pas de seuil pour avoir tous les résultats
                debug=False
            )
            
            if search_results:
                # Traiter les résultats
                processed_results = self._process_qdrant_results(
                    search_results, QueryAnalysis(), False
                )
                
                # Filtrer les outliers (scores très bas)
                outliers = [
                    r for r in processed_results 
                    if r.semantic_score and r.semantic_score < similarity_threshold
                ]
                
                # Trier par score croissant (plus atypiques en premier)
                outliers.sort(key=lambda x: x.semantic_score or 0)
                
                return outliers[:limit]
            
            return []
            
        except Exception as e:
            logger.error(f"Outlier detection failed for user {user_id}: {e}")
            return []
    
    def get_search_suggestions(
        self,
        partial_query: str,
        user_id: int,
        max_suggestions: int = 5
    ) -> List[str]:
        """
        Génère des suggestions de recherche basées sur une requête partielle.
        
        Args:
            partial_query: Début de requête
            user_id: ID de l'utilisateur
            max_suggestions: Nombre maximum de suggestions
            
        Returns:
            Liste de suggestions de recherche
        """
        # Suggestions basées sur des patterns financiers courants
        financial_patterns = [
            "restaurant", "supermarché", "essence", "pharmacie", "virement",
            "carte bancaire", "prélèvement", "remboursement", "salaire",
            "facture", "achat en ligne", "transport", "santé", "loisirs"
        ]
        
        # Filtrer les patterns qui commencent par la requête partielle
        suggestions = [
            pattern for pattern in financial_patterns
            if pattern.lower().startswith(partial_query.lower())
        ]
        
        # Ajouter des suggestions contextuelles
        if "montant" in partial_query.lower():
            suggestions.extend([
                "montant supérieur à 100",
                "montant inférieur à 50",
                "gros montants"
            ])
        
        if "date" in partial_query.lower():
            suggestions.extend([
                "transactions récentes",
                "ce mois-ci",
                "la semaine dernière"
            ])
        
        return suggestions[:max_suggestions]
    
    def update_config(self, new_config: SemanticSearchConfig) -> None:
        """Met à jour la configuration du moteur."""
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
        
        logger.info("Semantic engine configuration updated")
    
    def reset_metrics(self) -> None:
        """Remet à zéro les métriques de performance."""
        self.search_count = 0
        self.total_processing_time = 0.0
        self.embedding_generation_time = 0.0
        self.cache_hits = 0
        self.failed_searches = 0
        self.threshold_adjustments = 0
        self.quality_distribution = {quality.value: 0 for quality in SearchQuality}
        
        logger.info("Semantic engine metrics reset")