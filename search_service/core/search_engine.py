"""
Moteur de recherche principal.

Ce module orchestre la recherche hybride en combinant recherche lexicale,
sémantique et reranking pour optimiser la pertinence des résultats.
"""
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
import asyncio
from collections import defaultdict

from search_service.models import SearchQuery, SearchResponse, SearchResult, SearchType
from search_service.core.embeddings import embedding_service
from search_service.core.reranker import reranker_service
from search_service.core.query_processor import QueryProcessor
from search_service.storage.elastic_client import ElasticClient
from search_service.storage.qdrant_client import QdrantClient
from search_service.utils.cache import SearchCache

logger = logging.getLogger(__name__)


class SearchEngine:
    """Moteur de recherche hybride principal."""
    
    def __init__(
        self,
        elastic_client: Optional[ElasticClient] = None,
        qdrant_client: Optional[QdrantClient] = None,
        cache: Optional[SearchCache] = None
    ):
        self.elastic_client = elastic_client
        self.qdrant_client = qdrant_client
        self.cache = cache
        self.query_processor = QueryProcessor()
        
    async def search(self, query: SearchQuery) -> SearchResponse:
        """
        Effectue une recherche selon le type demandé.
        
        Args:
            query: Requête de recherche
            
        Returns:
            SearchResponse: Résultats de recherche
        """
        start_time = time.time()
        timings = {}
        
        # Traiter la requête
        query_start = time.time()
        processed_query = await self.query_processor.process(query.query)
        timings["query_processing"] = time.time() - query_start
        
        # Exécuter la recherche selon le type
        if query.search_type == SearchType.LEXICAL:
            results = await self._lexical_search(query, processed_query)
        elif query.search_type == SearchType.SEMANTIC:
            results = await self._semantic_search(query, processed_query)
        else:  # HYBRID
            results = await self._hybrid_search(query, processed_query, timings)
        
        # Appliquer le reranking si demandé
        if query.use_reranking and results and reranker_service.is_initialized():
            rerank_start = time.time()
            results = await self._rerank_results(query.query, results)
            timings["reranking"] = time.time() - rerank_start
        
        # Limiter les résultats selon la pagination
        total_found = len(results)
        results = results[query.offset:query.offset + query.limit]
        
        # Construire la réponse
        processing_time = time.time() - start_time
        timings["total"] = processing_time
        
        return SearchResponse(
            query=query.query,
            search_type=query.search_type,
            results=results,
            total_found=total_found,
            limit=query.limit,
            offset=query.offset,
            has_more=(query.offset + len(results)) < total_found,
            processing_time=processing_time,
            timings=timings if query.include_explanations else None,
            filters_applied=self._get_applied_filters(query),
            suggestions=processed_query.get("suggestions")
        )
    
    async def _hybrid_search(
        self,
        query: SearchQuery,
        processed_query: Dict[str, Any],
        timings: Dict[str, float]
    ) -> List[SearchResult]:
        """
        Effectue une recherche hybride combinant lexical et sémantique.
        
        Args:
            query: Requête de recherche
            processed_query: Requête traitée
            timings: Dictionnaire pour enregistrer les temps
            
        Returns:
            List[SearchResult]: Résultats fusionnés et triés
        """
        # Lancer les recherches en parallèle
        lexical_task = None
        semantic_task = None
        
        if self.elastic_client and query.lexical_weight > 0:
            lexical_task = asyncio.create_task(
                self._lexical_search(query, processed_query)
            )
        
        if self.qdrant_client and query.semantic_weight > 0:
            semantic_task = asyncio.create_task(
                self._semantic_search(query, processed_query)
            )
        
        # Attendre les résultats
        lexical_results = []
        semantic_results = []
        
        if lexical_task:
            lex_start = time.time()
            lexical_results = await lexical_task
            timings["lexical_search"] = time.time() - lex_start
            
        if semantic_task:
            sem_start = time.time()
            semantic_results = await semantic_task
            timings["semantic_search"] = time.time() - sem_start
        
        # Fusionner les résultats
        fusion_start = time.time()
        merged_results = self._merge_results(
            lexical_results,
            semantic_results,
            query.lexical_weight,
            query.semantic_weight
        )
        timings["result_fusion"] = time.time() - fusion_start
        
        return merged_results
    
    async def _lexical_search(
        self,
        query: SearchQuery,
        processed_query: Dict[str, Any]
    ) -> List[SearchResult]:
        """
        Effectue une recherche lexicale via Elasticsearch.
        
        Args:
            query: Requête de recherche
            processed_query: Requête traitée
            
        Returns:
            List[SearchResult]: Résultats de recherche lexicale
        """
        if not self.elastic_client:
            logger.warning("Elastic client not available for lexical search")
            return []
        
        try:
            # Construire la requête Elasticsearch
            es_query = self._build_elastic_query(query, processed_query)
            
            # Exécuter la recherche
            es_results = await self.elastic_client.search(
                user_id=query.user_id,
                query=es_query,
                limit=query.limit * 2,  # Récupérer plus pour la fusion
                filters=self._build_filters(query),
                include_highlights=query.include_highlights
            )
            
            # Convertir en SearchResult
            results = []
            for hit in es_results:
                result = self._elastic_hit_to_result(hit, query.include_explanations)
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Lexical search error: {e}")
            return []
    
    async def _semantic_search(
        self,
        query: SearchQuery,
        processed_query: Dict[str, Any]
    ) -> List[SearchResult]:
        """
        Effectue une recherche sémantique via Qdrant.
        
        Args:
            query: Requête de recherche
            processed_query: Requête traitée
            
        Returns:
            List[SearchResult]: Résultats de recherche sémantique
        """
        if not self.qdrant_client:
            logger.warning("Qdrant client not available for semantic search")
            return []
        
        try:
            # Générer l'embedding de la requête
            query_text = processed_query.get("expanded_query", query.query)
            query_embedding = await embedding_service.generate_embedding(query_text)
            
            # Exécuter la recherche vectorielle
            qdrant_results = await self.qdrant_client.search(
                query_vector=query_embedding,
                user_id=query.user_id,
                limit=query.limit * 2,  # Récupérer plus pour la fusion
                filters=self._build_filters(query)
            )
            
            # Convertir en SearchResult
            results = []
            for hit in qdrant_results:
                result = self._qdrant_hit_to_result(hit, query.include_explanations)
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Semantic search error: {e}")
            return []
    
    async def _rerank_results(
        self,
        query: str,
        results: List[SearchResult]
    ) -> List[SearchResult]:
        """
        Applique le reranking aux résultats.
        
        Args:
            query: Requête originale
            results: Résultats à reranker
            
        Returns:
            List[SearchResult]: Résultats rerankés
        """
        if not results:
            return results
        
        try:
            # Préparer les documents pour le reranking
            documents = []
            for result in results:
                # Construire le texte du document
                doc_text = f"{result.description} {result.merchant_name or ''}"
                if result.category_name:
                    doc_text += f" {result.category_name}"
                documents.append(doc_text)
            
            # Appliquer le reranking
            reranked_scores = await reranker_service.rerank(
                query=query,
                documents=documents
            )
            
            # Mettre à jour les scores
            for i, (result, rerank_score) in enumerate(zip(results, reranked_scores)):
                result.rerank_score = rerank_score
                # Combiner avec le score original
                result.score = 0.7 * rerank_score + 0.3 * result.score
            
            # Retrier par nouveau score
            results.sort(key=lambda x: x.score, reverse=True)
            
            return results
            
        except Exception as e:
            logger.error(f"Reranking error: {e}")
            # Retourner les résultats originaux en cas d'erreur
            return results
    
    def _merge_results(
        self,
        lexical_results: List[SearchResult],
        semantic_results: List[SearchResult],
        lexical_weight: float,
        semantic_weight: float
    ) -> List[SearchResult]:
        """
        Fusionne les résultats lexicaux et sémantiques.
        
        Args:
            lexical_results: Résultats de recherche lexicale
            semantic_results: Résultats de recherche sémantique
            lexical_weight: Poids de la recherche lexicale
            semantic_weight: Poids de la recherche sémantique
            
        Returns:
            List[SearchResult]: Résultats fusionnés et triés
        """
        # Créer un dictionnaire pour fusionner par transaction_id
        merged = {}
        
        # Ajouter les résultats lexicaux
        for result in lexical_results:
            result.lexical_score = result.score
            result.score = result.score * lexical_weight
            merged[result.transaction_id] = result
        
        # Fusionner avec les résultats sémantiques
        for result in semantic_results:
            if result.transaction_id in merged:
                # Combiner les scores
                existing = merged[result.transaction_id]
                existing.semantic_score = result.score
                existing.score += result.score * semantic_weight
                
                # Fusionner les highlights
                if result.highlights and existing.highlights:
                    for field, highlights in result.highlights.items():
                        if field in existing.highlights:
                            existing.highlights[field].extend(highlights)
                        else:
                            existing.highlights[field] = highlights
            else:
                # Nouveau résultat sémantique uniquement
                result.semantic_score = result.score
                result.score = result.score * semantic_weight
                merged[result.transaction_id] = result
        
        # Trier par score combiné
        results = list(merged.values())
        results.sort(key=lambda x: x.score, reverse=True)
        
        return results
    
    def _build_elastic_query(
        self,
        query: SearchQuery,
        processed_query: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Construit la requête Elasticsearch."""
        # Utiliser la requête étendue si disponible
        search_text = processed_query.get("expanded_query", query.query)
        
        # Construire une requête multi_match
        es_query = {
            "multi_match": {
                "query": search_text,
                "fields": [
                    "searchable_text^3",
                    "primary_description^2",
                    "merchant_name^2",
                    "category_name"
                ],
                "type": "best_fields",
                "operator": "or",
                "fuzziness": "AUTO"
            }
        }
        
        # Ajouter des boosters pour les mots-clés importants
        if processed_query.get("keywords"):
            es_query = {
                "bool": {
                    "should": [
                        es_query,
                        {
                            "terms": {
                                "primary_description": processed_query["keywords"],
                                "boost": 2.0
                            }
                        }
                    ]
                }
            }
        
        return es_query
    
    def _build_filters(self, query: SearchQuery) -> Dict[str, Any]:
        """Construit les filtres pour la recherche."""
        filters = {}
        
        if query.date_from:
            filters["date_from"] = query.date_from.isoformat()
        if query.date_to:
            filters["date_to"] = query.date_to.isoformat()
        if query.amount_min is not None:
            filters["amount_min"] = query.amount_min
        if query.amount_max is not None:
            filters["amount_max"] = query.amount_max
        if query.categories:
            filters["categories"] = query.categories
        if query.account_ids:
            filters["account_ids"] = query.account_ids
        if query.transaction_types:
            filters["transaction_types"] = query.transaction_types
        
        # Ajouter les filtres personnalisés
        if query.filters:
            filters.update(query.filters)
        
        return filters
    
    def _elastic_hit_to_result(
        self,
        hit: Dict[str, Any],
        include_explanations: bool
    ) -> SearchResult:
        """Convertit un hit Elasticsearch en SearchResult."""
        source = hit["_source"]
        
        result = SearchResult(
            transaction_id=source["transaction_id"],
            user_id=source["user_id"],
            description=source["primary_description"],
            amount=source["amount"],
            date=source["date"],
            currency=source.get("currency_code", "EUR"),
            category_id=source.get("category_id"),
            category_name=source.get("category_name"),
            merchant_name=source.get("merchant_name"),
            score=hit["_score"],
            highlights=hit.get("highlight"),
            explanations=hit.get("_explanation") if include_explanations else None,
            metadata=source.get("metadata", {})
        )
        
        return result
    
    def _qdrant_hit_to_result(
        self,
        hit: Dict[str, Any],
        include_explanations: bool
    ) -> SearchResult:
        """Convertit un hit Qdrant en SearchResult."""
        payload = hit["payload"]
        
        result = SearchResult(
            transaction_id=payload["transaction_id"],
            user_id=payload["user_id"],
            description=payload["primary_description"],
            amount=payload["amount"],
            date=payload["date"],
            currency=payload.get("currency_code", "EUR"),
            category_id=payload.get("category_id"),
            category_name=payload.get("category_name"),
            merchant_name=payload.get("merchant_name"),
            score=hit["score"],
            highlights=None,  # Qdrant ne fournit pas de highlights
            explanations={"vector_similarity": hit["score"]} if include_explanations else None,
            metadata=payload.get("metadata", {})
        )
        
        return result
    
    def _get_applied_filters(self, query: SearchQuery) -> Dict[str, Any]:
        """Retourne les filtres effectivement appliqués."""
        applied = {}
        
        if query.date_from:
            applied["date_from"] = query.date_from.isoformat()
        if query.date_to:
            applied["date_to"] = query.date_to.isoformat()
        if query.amount_min is not None:
            applied["amount_min"] = query.amount_min
        if query.amount_max is not None:
            applied["amount_max"] = query.amount_max
        if query.categories:
            applied["categories"] = query.categories
        if query.account_ids:
            applied["account_ids"] = query.account_ids
        if query.transaction_types:
            applied["transaction_types"] = query.transaction_types
        
        return applied