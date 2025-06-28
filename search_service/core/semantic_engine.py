"""
Moteur de recherche sémantique pour Qdrant.

Ce module implémente la recherche sémantique optimisée pour les transactions
financières, basé sur les résultats du validateur harena_search_validator.
"""
import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from search_service.clients.qdrant_client import QdrantClient
from search_service.core.embeddings import EmbeddingManager
from search_service.core.query_processor import QueryProcessor, QueryAnalysis
from search_service.models.search_types import SearchQuality, SortOrder
from search_service.models.responses import SearchResultItem

logger = logging.getLogger(__name__)


@dataclass
class SemanticSearchConfig:
    """Configuration pour la recherche sémantique."""
    max_results: int = 50
    default_similarity_threshold: float = 0.5
    strict_similarity_threshold: float = 0.7
    loose_similarity_threshold: float = 0.3
    enable_filtering: bool = True
    fallback_to_unfiltered: bool = True
    recommendation_threshold: float = 0.6


@dataclass
class SemanticSearchResult:
    """Résultat d'une recherche sémantique."""
    results: List[SearchResultItem]
    total_found: int
    max_score: float
    avg_score: float
    min_score: float
    processing_time_ms: float
    quality: SearchQuality
    query_embedding: Optional[List[float]]
    filtering_method: str  # "qdrant_native", "manual", "none"
    debug_info: Optional[Dict[str, Any]] = None


class SemanticSearchEngine:
    """
    Moteur de recherche sémantique optimisé pour Qdrant.
    
    Implémente les corrections identifiées par le validateur:
    - Fallback au filtrage manuel si Qdrant filtering échoue
    - Seuils de similarité adaptatifs
    - Gestion des erreurs robuste
    - Métriques de qualité améliorées
    """
    
    def __init__(
        self,
        qdrant_client: QdrantClient,
        embedding_manager: EmbeddingManager,
        query_processor: QueryProcessor,
        config: Optional[SemanticSearchConfig] = None
    ):
        self.qdrant_client = qdrant_client
        self.embedding_manager = embedding_manager
        self.query_processor = query_processor
        self.config = config or SemanticSearchConfig()
        
        # Métriques
        self.search_count = 0
        self.total_processing_time = 0.0
        self.error_count = 0
        self.fallback_count = 0
        self.embedding_generation_time = 0.0
        
        logger.info("Semantic search engine initialized")
    
    async def search(
        self,
        query: str,
        user_id: int,
        limit: int = 15,
        offset: int = 0,
        similarity_threshold: Optional[float] = None,
        sort_order: SortOrder = SortOrder.RELEVANCE,
        filters: Optional[Dict[str, Any]] = None,
        debug: bool = False
    ) -> SemanticSearchResult:
        """
        Effectue une recherche sémantique optimisée.
        
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
        """
        start_time = time.time()
        self.search_count += 1
        
        try:
            # 1. Analyser et traiter la requête
            query_analysis = self.query_processor.process_query(query)
            optimized_query = self.query_processor.optimize_for_semantic_search(query_analysis)
            
            # 2. Générer l'embedding pour la requête
            embedding_start = time.time()
            query_embedding = await self.embedding_manager.generate_embedding(
                optimized_query, use_cache=True
            )
            embedding_time = time.time() - embedding_start
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
                limit=min(limit + offset, self.config.max_results),
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
            
            # 8. Construire le résultat final
            semantic_result = SemanticSearchResult(
                results=final_results.results,
                total_found=len(processed_results.results),
                max_score=final_results.max_score,
                avg_score=final_results.avg_score,
                min_score=final_results.min_score,
                processing_time_ms=processing_time,
                quality=quality,
                query_embedding=query_embedding if debug else None,
                filtering_method=search_results.get("filtering_method", "unknown"),
                debug_info=final_results.debug_info if debug else None
            )
            
            logger.debug(
                f"Semantic search completed: {len(final_results.results)} results, "
                f"quality: {quality}, time: {processing_time:.2f}ms, "
                f"filtering: {semantic_result.filtering_method}"
            )
            
            return semantic_result
            
        except Exception as e:
            self.error_count += 1
            processing_time = (time.time() - start_time) * 1000
            logger.error(f"Semantic search failed: {e}")
            
            return SemanticSearchResult(
                results=[],
                total_found=0,
                max_score=0.0,
                avg_score=0.0,
                min_score=0.0,
                processing_time_ms=processing_time,
                quality=SearchQuality.FAILED,
                query_embedding=None,
                filtering_method="failed",
                debug_info={"error": str(e)} if debug else None
            )
    
    def _determine_optimal_threshold(self, query_analysis: QueryAnalysis) -> float:
        """Détermine le seuil de similarité optimal basé sur l'analyse de requête."""
        # Seuil par défaut
        threshold = self.config.default_similarity_threshold
        
        # Ajustements basés sur le type de requête
        query_type = query_analysis.query_type
        
        if query_type in ["amount_search", "date_search"]:
            # Requêtes avec entités spécifiques = seuil plus permissif
            threshold = self.config.loose_similarity_threshold
        elif query_type in ["category_search", "merchant_query"]:
            # Requêtes conceptuelles = seuil standard
            threshold = self.config.default_similarity_threshold
        elif query_type == "free_text" and len(query_analysis.cleaned_query.split()) == 1:
            # Mot unique = seuil strict
            threshold = self.config.strict_similarity_threshold
        
        # Ajustements basés sur la confiance
        if query_analysis.confidence < 0.5:
            # Faible confiance = seuil plus permissif
            threshold = max(threshold - 0.1, self.config.loose_similarity_threshold)
        elif query_analysis.confidence > 0.8:
            # Haute confiance = seuil plus strict
            threshold = min(threshold + 0.1, self.config.strict_similarity_threshold)
        
        return threshold
    
    async def _search_with_fallback_strategy(
        self,
        query_embedding: List[float],
        user_id: int,
        limit: int,
        similarity_threshold: float,
        filters: Optional[Dict[str, Any]] = None,
        debug: bool = False
    ) -> Dict[str, Any]:
        """
        Effectue la recherche avec stratégie de fallback basée sur le validateur.
        
        Le validateur montre des problèmes avec le filtrage Qdrant,
        donc on implémente un fallback au filtrage manuel.
        """
        # Stratégie 1: Recherche avec filtrage Qdrant natif
        if self.config.enable_filtering:
            try:
                qdrant_results = await self.qdrant_client.search_similar_transactions(
                    query_vector=query_embedding,
                    user_id=user_id,
                    limit=limit,
                    score_threshold=similarity_threshold,
                    filters=filters,
                    with_payload=True,
                    with_vector=False
                )
                
                results = qdrant_results.get("result", [])
                
                # Si on a des résultats satisfaisants, les retourner
                if len(results) >= min(3, limit // 2):
                    return {
                        "result": results,
                        "filtering_method": "qdrant_native",
                        "total_found": len(results)
                    }
                else:
                    logger.debug(f"Qdrant native filtering returned only {len(results)} results, trying fallback")
            
            except Exception as e:
                logger.warning(f"Qdrant native filtering failed: {e}")
        
        # Stratégie 2: Fallback au filtrage manuel (comme observé dans le validateur)
        if self.config.fallback_to_unfiltered:
            try:
                self.fallback_count += 1
                
                # Recherche sans filtre avec limite élargie
                fallback_results = await self.qdrant_client.search_similar_transactions(
                    query_vector=query_embedding,
                    user_id=user_id,
                    limit=limit * 3,  # Chercher plus pour avoir assez après filtrage
                    score_threshold=max(similarity_threshold - 0.1, 0.3),  # Seuil plus permissif
                    filters=None,  # Pas de filtres Qdrant
                    with_payload=True,
                    with_vector=False
                )
                
                all_results = fallback_results.get("result", [])
                
                # Filtrage manuel par user_id
                user_results = [
                    point for point in all_results
                    if point.get("payload", {}).get("user_id") == user_id
                ]
                
                # Appliquer les filtres additionnels manuellement si fournis
                if filters:
                    user_results = self._apply_manual_filters(user_results, filters)
                
                # Limiter au nombre demandé
                final_results = user_results[:limit]
                
                return {
                    "result": final_results,
                    "filtering_method": "manual",
                    "total_found": len(user_results),
                    "unfiltered_count": len(all_results)
                }
                
            except Exception as e:
                logger.error(f"Fallback search also failed: {e}")
        
        # Stratégie 3: Dernière chance - recherche très basique
        try:
            basic_results = await self.qdrant_client.search_similar_transactions(
                query_vector=query_embedding,
                user_id=user_id,
                limit=limit,
                score_threshold=0.3,  # Seuil très permissif
                filters=None,
                with_payload=True,
                with_vector=False
            )
            
            return {
                "result": basic_results.get("result", []),
                "filtering_method": "basic",
                "total_found": len(basic_results.get("result", []))
            }
            
        except Exception as e:
            logger.error(f"All search strategies failed: {e}")
            return {
                "result": [],
                "filtering_method": "failed",
                "total_found": 0,
                "error": str(e)
            }
    
    def _apply_manual_filters(
        self, 
        results: List[Dict[str, Any]], 
        filters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Applique les filtres manuellement sur les résultats."""
        filtered_results = []
        
        for result in results:
            payload = result.get("payload", {})
            
            # Vérifier tous les filtres
            if self._matches_manual_filters(payload, filters):
                filtered_results.append(result)
        
        return filtered_results
    
    def _matches_manual_filters(self, payload: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Vérifie si un payload correspond aux filtres."""
        # Filtre de montant
        amount = payload.get("amount", 0)
        if "amount_min" in filters and amount < filters["amount_min"]:
            return False
        if "amount_max" in filters and amount > filters["amount_max"]:
            return False
        
        # Filtre de date
        date_str = payload.get("date", "")
        if "date_from" in filters and date_str < filters["date_from"]:
            return False
        if "date_to" in filters and date_str > filters["date_to"]:
            return False
        
        # Filtre de catégories
        category_id = payload.get("category_id")
        if "category_ids" in filters and filters["category_ids"]:
            if category_id not in filters["category_ids"]:
                return False
        
        # Filtre de comptes
        account_id = payload.get("account_id")
        if "account_ids" in filters and filters["account_ids"]:
            if account_id not in filters["account_ids"]:
                return False
        
        # Filtre de type de transaction
        transaction_type = payload.get("transaction_type", "")
        if "transaction_type" in filters and filters["transaction_type"] != "all":
            if transaction_type != filters["transaction_type"]:
                return False
        
        return True
    
    def _process_qdrant_results(
        self,
        search_results: Dict[str, Any],
        query_analysis: QueryAnalysis,
        debug: bool = False
    ) -> 'ProcessedSemanticResults':
        """Traite les résultats bruts de Qdrant."""
        points = search_results.get("result", [])
        
        processed_results = []
        scores = []
        
        for point in points:
            result_item = self._convert_qdrant_point_to_result_item(point, query_analysis)
            if result_item:
                processed_results.append(result_item)
                scores.append(result_item.score)
        
        # Calculer les statistiques
        max_score = max(scores) if scores else 0.0
        avg_score = sum(scores) / len(scores) if scores else 0.0
        min_score = min(scores) if scores else 0.0
        
        # Informations de debug
        debug_info = None
        if debug:
            debug_info = {
                "qdrant_results": len(points),
                "processed_results": len(processed_results),
                "filtering_method": search_results.get("filtering_method", "unknown"),
                "total_found": search_results.get("total_found", 0),
                "score_distribution": {
                    "min": min_score,
                    "max": max_score,
                    "avg": avg_score,
                    "count": len(scores)
                },
                "query_analysis": {
                    "original_query": query_analysis.original_query,
                    "optimized_query": query_analysis.expanded_query,
                    "detected_entities": query_analysis.detected_entities
                }
            }
            
            if search_results.get("unfiltered_count"):
                debug_info["unfiltered_count"] = search_results["unfiltered_count"]
        
        return ProcessedSemanticResults(
            results=processed_results,
            total_found=search_results.get("total_found", len(processed_results)),
            max_score=max_score,
            avg_score=avg_score,
            min_score=min_score,
            debug_info=debug_info
        )
    
    def _convert_qdrant_point_to_result_item(
        self,
        point: Dict[str, Any],
        query_analysis: QueryAnalysis
    ) -> Optional[SearchResultItem]:
        """Convertit un point Qdrant en SearchResultItem."""
        try:
            payload = point.get("payload", {})
            score = point.get("score", 0.0)
            
            # Calculer des scores détaillés
            semantic_score = score
            lexical_score = None  # N/A pour recherche sémantique
            combined_score = score
            
            # Créer l'item de résultat
            result_item = SearchResultItem(
                transaction_id=payload.get("transaction_id"),
                user_id=payload.get("user_id"),
                account_id=payload.get("account_id"),
                score=score,
                lexical_score=lexical_score,
                semantic_score=semantic_score,
                combined_score=combined_score,
                primary_description=payload.get("primary_description", ""),
                searchable_text=payload.get("searchable_text", ""),
                merchant_name=payload.get("merchant_name"),
                amount=payload.get("amount", 0.0),
                currency_code=payload.get("currency_code", "EUR"),
                transaction_type=payload.get("transaction_type", ""),
                transaction_date=payload.get("date", ""),
                created_at=payload.get("created_at"),
                category_id=payload.get("category_id"),
                operation_type=payload.get("operation_type"),
                highlights=None,  # Pas de highlighting pour recherche sémantique
                metadata={
                    "qdrant_score": score,
                    "search_type": "semantic",
                    "query_type": query_analysis.query_type,
                    "semantic_relevance": self._calculate_semantic_relevance(payload, query_analysis)
                }
            )
            
            return result_item
            
        except Exception as e:
            logger.warning(f"Failed to convert Qdrant point to result item: {e}")
            return None
    
    def _calculate_semantic_relevance(
        self,
        payload: Dict[str, Any],
        query_analysis: QueryAnalysis
    ) -> Dict[str, Any]:
        """Calcule la pertinence sémantique d'un résultat."""
        relevance = {
            "concept_match": False,
            "entity_match": False,
            "category_match": False
        }
        
        # Texte combiné pour l'analyse
        all_text = f"{payload.get('searchable_text', '')} {payload.get('primary_description', '')}".lower()
        
        # Correspondance conceptuelle
        entities = query_analysis.detected_entities
        if entities.get("categories"):
            for category in entities["categories"]:
                if category.lower() in all_text:
                    relevance["concept_match"] = True
                    relevance["category_match"] = True
                    break
        
        # Correspondance d'entités
        if entities.get("amounts") or entities.get("dates"):
            relevance["entity_match"] = True
        
        return relevance
    
    def _apply_pagination_and_sorting(
        self,
        results: 'ProcessedSemanticResults',
        offset: int,
        limit: int,
        sort_order: SortOrder
    ) -> 'ProcessedSemanticResults':
        """Applique pagination et tri aux résultats."""
        sorted_results = results.results.copy()
        
        # Tri (sémantique privilégie la similarité)
        if sort_order == SortOrder.DATE_DESC:
            sorted_results.sort(
                key=lambda x: (x.transaction_date, x.score), 
                reverse=True
            )
        elif sort_order == SortOrder.DATE_ASC:
            sorted_results.sort(
                key=lambda x: (x.transaction_date, x.score)
            )
        elif sort_order == SortOrder.AMOUNT_DESC:
            sorted_results.sort(
                key=lambda x: (abs(x.amount), x.score), 
                reverse=True
            )
        elif sort_order == SortOrder.AMOUNT_ASC:
            sorted_results.sort(
                key=lambda x: (abs(x.amount), x.score)
            )
        # RELEVANCE: déjà trié par score de similarité
        
        # Pagination
        paginated_results = sorted_results[offset:offset + limit]
        
        return ProcessedSemanticResults(
            results=paginated_results,
            total_found=results.total_found,
            max_score=results.max_score,
            avg_score=results.avg_score,
            min_score=results.min_score,
            debug_info=results.debug_info
        )
    
    def _assess_semantic_quality(
        self,
        results: 'ProcessedSemanticResults',
        query_analysis: QueryAnalysis
    ) -> SearchQuality:
        """Évalue la qualité des résultats sémantiques."""
        if not results.results:
            return SearchQuality.FAILED
        
        quality_score = 0.0
        
        # 1. Score de similarité moyen (plus important pour sémantique)
        if results.avg_score >= 0.7:
            quality_score += 0.4
        elif results.avg_score >= 0.5:
            quality_score += 0.3
        elif results.avg_score >= 0.3:
            quality_score += 0.1
        
        # 2. Nombre de résultats pertinents
        if len(results.results) >= 3:
            quality_score += 0.2
        elif len(results.results) >= 1:
            quality_score += 0.1
        
        # 3. Correspondance conceptuelle
        conceptual_matches = sum(
            1 for r in results.results[:5]
            if r.metadata.get("semantic_relevance", {}).get("concept_match", False)
        )
        if conceptual_matches > 0:
            quality_score += 0.2 * (conceptual_matches / min(5, len(results.results)))
        
        # 4. Cohérence des scores (variance faible = meilleure qualité)
        if len(results.results) > 1:
            scores = [r.score for r in results.results]
            score_variance = sum((s - results.avg_score) ** 2 for s in scores) / len(scores)
            if score_variance < 0.05:  # Scores cohérents
                quality_score += 0.1
        
        # 5. Seuil minimum de similarité respecté
        min_acceptable_score = 0.3
        acceptable_results = sum(1 for r in results.results if r.score >= min_acceptable_score)
        if acceptable_results == len(results.results):
            quality_score += 0.1
        
        # Convertir en enum
        if quality_score >= 0.9:
            return SearchQuality.EXCELLENT
        elif quality_score >= 0.7:
            return SearchQuality.GOOD
        elif quality_score >= 0.5:
            return SearchQuality.MEDIUM
        elif quality_score >= 0.3:
            return SearchQuality.POOR
        else:
            return SearchQuality.FAILED
    
    async def find_similar_transactions(
        self,
        reference_transaction_id: int,
        user_id: int,
        limit: int = 10,
        similarity_threshold: float = 0.6
    ) -> List[SearchResultItem]:
        """Trouve des transactions similaires à une transaction de référence."""
        try:
            similar_results = await self.qdrant_client.get_similar_by_id(
                transaction_id=reference_transaction_id,
                user_id=user_id,
                limit=limit,
                score_threshold=similarity_threshold
            )
            
            # Convertir en SearchResultItem
            result_items = []
            for point in similar_results:
                # Analyse vide pour la conversion
                dummy_analysis = QueryAnalysis(
                    original_query="",
                    cleaned_query="",
                    expanded_query="",
                    detected_entities={},
                    query_type="similarity_search",
                    confidence=1.0,
                    suggested_filters={},
                    processing_notes=[]
                )
                
                item = self._convert_qdrant_point_to_result_item(point, dummy_analysis)
                if item:
                    result_items.append(item)
            
            return result_items
            
        except Exception as e:
            logger.error(f"Failed to find similar transactions: {e}")
            return []
    
    async def get_recommendations(
        self,
        positive_transaction_ids: List[int],
        user_id: int,
        negative_transaction_ids: Optional[List[int]] = None,
        limit: int = 10
    ) -> List[SearchResultItem]:
        """Génère des recommandations basées sur des transactions appréciées."""
        try:
            recommendations = await self.qdrant_client.recommend_transactions(
                positive_ids=positive_transaction_ids,
                user_id=user_id,
                negative_ids=negative_transaction_ids,
                limit=limit,
                score_threshold=self.config.recommendation_threshold
            )
            
            # Convertir en SearchResultItem
            result_items = []
            for point in recommendations:
                dummy_analysis = QueryAnalysis(
                    original_query="",
                    cleaned_query="",
                    expanded_query="",
                    detected_entities={},
                    query_type="recommendation",
                    confidence=1.0,
                    suggested_filters={},
                    processing_notes=[]
                )
                
                item = self._convert_qdrant_point_to_result_item(point, dummy_analysis)
                if item:
                    result_items.append(item)
            
            return result_items
            
        except Exception as e:
            logger.error(f"Failed to generate recommendations: {e}")
            return []
    
    def get_engine_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques du moteur sémantique."""
        avg_processing_time = (
            self.total_processing_time / self.search_count
            if self.search_count > 0 else 0
        )
        
        avg_embedding_time = (
            self.embedding_generation_time / self.search_count
            if self.search_count > 0 else 0
        )
        
        error_rate = self.error_count / self.search_count if self.search_count > 0 else 0
        fallback_rate = self.fallback_count / self.search_count if self.search_count > 0 else 0
        
        return {
            "engine_type": "semantic",
            "search_count": self.search_count,
            "error_count": self.error_count,
            "error_rate": error_rate,
            "fallback_count": self.fallback_count,
            "fallback_rate": fallback_rate,
            "avg_processing_time_ms": avg_processing_time,
            "avg_embedding_time_ms": avg_embedding_time * 1000,
            "config": {
                "max_results": self.config.max_results,
                "default_similarity_threshold": self.config.default_similarity_threshold,
                "enable_filtering": self.config.enable_filtering,
                "fallback_to_unfiltered": self.config.fallback_to_unfiltered
            },
            "qdrant_client_stats": self.qdrant_client.get_metrics(),
            "embedding_manager_stats": self.embedding_manager.get_manager_stats()
        }
    
    def reset_stats(self):
        """Remet à zéro les statistiques."""
        self.search_count = 0
        self.total_processing_time = 0.0
        self.error_count = 0
        self.fallback_count = 0
        self.embedding_generation_time = 0.0
        logger.info("Semantic search engine stats reset")
    
    async def warmup(self, user_id: int) -> bool:
        """Réchauffe le moteur avec des requêtes de test."""
        warmup_queries = [
            "restaurant", "virement bancaire", "carte bancaire",
            "supermarché", "essence", "pharmacie"
        ]
        
        success_count = 0
        
        for query in warmup_queries:
            try:
                result = await self.search(query, user_id, limit=5)
                if result.quality != SearchQuality.FAILED:
                    success_count += 1
                    
                # Petit délai entre les requêtes
                await asyncio.sleep(0.2)
                
            except Exception as e:
                logger.warning(f"Warmup query '{query}' failed: {e}")
        
        warmup_success = success_count >= len(warmup_queries) // 2
        logger.info(f"Semantic engine warmup: {success_count}/{len(warmup_queries)} queries successful")
        
        return warmup_success


@dataclass
class ProcessedSemanticResults:
    """Résultats traités de recherche sémantique."""
    results: List[SearchResultItem]
    total_found: int
    max_score: float
    avg_score: float
    min_score: float
    debug_info: Optional[Dict[str, Any]] = None


# Import pour éviter la référence circulaire
import asyncio