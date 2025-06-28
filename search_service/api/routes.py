"""
Routes API pour le service de recherche.

Ce module définit tous les endpoints REST pour la recherche hybride
de transactions financières.
"""
import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query, Path
from fastapi.responses import JSONResponse

from search_service.models import (
    SearchRequest, AdvancedSearchRequest, SuggestionsRequest, StatsRequest,
    BulkSearchRequest, ExplainRequest, HealthCheckRequest,
    SearchResponse, SuggestionsResponse, StatsResponse, HealthResponse,
    BulkSearchResponse, ExplainResponse, ErrorResponse, MetricsResponse,
    SearchType, SortOrder, SearchQuality
)
from search_service.core.search_engine import HybridSearchEngine
from search_service.core.lexical_engine import LexicalSearchEngine
from search_service.core.semantic_engine import SemanticSearchEngine
from search_service.core.query_processor import QueryProcessor
from search_service.clients import ElasticsearchClient, QdrantClient
from search_service.core.embeddings import EmbeddingManager
from search_service.utils.cache import get_cache_metrics
from search_service.api.dependencies import get_current_user, validate_search_request, rate_limit

logger = logging.getLogger(__name__)

# Router principal
router = APIRouter()

# Variables globales injectées depuis main.py
elasticsearch_client: Optional[ElasticsearchClient] = None
qdrant_client: Optional[QdrantClient] = None
embedding_manager: Optional[EmbeddingManager] = None
query_processor: Optional[QueryProcessor] = None
lexical_engine: Optional[LexicalSearchEngine] = None
semantic_engine: Optional[SemanticSearchEngine] = None
hybrid_engine: Optional[HybridSearchEngine] = None


# ==========================================
# ENDPOINTS PRINCIPAUX DE RECHERCHE
# ==========================================

@router.post("/search", response_model=SearchResponse)
async def search_transactions(
    request: SearchRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    _: None = Depends(rate_limit)
):
    """
    Recherche hybride principale de transactions.
    
    Combine recherche lexicale (Elasticsearch) et sémantique (Qdrant)
    pour fournir les meilleurs résultats possibles.
    """
    # Vérification des permissions
    if request.user_id != current_user.get("id") and not current_user.get("is_superuser", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot search for other users"
        )
    
    try:
        # Initialiser le moteur hybride si pas encore fait
        global hybrid_engine
        if not hybrid_engine:
            hybrid_engine = HybridSearchEngine(
                lexical_engine=lexical_engine,
                semantic_engine=semantic_engine,
                query_processor=query_processor
            )
        
        # Effectuer la recherche
        search_result = await hybrid_engine.search(
            query=request.query,
            user_id=request.user_id,
            search_type=request.search_type,
            limit=request.limit,
            offset=request.offset,
            lexical_weight=request.lexical_weight,
            semantic_weight=request.semantic_weight,
            similarity_threshold=request.similarity_threshold,
            sort_order=request.sort_order,
            filters={
                "transaction_type": request.transaction_type.value,
                "account_ids": request.account_ids,
                "category_ids": request.category_ids
            },
            use_cache=True,
            debug=False
        )
        
        # Construire la réponse
        response = SearchResponse(
            query=request.query,
            search_type=search_result.search_type,
            user_id=request.user_id,
            results=search_result.results,
            total_found=search_result.total_found,
            returned_count=len(search_result.results),
            offset=request.offset,
            limit=request.limit,
            has_more=search_result.total_found > (request.offset + len(search_result.results)),
            processing_time_ms=search_result.processing_time_ms,
            search_quality=search_result.quality,
            lexical_results_count=search_result.lexical_results_count,
            semantic_results_count=search_result.semantic_results_count,
            cache_hit=search_result.cache_hit
        )
        
        logger.info(f"Search completed: '{request.query}' for user {request.user_id} - {len(search_result.results)} results")
        return response
        
    except Exception as e:
        logger.error(f"Search failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )


@router.get("/lexical")
async def lexical_search(
    query: str = Query(..., min_length=1, max_length=500),
    user_id: int = Query(...),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    sort_order: SortOrder = Query(default=SortOrder.RELEVANCE),
    current_user: Dict[str, Any] = Depends(get_current_user),
    _: None = Depends(rate_limit)
):
    """
    Recherche lexicale pure via Elasticsearch.
    
    Utilise uniquement la recherche textuelle sans analyse sémantique.
    Idéal pour les requêtes avec termes exacts ou noms de marchands.
    """
    if user_id != current_user.get("id") and not current_user.get("is_superuser", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot search for other users"
        )
    
    if not lexical_engine:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Lexical search engine not available"
        )
    
    try:
        lexical_result = await lexical_engine.search(
            query=query,
            user_id=user_id,
            limit=limit,
            offset=offset,
            sort_order=sort_order,
            debug=False
        )
        
        response = SearchResponse(
            query=query,
            search_type=SearchType.LEXICAL,
            user_id=user_id,
            results=lexical_result.results,
            total_found=lexical_result.total_found,
            returned_count=len(lexical_result.results),
            offset=offset,
            limit=limit,
            has_more=lexical_result.total_found > (offset + len(lexical_result.results)),
            processing_time_ms=lexical_result.processing_time_ms,
            search_quality=lexical_result.quality,
            lexical_results_count=len(lexical_result.results),
            semantic_results_count=0
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Lexical search failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lexical search failed: {str(e)}"
        )


@router.get("/semantic")
async def semantic_search(
    query: str = Query(..., min_length=1, max_length=500),
    user_id: int = Query(...),
    limit: int = Query(default=15, ge=1, le=50),
    offset: int = Query(default=0, ge=0),
    similarity_threshold: float = Query(default=0.5, ge=0.1, le=0.95),
    sort_order: SortOrder = Query(default=SortOrder.RELEVANCE),
    current_user: Dict[str, Any] = Depends(get_current_user),
    _: None = Depends(rate_limit)
):
    """
    Recherche sémantique pure via Qdrant.
    
    Utilise les embeddings pour trouver des transactions conceptuellement similaires.
    Idéal pour les recherches par intention ou catégorie.
    """
    if user_id != current_user.get("id") and not current_user.get("is_superuser", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot search for other users"
        )
    
    if not semantic_engine:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Semantic search engine not available"
        )
    
    try:
        semantic_result = await semantic_engine.search(
            query=query,
            user_id=user_id,
            limit=limit,
            offset=offset,
            similarity_threshold=similarity_threshold,
            sort_order=sort_order,
            debug=False
        )
        
        response = SearchResponse(
            query=query,
            search_type=SearchType.SEMANTIC,
            user_id=user_id,
            results=semantic_result.results,
            total_found=semantic_result.total_found,
            returned_count=len(semantic_result.results),
            offset=offset,
            limit=limit,
            has_more=semantic_result.total_found > (offset + len(semantic_result.results)),
            processing_time_ms=semantic_result.processing_time_ms,
            search_quality=semantic_result.quality,
            lexical_results_count=0,
            semantic_results_count=len(semantic_result.results)
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Semantic search failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Semantic search failed: {str(e)}"
        )


@router.post("/advanced", response_model=SearchResponse)
async def advanced_search(
    request: AdvancedSearchRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    _: None = Depends(rate_limit)
):
    """
    Recherche avancée avec filtres complexes.
    
    Permet d'utiliser des filtres sophistiqués (montants, dates, catégories)
    combinés avec la recherche textuelle.
    """
    if request.user_id != current_user.get("id") and not current_user.get("is_superuser", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot search for other users"
        )
    
    try:
        # Construire les filtres avancés
        advanced_filters = {}
        
        if request.filters:
            advanced_filters.update(request.filters.to_elasticsearch_query())
        
        # Ajouter les filtres simples
        if request.date_from or request.date_to:
            advanced_filters["date_from"] = request.date_from
            advanced_filters["date_to"] = request.date_to
        
        if request.amount_min is not None or request.amount_max is not None:
            advanced_filters["amount_min"] = request.amount_min
            advanced_filters["amount_max"] = request.amount_max
        
        if request.account_ids:
            advanced_filters["account_ids"] = request.account_ids
        
        if request.category_ids:
            advanced_filters["category_ids"] = request.category_ids
        
        # Initialiser le moteur hybride si nécessaire
        global hybrid_engine
        if not hybrid_engine:
            hybrid_engine = HybridSearchEngine(
                lexical_engine=lexical_engine,
                semantic_engine=semantic_engine,
                query_processor=query_processor
            )
        
        # Effectuer la recherche avancée
        search_result = await hybrid_engine.search(
            query=request.query,
            user_id=request.user_id,
            search_type=request.search_type,
            limit=request.limit,
            offset=request.offset,
            lexical_weight=request.lexical_weight,
            semantic_weight=request.semantic_weight,
            similarity_threshold=request.similarity_threshold,
            sort_order=request.sort_order,
            filters=advanced_filters,
            use_cache=request.use_cache,
            debug=request.explain_scoring
        )
        
        response = SearchResponse(
            query=request.query,
            search_type=search_result.search_type,
            user_id=request.user_id,
            results=search_result.results,
            total_found=search_result.total_found,
            returned_count=len(search_result.results),
            offset=request.offset,
            limit=request.limit,
            has_more=search_result.total_found > (request.offset + len(search_result.results)),
            processing_time_ms=search_result.processing_time_ms,
            search_quality=search_result.quality,
            lexical_results_count=search_result.lexical_results_count,
            semantic_results_count=search_result.semantic_results_count,
            cache_hit=search_result.cache_hit
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Advanced search failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Advanced search failed: {str(e)}"
        )


# ==========================================
# ENDPOINTS DE SUGGESTIONS ET AIDE
# ==========================================

@router.post("/suggestions", response_model=SuggestionsResponse)
async def get_search_suggestions(
    request: SuggestionsRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    _: None = Depends(rate_limit)
):
    """
    Auto-complétion et suggestions de recherche.
    
    Fournit des suggestions basées sur les noms de marchands,
    descriptions de transactions et recherches récentes.
    """
    if request.user_id != current_user.get("id") and not current_user.get("is_superuser", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot get suggestions for other users"
        )
    
    if not elasticsearch_client:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Suggestions service not available"
        )
    
    try:
        import time
        start_time = time.time()
        
        # Obtenir les suggestions depuis Elasticsearch
        es_suggestions = await elasticsearch_client.get_suggestions(
            partial_query=request.partial_query,
            user_id=request.user_id,
            max_suggestions=request.max_suggestions
        )
        
        # Traiter les résultats d'agrégation
        suggestions = []
        
        if "aggregations" in es_suggestions:
            aggs = es_suggestions["aggregations"]
            
            # Suggestions de marchands
            if request.include_merchants and "merchants" in aggs:
                for bucket in aggs["merchants"]["buckets"]:
                    suggestions.append({
                        "text": bucket["key"],
                        "type": "merchant",
                        "frequency": bucket["doc_count"],
                        "category": "merchant"
                    })
            
            # Suggestions de descriptions
            if request.include_descriptions and "descriptions" in aggs:
                for bucket in aggs["descriptions"]["buckets"]:
                    # Filtrer les descriptions trop génériques
                    if len(bucket["key"]) > 5:
                        suggestions.append({
                            "text": bucket["key"],
                            "type": "description",
                            "frequency": bucket["doc_count"],
                            "category": "description"
                        })
        
        # Ajouter des suggestions de catégories si activé
        if request.include_categories:
            category_suggestions = [
                "restaurant", "supermarché", "essence", "pharmacie", 
                "virement", "carte bancaire", "abonnement", "shopping"
            ]
            
            for category in category_suggestions:
                if request.partial_query.lower() in category:
                    suggestions.append({
                        "text": category,
                        "type": "category",
                        "frequency": None,
                        "category": "category"
                    })
        
        # Limiter et trier les suggestions
        suggestions.sort(key=lambda x: x.get("frequency", 0), reverse=True)
        suggestions = suggestions[:request.max_suggestions]
        
        processing_time = (time.time() - start_time) * 1000
        
        response = SuggestionsResponse(
            partial_query=request.partial_query,
            suggestions=suggestions,
            processing_time_ms=processing_time
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Suggestions failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Suggestions failed: {str(e)}"
        )


# ==========================================
# ENDPOINTS D'ANALYSE ET SIMILARITÉ
# ==========================================

@router.get("/similar/{transaction_id}")
async def find_similar_transactions(
    transaction_id: int = Path(..., description="ID de la transaction de référence"),
    user_id: int = Query(...),
    limit: int = Query(default=10, ge=1, le=20),
    similarity_threshold: float = Query(default=0.6, ge=0.3, le=0.9),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Trouve des transactions similaires à une transaction donnée.
    
    Utilise la recherche sémantique pour identifier des transactions
    conceptuellement proches de la transaction de référence.
    """
    if user_id != current_user.get("id") and not current_user.get("is_superuser", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot search for other users"
        )
    
    if not semantic_engine:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Semantic search engine not available"
        )
    
    try:
        similar_transactions = await semantic_engine.find_similar_transactions(
            reference_transaction_id=transaction_id,
            user_id=user_id,
            limit=limit,
            similarity_threshold=similarity_threshold
        )
        
        return {
            "reference_transaction_id": transaction_id,
            "similar_transactions": similar_transactions,
            "count": len(similar_transactions),
            "similarity_threshold": similarity_threshold
        }
        
    except Exception as e:
        logger.error(f"Similar transactions search failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Similar transactions search failed: {str(e)}"
        )


@router.post("/recommendations")
async def get_transaction_recommendations(
    positive_transaction_ids: List[int],
    user_id: int,
    negative_transaction_ids: Optional[List[int]] = None,
    limit: int = Query(default=10, ge=1, le=20),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Génère des recommandations basées sur des transactions appréciées.
    
    Utilise l'algorithme de recommandation de Qdrant pour suggérer
    des transactions similaires aux exemples positifs.
    """
    if user_id != current_user.get("id") and not current_user.get("is_superuser", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot get recommendations for other users"
        )
    
    if not semantic_engine:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Recommendation engine not available"
        )
    
    try:
        recommendations = await semantic_engine.get_recommendations(
            positive_transaction_ids=positive_transaction_ids,
            user_id=user_id,
            negative_transaction_ids=negative_transaction_ids,
            limit=limit
        )
        
        return {
            "positive_examples": positive_transaction_ids,
            "negative_examples": negative_transaction_ids,
            "recommendations": recommendations,
            "count": len(recommendations)
        }
        
    except Exception as e:
        logger.error(f"Recommendations failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Recommendations failed: {str(e)}"
        )


# ==========================================
# ENDPOINTS DE MONITORING ET DIAGNOSTICS
# ==========================================

@router.get("/health", response_model=HealthResponse)
async def health_check(request: HealthCheckRequest = Depends()):
    """
    Vérification de santé du service de recherche.
    
    Teste la connectivité et les performances des différents
    composants (Elasticsearch, Qdrant, embeddings).
    """
    try:
        import time
        start_time = time.time()
        
        services = []
        overall_status = "healthy"
        
        # Test Elasticsearch
        if elasticsearch_client:
            try:
                es_health = await elasticsearch_client.health_check()
                services.append({
                    "service_name": "elasticsearch",
                    "status": es_health.get("status", "unknown"),
                    "response_time_ms": es_health.get("response_time", 0) * 1000,
                    "details": es_health
                })
            except Exception as e:
                services.append({
                    "service_name": "elasticsearch",
                    "status": "unhealthy",
                    "error": str(e)
                })
                overall_status = "degraded"
        else:
            services.append({
                "service_name": "elasticsearch",
                "status": "not_configured"
            })
        
        # Test Qdrant
        if qdrant_client:
            try:
                qdrant_health = await qdrant_client.health_check()
                services.append({
                    "service_name": "qdrant",
                    "status": qdrant_health.get("status", "unknown"),
                    "response_time_ms": qdrant_health.get("response_time", 0) * 1000,
                    "details": qdrant_health
                })
            except Exception as e:
                services.append({
                    "service_name": "qdrant",
                    "status": "unhealthy",
                    "error": str(e)
                })
                overall_status = "degraded"
        else:
            services.append({
                "service_name": "qdrant",
                "status": "not_configured"
            })
        
        # Test Embedding Service
        if embedding_manager:
            try:
                embedding_stats = embedding_manager.get_manager_stats()
                services.append({
                    "service_name": "embeddings",
                    "status": "healthy" if embedding_stats["primary"]["usage"]["api_calls"] >= 0 else "unknown",
                    "details": embedding_stats
                })
            except Exception as e:
                services.append({
                    "service_name": "embeddings",
                    "status": "unhealthy",
                    "error": str(e)
                })
                overall_status = "degraded"
        else:
            services.append({
                "service_name": "embeddings",
                "status": "not_configured"
            })
        
        # Compter les services par statut
        healthy_count = sum(1 for s in services if s.get("status") == "healthy")
        unhealthy_count = sum(1 for s in services if s.get("status") == "unhealthy")
        
        response = HealthResponse(
            overall_status=overall_status,
            services=services,
            total_services=len(services),
            healthy_services=healthy_count,
            degraded_services=0,  # On simplifie pour cet exemple
            unhealthy_services=unhealthy_count,
            search_engine_ready=lexical_engine is not None or semantic_engine is not None,
            cache_operational=True  # Cache toujours opérationnel (en mémoire)
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Health check failed: {str(e)}"
        )


@router.get("/metrics", response_model=MetricsResponse)
async def get_service_metrics(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Métriques détaillées du service de recherche.
    
    Accessible uniquement aux administrateurs.
    Fournit des statistiques de performance et d'utilisation.
    """
    if not current_user.get("is_superuser", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    try:
        # Métriques des moteurs de recherche
        search_metrics = {}
        
        if hybrid_engine:
            search_metrics = hybrid_engine.get_engine_stats()
        elif lexical_engine:
            search_metrics = lexical_engine.get_engine_stats()
        elif semantic_engine:
            search_metrics = semantic_engine.get_engine_stats()
        
        # Métriques du cache
        cache_metrics = get_cache_metrics()
        
        # Construire la réponse
        response = MetricsResponse(
            total_searches_24h=search_metrics.get("search_count", 0),
            average_response_time_ms=search_metrics.get("avg_processing_time_ms", 0),
            cache_hit_rate=cache_metrics.get("overall", {}).get("overall_hit_rate", 0),
            error_rate=search_metrics.get("error_rate", 0),
            search_type_distribution=search_metrics.get("fusion_strategy_distribution", {}),
            quality_distribution=search_metrics.get("quality_distribution", {}),
            elasticsearch_health="healthy" if elasticsearch_client else "not_configured",
            qdrant_health="healthy" if qdrant_client else "not_configured",
            openai_health="healthy" if embedding_manager else "not_configured"
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Metrics collection failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Metrics collection failed: {str(e)}"
        )


# ==========================================
# ENDPOINTS D'ADMINISTRATION
# ==========================================

@router.post("/admin/clear-cache")
async def clear_search_cache(
    cache_type: Optional[str] = Query(default=None, description="Type de cache à vider"),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Vide le cache de recherche.
    
    Accessible uniquement aux administrateurs.
    """
    if not current_user.get("is_superuser", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    try:
        from search_service.utils.cache import global_cache
        
        if cache_type:
            global_cache.clear_cache(cache_type)
            message = f"Cache {cache_type} cleared"
        else:
            global_cache.clear_all()
            message = "All caches cleared"
        
        if hybrid_engine:
            hybrid_engine.clear_cache()
        
        return {"message": message, "timestamp": time.time()}
        
    except Exception as e:
        logger.error(f"Cache clear failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Cache clear failed: {str(e)}"
        )


@router.post("/admin/warmup")
async def warmup_search_engines(
    user_id: int = Query(..., description="User ID pour les tests de warmup"),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Réchauffe les moteurs de recherche.
    
    Accessible uniquement aux administrateurs.
    """
    if not current_user.get("is_superuser", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    try:
        warmup_results = {}
        
        if hybrid_engine:
            warmup_results = await hybrid_engine.warmup(user_id)
        else:
            if lexical_engine:
                warmup_results["lexical"] = await lexical_engine.warmup(user_id)
            if semantic_engine:
                warmup_results["semantic"] = await semantic_engine.warmup(user_id)
        
        return {
            "message": "Warmup completed",
            "results": warmup_results,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Warmup failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Warmup failed: {str(e)}"
        )


# ==========================================
# GESTIONNAIRE D'ERREURS
# ==========================================

@router.exception_handler(HTTPException)
async def search_http_exception_handler(request, exc):
    """Gestionnaire d'erreurs HTTP spécifique à la recherche."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error="search_error",
            message=exc.detail,
            details={"status_code": exc.status_code}
        ).dict()
    )


@router.exception_handler(Exception)
async def search_general_exception_handler(request, exc):
    """Gestionnaire d'erreurs générales pour la recherche."""
    logger.error(f"Unhandled search error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="internal_search_error",
            message="Une erreur interne s'est produite lors de la recherche",
            details={"type": type(exc).__name__}
        ).dict()
    )


# Import pour time
import time