from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any

from user_service.db.session import get_db
from user_service.api.deps import get_current_active_user
from user_service.models.user import User

from search_service.schemas.query import SearchQuery
from search_service.schemas.response import SearchResponse
from search_service.services.query_processor import process_query
from search_service.services.hybrid_search import execute_hybrid_search
from search_service.services.result_fusion import fuse_results
from search_service.services.reranker import rerank_results
from search_service.services.structured_filter import apply_filters
from search_service.services.aggregation import calculate_aggregations
from search_service.utils.timing import timer
from search_service.utils.metrics import record_search_metrics
from config_service.config import settings

import logging
import time
from datetime import datetime

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/", response_model=SearchResponse)
async def search(
    query: SearchQuery,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Endpoint principal de recherche hybride.
    Exécute une recherche combinant lexicale, vectorielle, filtres structurés et agrégations.
    """
    start_time = time.time()
    user_id = current_user.id
    
    logger.info(f"Recherche pour utilisateur {user_id}: {query.query.text}")
    
    try:
        # Étape 1: Traitement et expansion de la requête avec DeepSeek
        with timer("query_processing"):
            processed_query = await process_query(query, user_id, db)
        
        # Étape 2: Exécution de la recherche hybride
        with timer("hybrid_search"):
            lexical_results, vector_results = await execute_hybrid_search(
                processed_query, 
                user_id,
                top_k=processed_query.search_params.top_k_initial
            )
        
        # Étape 3: Fusion des résultats
        with timer("result_fusion"):
            fused_results = await fuse_results(
                lexical_results,
                vector_results,
                processed_query.search_params.lexical_weight,
                processed_query.search_params.semantic_weight
            )
        
        # Étape 4: Reranking des résultats
        with timer("reranking"):
            reranked_results = await rerank_results(
                processed_query.query.text,
                fused_results,
                top_k=processed_query.search_params.top_k_final
            )
        
        # Étape 5: Application des filtres structurés
        with timer("filtering"):
            filtered_results = await apply_filters(
                reranked_results,
                processed_query.filters
            )
        
        # Étape 6: Calcul des agrégations si nécessaire
        aggregations = None
        if processed_query.aggregation:
            with timer("aggregation"):
                aggregations = await calculate_aggregations(
                    filtered_results,
                    processed_query.aggregation
                )
        
        # Préparation de la réponse
        execution_time_ms = int((time.time() - start_time) * 1000)
        
        response = SearchResponse(
            results=filtered_results,
            aggregations=aggregations,
            metadata={
                "total_count": len(fused_results),
                "filtered_count": len(filtered_results),
                "returned_count": len(filtered_results),
                "execution_time_ms": execution_time_ms,
                "data_freshness": datetime.now(),
                "search_type": "hybrid_reranked",
                "query_parsed": processed_query.query.expanded_text or processed_query.query.text
            },
            pagination=None  # À implémenter si nécessaire
        )
        
        # Enregistrement des métriques en arrière-plan
        background_tasks.add_task(
            record_search_metrics,
            user_id=user_id,
            query_text=query.query.text,
            results_count=len(filtered_results),
            execution_time_ms=execution_time_ms
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Erreur lors de la recherche: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur lors de la recherche: {str(e)}")

@router.post("/feedback", response_model=Dict[str, Any])
async def search_feedback(
    result_id: str,
    relevance_score: int = Query(..., ge=1, le=5),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Endpoint pour collecter le feedback utilisateur sur les résultats de recherche.
    Utilisé pour améliorer les modèles de recherche.
    """
    user_id = current_user.id
    
    # Enregistrer le feedback
    # Implémentation à faire
    
    return {"status": "success", "message": "Feedback enregistré"}