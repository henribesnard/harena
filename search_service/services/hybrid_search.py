"""
Moteur de recherche hybride pour les données financières.

Ce module gère l'exécution parallèle des recherches lexicales et vectorielles,
en optimisant les requêtes pour chaque moteur de recherche.
"""
import logging
import asyncio
from typing import Tuple, List, Dict, Any, Optional

from search_service.schemas.query import SearchQuery, SearchType
from search_service.storage.unified_engine import get_unified_engine, SearchEngineType
from search_service.storage.qdrant import get_qdrant_client
from search_service.services.embedding_service import EmbeddingService
from search_service.storage.memory_cache import get_cache, set_cache
from search_service.utils.field_weights import adjust_weights_for_query
from config_service.config import settings

logger = logging.getLogger(__name__)

async def execute_hybrid_search(
    query: SearchQuery,
    user_id: int,
    top_k: int = 50
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Exécute à la fois une recherche lexicale et vectorielle.
    
    Args:
        query: Requête de recherche
        user_id: ID de l'utilisateur
        top_k: Nombre de résultats à retourner
        
    Returns:
        Tuple contenant (résultats_lexicaux, résultats_vectoriels)
    """
    # Vérifier si le type de recherche est spécifié
    search_type = query.query.type or SearchType.HYBRID
    
    # Vérifier le cache
    cache_key = f"search:{user_id}:{query.query.text}:{search_type.value}"
    cached_results = await get_cache(cache_key)
    
    if cached_results:
        logger.debug(f"Résultats récupérés du cache pour requête: {query.query.text}")
        return cached_results
    
    # Exécuter les recherches selon le type demandé
    if search_type == SearchType.LEXICAL:
        # Recherche lexicale uniquement
        lexical_results = await execute_lexical_search(query, user_id, top_k)
        vector_results = []
    elif search_type == SearchType.SEMANTIC:
        # Recherche vectorielle uniquement
        lexical_results = []
        vector_results = await execute_vector_search(query, user_id, top_k)
    else:  # HYBRID (défaut)
        # Exécuter les recherches en parallèle
        lexical_task = asyncio.create_task(execute_lexical_search(query, user_id, top_k))
        vector_task = asyncio.create_task(execute_vector_search(query, user_id, top_k))
        
        # Attendre les résultats
        lexical_results, vector_results = await asyncio.gather(lexical_task, vector_task)
    
    # Mettre en cache les résultats
    await set_cache(cache_key, (lexical_results, vector_results), ttl=3600)
    
    return lexical_results, vector_results

async def execute_lexical_search(
    query: SearchQuery,
    user_id: int,
    top_k: int = 50
) -> List[Dict[str, Any]]:
    """
    Exécute une recherche lexicale via le moteur BM25/Whoosh.
    
    Args:
        query: Requête de recherche
        user_id: ID de l'utilisateur
        top_k: Nombre de résultats à retourner
        
    Returns:
        Liste des résultats de recherche lexicale
    """
    # Utiliser le texte enrichi si disponible
    search_text = query.query.expanded_text or query.query.text
    
    try:
        # Obtenir l'instance du moteur de recherche unifié
        engine = get_unified_engine()
        
        # Ajuster les poids des champs en fonction de la requête
        field_weights = adjust_weights_for_query(search_text)
        
        # Exécuter la recherche
        logger.debug(f"Exécution de la recherche lexicale pour: {search_text}")
        results = await engine.search(
            user_id=user_id,
            query_text=search_text,
            engine_type=SearchEngineType.WHOOSH,  # Utiliser Whoosh par défaut
            field_weights=field_weights,
            top_k=top_k,
            filters=query.filters.dict() if query.filters else None
        )
        
        logger.info(f"Recherche lexicale: {len(results)} résultats trouvés")
        return results
        
    except Exception as e:
        logger.error(f"Erreur lors de la recherche lexicale: {str(e)}", exc_info=True)
        return []

async def execute_vector_search(
    query: SearchQuery,
    user_id: int,
    top_k: int = 50
) -> List[Dict[str, Any]]:
    """
    Exécute une recherche vectorielle via Qdrant.
    
    Args:
        query: Requête de recherche
        user_id: ID de l'utilisateur
        top_k: Nombre de résultats à retourner
        
    Returns:
        Liste des résultats de recherche vectorielle
    """
    try:
        import qdrant_client.models as qmodels
        
        qdrant_client = await get_qdrant_client()
        embedding_service = EmbeddingService()
        
        # Générer l'embedding pour la requête
        query_vector = await embedding_service.get_embedding(query.query.text)
        
        # Préparer les filtres pour Qdrant
        qdrant_filter = qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="user_id",
                    match=qmodels.MatchValue(value=user_id)
                )
            ]
        )
        
        # Ajouter les filtres structurés si présents
        if query.filters:
            # Filtre de dates
            if query.filters.date_range:
                date_range = {}
                if query.filters.date_range.start:
                    date_range["gte"] = query.filters.date_range.start.isoformat()
                if query.filters.date_range.end:
                    date_range["lte"] = query.filters.date_range.end.isoformat()
                
                if date_range:
                    qdrant_filter.must.append(
                        qmodels.FieldCondition(
                            key="transaction_date",
                            range=qmodels.Range(**date_range)
                        )
                    )
            
            # Filtre de montants
            if query.filters.amount_range:
                amount_range = {}
                if query.filters.amount_range.min is not None:
                    amount_range["gte"] = query.filters.amount_range.min
                if query.filters.amount_range.max is not None:
                    amount_range["lte"] = query.filters.amount_range.max
                
                if amount_range:
                    qdrant_filter.must.append(
                        qmodels.FieldCondition(
                            key="amount",
                            range=qmodels.Range(**amount_range)
                        )
                    )
            
            # Filtre de catégories
            if query.filters.categories:
                if len(query.filters.categories) == 1:
                    qdrant_filter.must.append(
                        qmodels.FieldCondition(
                            key="category",
                            match=qmodels.MatchValue(value=query.filters.categories[0])
                        )
                    )
                else:
                    qdrant_filter.must.append(
                        qmodels.FieldCondition(
                            key="category",
                            match=qmodels.MatchAny(any=query.filters.categories)
                        )
                    )
            
            # Filtre de types d'opération
            if query.filters.operation_types:
                for op_type in query.filters.operation_types:
                    if op_type.value == "debit":
                        # Les transactions débit sont négatives
                        qdrant_filter.must.append(
                            qmodels.FieldCondition(
                                key="amount",
                                range=qmodels.Range(lt=0)
                            )
                        )
                    elif op_type.value == "credit":
                        # Les transactions crédit sont positives
                        qdrant_filter.must.append(
                            qmodels.FieldCondition(
                                key="amount",
                                range=qmodels.Range(gt=0)
                            )
                        )
        
        # Exécuter la recherche vectorielle
        logger.debug(f"Exécution de la recherche vectorielle pour: {query.query.text}")
        search_result = await qdrant_client.search(
            collection_name="transactions",
            query_vector=query_vector,
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True
        )
        
        # Traiter les résultats
        results = []
        for hit in search_result:
            result = {
                "id": hit.id,
                "type": "transaction",
                "content": hit.payload,
                "score": hit.score,
                "match_details": {
                    "semantic_score": hit.score
                }
            }
            results.append(result)
        
        logger.info(f"Recherche vectorielle: {len(results)} résultats trouvés")
        return results
        
    except Exception as e:
        logger.error(f"Erreur lors de la recherche vectorielle: {str(e)}", exc_info=True)
        return []