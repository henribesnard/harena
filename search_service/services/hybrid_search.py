"""
Moteur de recherche hybride pour les données financières.

Ce module gère l'exécution parallèle des recherches lexicales et vectorielles,
en optimisant les requêtes pour chaque moteur de recherche.
"""
import logging
import asyncio
from typing import Tuple, List, Dict, Any, Optional

from search_service.schemas.query import SearchQuery, SearchType
from search_service.storage.elasticsearch import get_es_client
from search_service.storage.qdrant import get_qdrant_client
from search_service.services.embedding_service import EmbeddingService
from search_service.storage.memory_cache import get_cache, set_cache
from search_service.core.config import settings

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
    Exécute une recherche lexicale via Elasticsearch.
    
    Args:
        query: Requête de recherche
        user_id: ID de l'utilisateur
        top_k: Nombre de résultats à retourner
        
    Returns:
        Liste des résultats de recherche lexicale
    """
    es_client = await get_es_client()
    
    # Utiliser le texte enrichi si disponible
    search_text = query.query.expanded_text or query.query.text
    
    # Préparer la requête Elasticsearch avec BM25F
    es_query = {
        "query": {
            "bool": {
                "must": [
                    {"term": {"user_id": user_id}},
                    {"multi_match": {
                        "query": search_text,
                        "fields": ["description^3", "merchant_name^4", "category^2", "clean_description^3.5"],
                        "type": "best_fields",
                        "operator": "or",
                        "fuzziness": "AUTO"
                    }}
                ]
            }
        },
        "size": top_k,
        "highlight": {
            "fields": {
                "description": {},
                "merchant_name": {},
                "clean_description": {}
            },
            "pre_tags": ["<em>"],
            "post_tags": ["</em>"]
        }
    }
    
    # Ajouter les filtres structurés si présents
    if query.filters:
        filter_clauses = []
        
        # Filtre de dates
        if query.filters.date_range:
            date_range = {}
            if query.filters.date_range.start:
                date_range["gte"] = query.filters.date_range.start.isoformat()
            if query.filters.date_range.end:
                date_range["lte"] = query.filters.date_range.end.isoformat()
            
            if date_range:
                filter_clauses.append({"range": {"transaction_date": date_range}})
        
        # Filtre de montants
        if query.filters.amount_range:
            amount_range = {}
            if query.filters.amount_range.min is not None:
                amount_range["gte"] = query.filters.amount_range.min
            if query.filters.amount_range.max is not None:
                amount_range["lte"] = query.filters.amount_range.max
            
            if amount_range:
                filter_clauses.append({"range": {"amount": amount_range}})
        
        # Filtre de catégories
        if query.filters.categories:
            if len(query.filters.categories) == 1:
                filter_clauses.append({"term": {"category": query.filters.categories[0]}})
            else:
                filter_clauses.append({"terms": {"category": query.filters.categories}})
        
        # Filtre de marchands
        if query.filters.merchants:
            merchant_clauses = []
            for merchant in query.filters.merchants:
                merchant_clauses.append({"match_phrase": {"merchant_name": merchant}})
            
            filter_clauses.append({"bool": {"should": merchant_clauses, "minimum_should_match": 1}})
        
        # Filtre de types d'opération
        if query.filters.operation_types:
            op_values = []
            for op_type in query.filters.operation_types:
                if op_type.value == "debit":
                    # Les transactions débit sont négatives
                    filter_clauses.append({"range": {"amount": {"lt": 0}}})
                elif op_type.value == "credit":
                    # Les transactions crédit sont positives
                    filter_clauses.append({"range": {"amount": {"gt": 0}}})
        
        # Ajouter tous les filtres à la requête
        if filter_clauses:
            es_query["query"]["bool"]["filter"] = filter_clauses
    
    try:
        # Exécuter la requête
        logger.debug(f"Exécution de la recherche lexicale pour: {search_text}")
        response = await es_client.search(
            index=f"transactions_{user_id}",
            body=es_query
        )
        
        # Traiter les résultats
        results = []
        for hit in response["hits"]["hits"]:
            result = {
                "id": hit["_id"],
                "type": "transaction",
                "content": hit["_source"],
                "score": hit["_score"],
                "match_details": {
                    "lexical_score": hit["_score"]
                },
                "highlight": hit.get("highlight", {})
            }
            results.append(result)
        
        logger.info(f"Recherche lexicale: {len(results)} résultats trouvés")
        return results
        
    except Exception as e:
        logger.error(f"Erreur lors de la recherche Elasticsearch: {str(e)}", exc_info=True)
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
    
    try:
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
        logger.error(f"Erreur lors de la recherche Qdrant: {str(e)}", exc_info=True)
        return []