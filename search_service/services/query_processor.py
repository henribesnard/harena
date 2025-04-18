"""
Processeur de requêtes pour le service de recherche.

Ce module est responsable de l'analyse, l'expansion et la structuration
des requêtes de recherche en utilisant DeepSeek pour l'analyse avancée.
"""
import logging
from typing import Optional, Dict, Any, List
import json

from search_service.schemas.query import (
    SearchQuery, SearchParameters
)
from search_service.core.config import settings
from search_service.services.deepseek_processor import process_query_with_deepseek
from search_service.storage.memory_cache import get_cache, set_cache

logger = logging.getLogger(__name__)

async def process_query(
    query: SearchQuery,
    user_id: int,
    db = None
) -> SearchQuery:
    """
    Traite et enrichit une requête de recherche.
    
    Args:
        query: Requête de recherche originale
        user_id: ID de l'utilisateur
        db: Session de base de données (optionnelle)
        
    Returns:
        Requête enrichie et structurée
    """
    # Logger la requête entrante
    logger.info(f"Traitement de la requête: {query.query.text} pour user_id={user_id}")
    
    # Définir les paramètres par défaut si non spécifiés
    if not query.search_params:
        query.search_params = SearchParameters(
            lexical_weight=settings.DEFAULT_LEXICAL_WEIGHT if hasattr(settings, 'DEFAULT_LEXICAL_WEIGHT') else 0.5,
            semantic_weight=settings.DEFAULT_SEMANTIC_WEIGHT if hasattr(settings, 'DEFAULT_SEMANTIC_WEIGHT') else 0.5,
            top_k_initial=settings.DEFAULT_TOP_K_INITIAL if hasattr(settings, 'DEFAULT_TOP_K_INITIAL') else 50,
            top_k_final=settings.DEFAULT_TOP_K_FINAL if hasattr(settings, 'DEFAULT_TOP_K_FINAL') else 10
        )
    
    # Vérifier le cache
    cache_key = f"query_process:{user_id}:{query.query.text}"
    cached_result = await get_cache(cache_key)
    
    if cached_result:
        logger.debug(f"Résultat de traitement de requête récupéré du cache: {query.query.text}")
        return cached_result
    
    # Utiliser DeepSeek pour analyser et enrichir la requête
    if settings.DEEPSEEK_API_KEY:
        processed_query = await process_query_with_deepseek(query, user_id)
    else:
        # Si DeepSeek n'est pas configuré, faire un traitement minimal
        processed_query = query
        if not processed_query.query.expanded_text:
            processed_query.query.expanded_text = query.query.text
            
    # Mettre en cache le résultat
    await set_cache(cache_key, processed_query, ttl=3600)
    
    logger.debug(f"Requête traitée: {processed_query.query.expanded_text or processed_query.query.text}")
    if processed_query.filters:
        logger.debug(f"Filtres détectés: {json.dumps(processed_query.filters.dict() if processed_query.filters else {})}")
    
    return processed_query