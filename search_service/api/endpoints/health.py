"""
Endpoints de santé pour le service de recherche.

Ce module fournit des endpoints pour vérifier l'état du service
et obtenir des informations de diagnostic.
"""
from fastapi import APIRouter, Depends
from typing import Dict, Any

from user_service.api.deps import get_current_active_user
from user_service.models.user import User

from search_service.storage.memory_cache import get_cache_stats
from search_service.storage.unified_engine import get_unified_engine
from search_service.storage.qdrant import get_qdrant_client
from config_service.config import settings

import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/")
async def health_check():
    """
    Vérification de base de la santé du service de recherche.
    """
    return {
        "status": "ok",
        "service": "search_service",
        "version": "1.0.0"
    }

@router.get("/detailed")
async def detailed_health_check():
   """
   Vérification détaillée de la santé du service et de ses dépendances.
   """
   health_status = {
       "service": "search_service",
       "status": "ok",
       "components": {}
   }
   
   # Vérifier le moteur de recherche unifié
   try:
       engine = get_unified_engine()
       stats = engine.get_stats()
       health_status["components"]["search_engine"] = {
           "status": "ok",
           "primary_engine": stats["primary_engine"],
           "engines_available": list(stats["engines"].keys()),
           "total_documents": sum(engine.get("total_documents", 0) for engine_name, engine in stats["engines"].items() if isinstance(engine, dict)),
           "total_users": stats["engines"].get(stats["primary_engine"], {}).get("total_users", 0)
       }
   except Exception as e:
       health_status["components"]["search_engine"] = {
           "status": "error",
           "message": str(e)
       }
       health_status["status"] = "degraded"
   
   # Vérifier Qdrant pour la recherche vectorielle
   try:
       qdrant_client = await get_qdrant_client()
       if qdrant_client:
           collections = qdrant_client.get_collections()
           health_status["components"]["qdrant"] = {
               "status": "ok",
               "collections_count": len(collections.collections)
           }
       else:
           health_status["components"]["qdrant"] = {
               "status": "unavailable",
               "message": "Qdrant client not initialized"
           }
           health_status["status"] = "degraded"
   except Exception as e:
       health_status["components"]["qdrant"] = {
           "status": "error",
           "message": str(e)
       }
       health_status["status"] = "degraded"
   
   # Vérifier DeepSeek
   if settings.DEEPSEEK_API_KEY:
       health_status["components"]["deepseek"] = {
           "status": "configured",
           "models": {
               "reasoner": settings.DEEPSEEK_REASONER_MODEL,
               "chat": settings.DEEPSEEK_CHAT_MODEL
           }
       }
   else:
       health_status["components"]["deepseek"] = {
           "status": "unconfigured",
           "message": "DeepSeek API key not set"
       }
       health_status["status"] = "degraded"
   
   # Vérifier le cache en mémoire
   try:
       cache_stats = await get_cache_stats()
       health_status["components"]["memory_cache"] = {
           "status": "ok",
           "stats": cache_stats
       }
   except Exception as e:
       health_status["components"]["memory_cache"] = {
           "status": "error",
           "message": str(e)
       }
       health_status["status"] = "degraded"
   
   return health_status

@router.get("/stats", dependencies=[Depends(get_current_active_user)])
async def service_stats():
   """
   Statistiques d'utilisation du service (nécessite authentification).
   """
   try:
       # Obtenir les statistiques du cache
       cache_stats = await get_cache_stats()
       
       # Obtenir les statistiques des moteurs de recherche
       engine = get_unified_engine()
       engine_stats = engine.get_stats()
       
       return {
           "cache": cache_stats,
           "search_engines": {
               "primary_engine": engine_stats["primary_engine"],
               "usage_stats": engine_stats["usage_stats"],
               "engines": {name: {"total_documents": stats.get("total_documents", 0), 
                                  "total_users": stats.get("total_users", 0)}
                          for name, stats in engine_stats["engines"].items() 
                          if isinstance(stats, dict)}
           },
           "vector_search_available": engine_stats.get("vector_search_available", False),
           "config": {
               "batch_size": settings.BATCH_SIZE,
               "deepseek_timeout": settings.DEEPSEEK_TIMEOUT
           }
       }
   except Exception as e:
       logger.error(f"Erreur lors de la récupération des statistiques: {str(e)}", exc_info=True)
       return {
           "status": "error",
           "message": str(e)
       }

@router.get("/reindex/{user_id}", dependencies=[Depends(get_current_active_user)])
async def reindex_user_data(user_id: int):
    """
    Force la réindexation des données d'un utilisateur.
    
    Args:
        user_id: ID de l'utilisateur à réindexer
    """
    try:
        from search_service.utils.indexer import reindex_user_data
        
        result = await reindex_user_data(user_id)
        
        return {
            "status": result["status"],
            "user_id": user_id,
            "message": "Reindexing complete" if result["status"] == "success" else "Partial reindexing",
            "details": result
        }
    except Exception as e:
        logger.error(f"Erreur lors de la réindexation pour l'utilisateur {user_id}: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "user_id": user_id,
            "message": str(e)
        }