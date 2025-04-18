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
from search_service.storage.elasticsearch import get_es_client
from search_service.storage.qdrant import get_qdrant_client
from search_service.core.config import settings

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
   
   # Vérifier Elasticsearch via SearchBox
   try:
       es_client = await get_es_client()
       es_info = await es_client.info()
       health_status["components"]["elasticsearch"] = {
           "status": "ok",
           "version": es_info["version"]["number"],
           "provider": "SearchBox"
       }
   except Exception as e:
       health_status["components"]["elasticsearch"] = {
           "status": "error",
           "message": str(e)
       }
       health_status["status"] = "degraded"
   
   # Vérifier Qdrant
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
       cache_stats = await get_cache_stats()
       
       return {
           "cache": cache_stats,
           "dependencies": {
               "elasticsearch": settings.SEARCHBOX_URL != "",
               "qdrant": settings.QDRANT_URL != "",
               "deepseek": settings.DEEPSEEK_API_KEY != ""
           },
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