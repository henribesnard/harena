"""
Routes API pour le service de recherche.
VERSION CORRIGÉE - Corrige le bug 'dict' object has no attribute 'lower'
"""
import logging
import time
import asyncio
from typing import List, Dict, Any, Optional

from fastapi import APIRouter, HTTPException, Depends, Query, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator

from search_service.core.search_engine import SearchEngine
from search_service.models.requests import SearchRequest, ReindexRequest
from search_service.models.responses import SearchResponse, ReindexResponse
from config_service.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

# Variables globales pour les clients
elastic_client = None
qdrant_client = None
search_engine = None


class HealthResponse(BaseModel):
    """Modèle de réponse pour le health check."""
    status: str
    elasticsearch: Dict[str, Any]
    qdrant: Dict[str, Any]
    search_engine: Dict[str, Any]
    timestamp: float


async def get_search_engine():
    """Dependency pour obtenir le moteur de recherche."""
    global search_engine
    if not search_engine:
        raise HTTPException(
            status_code=503, 
            detail="Service de recherche non disponible"
        )
    return search_engine


def check_clients_availability():
    """Vérifie la disponibilité des clients."""
    global elastic_client, qdrant_client
    
    logger.info("🔍 Vérification des clients:")
    logger.info(f"   - elastic_client: {type(elastic_client).__name__ if elastic_client else 'None'}")
    logger.info(f"   - qdrant_client: {type(qdrant_client).__name__ if qdrant_client else 'None'}")
    logger.info(f"   - elastic_available: {elastic_client is not None}")
    logger.info(f"   - qdrant_available: {qdrant_client is not None}")
    
    # Vérifier l'état d'initialisation
    if elastic_client:
        elastic_initialized = getattr(elastic_client, '_initialized', False)
        logger.info(f"   - elastic_initialized: {elastic_initialized}")
    
    if qdrant_client:
        qdrant_initialized = getattr(qdrant_client, '_initialized', False)
        logger.info(f"   - qdrant_initialized: {qdrant_initialized}")
    
    # Déterminer les services disponibles
    available_services = []
    if elastic_client and getattr(elastic_client, '_initialized', False):
        available_services.append("Elasticsearch")
    if qdrant_client and getattr(qdrant_client, '_initialized', False):
        available_services.append("Qdrant")
    
    if available_services:
        logger.info(f"✅ Services disponibles: {', '.join(available_services)}")
        return True
    else:
        logger.error("❌ Aucun service de recherche disponible")
        return False


def create_search_engine():
    """Crée le moteur de recherche avec les clients disponibles."""
    global search_engine, elastic_client, qdrant_client
    
    try:
        # Vérifier que au moins un client est disponible
        if not check_clients_availability():
            logger.error("❌ Impossible de créer SearchEngine sans clients")
            return False
        
        # Créer le moteur de recherche
        search_engine = SearchEngine(
            elastic_client=elastic_client,
            qdrant_client=qdrant_client
        )
        
        logger.info("✅ SearchEngine créé avec succès")
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur création SearchEngine: {e}")
        return False


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Vérifie l'état de santé du service de recherche."""
    global elastic_client, qdrant_client, search_engine
    
    timestamp = time.time()
    
    # Vérifier Elasticsearch
    elasticsearch_status = {
        "available": False,
        "initialized": False,
        "healthy": False,
        "client_type": None,
        "error": None
    }
    
    if elastic_client:
        elasticsearch_status["available"] = True
        elasticsearch_status["initialized"] = getattr(elastic_client, '_initialized', False)
        elasticsearch_status["client_type"] = getattr(elastic_client, 'client_type', None)
        
        if elasticsearch_status["initialized"]:
            try:
                elasticsearch_status["healthy"] = await elastic_client.is_healthy()
            except Exception as e:
                elasticsearch_status["error"] = str(e)
    
    # Vérifier Qdrant
    qdrant_status = {
        "available": False,
        "initialized": False,
        "healthy": False,
        "error": None
    }
    
    if qdrant_client:
        qdrant_status["available"] = True
        qdrant_status["initialized"] = getattr(qdrant_client, '_initialized', False)
        
        if qdrant_status["initialized"]:
            try:
                qdrant_status["healthy"] = await qdrant_client.is_healthy()
            except Exception as e:
                qdrant_status["error"] = str(e)
    
    # Vérifier le moteur de recherche
    search_engine_status = {
        "available": search_engine is not None,
        "elasticsearch_enabled": False,
        "qdrant_enabled": False
    }
    
    if search_engine:
        search_engine_status["elasticsearch_enabled"] = search_engine.elasticsearch_enabled
        search_engine_status["qdrant_enabled"] = search_engine.qdrant_enabled
    
    # Déterminer le statut global
    overall_status = "healthy"
    if not (elasticsearch_status["healthy"] or qdrant_status["healthy"]):
        overall_status = "unhealthy"
    elif not search_engine_status["available"]:
        overall_status = "degraded"
    
    return HealthResponse(
        status=overall_status,
        elasticsearch=elasticsearch_status,
        qdrant=qdrant_status,
        search_engine=search_engine_status,
        timestamp=timestamp
    )


@router.post("/search", response_model=SearchResponse)
async def search_transactions(
    request: SearchRequest, 
    search_engine: SearchEngine = Depends(get_search_engine)
):
    """Recherche de transactions avec validation renforcée."""
    
    # VALIDATION CRITIQUE des paramètres d'entrée
    if not isinstance(request.query, str):
        logger.error(f"❌ Query doit être string, reçu: {type(request.query)} = {request.query}")
        raise HTTPException(
            status_code=400, 
            detail=f"Query must be a string, got {type(request.query).__name__}"
        )
    
    if not isinstance(request.user_id, int):
        logger.error(f"❌ user_id doit être int, reçu: {type(request.user_id)} = {request.user_id}")
        raise HTTPException(
            status_code=400, 
            detail=f"user_id must be an integer, got {type(request.user_id).__name__}"
        )
    
    user_id = request.user_id
    query = request.query.strip()
    search_type = request.type or "hybrid"
    limit = min(request.limit or 10, 50)
    use_reranking = request.use_reranking if request.use_reranking is not None else True
    
    logger.info(f"🔍 Nouvelle recherche pour user {user_id}")
    logger.info(f"   Query: '{query}' (type: {type(query)})")
    logger.info(f"   Type: {search_type}")
    logger.info(f"   Limit: {limit}")
    logger.info(f"   Use reranking: {use_reranking}")
    
    # Validation supplémentaire
    if not query:
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty"
        )
    
    if user_id <= 0:
        raise HTTPException(
            status_code=400,
            detail="user_id must be a positive integer"
        )
    
    if search_type not in ["lexical", "semantic", "hybrid"]:
        raise HTTPException(
            status_code=400,
            detail="search_type must be one of: lexical, semantic, hybrid"
        )
    
    start_time = time.time()
    
    try:
        # Effectuer la recherche selon le type demandé
        if search_type == "lexical":
            results = await search_engine.lexical_search(
                user_id=user_id,
                query=query,
                limit=limit
            )
        elif search_type == "semantic":
            results = await search_engine.semantic_search(
                user_id=user_id,
                query=query,
                limit=limit
            )
        elif search_type == "hybrid":
            results = await search_engine.hybrid_search(
                user_id=user_id,
                query=query,
                limit=limit,
                use_reranking=use_reranking
            )
        else:
            # Ne devrait jamais arriver grâce à la validation ci-dessus
            raise HTTPException(
                status_code=400,
                detail=f"Type de recherche non supporté: {search_type}"
            )
        
        query_time = time.time() - start_time
        
        # Formater les résultats
        formatted_results = []
        for result in results:
            formatted_result = {
                "id": result.get("id"),
                "score": result.get("score", 0.0),
                "transaction": result.get("source", {}),
                "highlights": result.get("highlights", {}),
                "search_type": result.get("search_type", search_type)
            }
            formatted_results.append(formatted_result)
        
        logger.info(f"✅ Recherche terminée en {query_time:.3f}s - {len(formatted_results)} résultats")
        
        return SearchResponse(
            results=formatted_results,
            total=len(formatted_results),
            query_time=query_time,
            search_type=search_type,
            user_id=user_id,
            query=query
        )
        
    except Exception as e:
        query_time = time.time() - start_time
        logger.error(f"❌ Erreur recherche après {query_time:.3f}s: {e}")
        logger.error(f"   Query: '{query}' (type: {type(query)})")
        logger.error(f"   User ID: {user_id} (type: {type(user_id)})")
        logger.error(f"   Search type: {search_type}")
        
        # Retourner une réponse vide plutôt qu'une erreur 500
        return SearchResponse(
            results=[],
            total=0,
            query_time=query_time,
            search_type=search_type,
            user_id=user_id,
            query=query,
            error=str(e)
        )


@router.post("/reindex", response_model=ReindexResponse)
async def reindex_transactions(
    request: ReindexRequest,
    search_engine: SearchEngine = Depends(get_search_engine)
):
    """Réindexe les transactions d'un utilisateur."""
    
    # Validation des paramètres
    if not isinstance(request.user_id, int):
        logger.error(f"❌ user_id doit être int, reçu: {type(request.user_id)} = {request.user_id}")
        raise HTTPException(
            status_code=400,
            detail=f"user_id must be an integer, got {type(request.user_id).__name__}"
        )
    
    user_id = request.user_id
    force_refresh = request.force_refresh or False
    
    if user_id <= 0:
        raise HTTPException(
            status_code=400,
            detail="user_id must be a positive integer"
        )
    
    logger.info(f"🔄 Réindexation pour user {user_id} (force_refresh: {force_refresh})")
    
    start_time = time.time()
    
    try:
        # Effectuer la réindexation
        result = await search_engine.reindex_user_transactions(
            user_id=user_id,
            force_refresh=force_refresh
        )
        
        reindex_time = time.time() - start_time
        
        logger.info(f"✅ Réindexation terminée en {reindex_time:.3f}s")
        logger.info(f"   Documents traités: {result.get('processed', 0)}")
        logger.info(f"   Documents indexés: {result.get('indexed', 0)}")
        logger.info(f"   Erreurs: {result.get('errors', 0)}")
        
        return ReindexResponse(
            success=True,
            processed=result.get('processed', 0),
            indexed=result.get('indexed', 0),
            errors=result.get('errors', 0),
            reindex_time=reindex_time,
            user_id=user_id
        )
        
    except Exception as e:
        reindex_time = time.time() - start_time
        logger.error(f"❌ Erreur réindexation après {reindex_time:.3f}s: {e}")
        
        return ReindexResponse(
            success=False,
            processed=0,
            indexed=0,
            errors=1,
            reindex_time=reindex_time,
            user_id=user_id,
            error=str(e)
        )


@router.get("/stats/{user_id}")
async def get_user_stats(
    user_id: int,
    search_engine: SearchEngine = Depends(get_search_engine)
):
    """Récupère les statistiques de recherche pour un utilisateur."""
    
    if user_id <= 0:
        raise HTTPException(
            status_code=400,
            detail="user_id must be a positive integer"
        )
    
    logger.info(f"📊 Récupération des stats pour user {user_id}")
    
    try:
        stats = await search_engine.get_user_stats(user_id)
        
        return {
            "user_id": user_id,
            "elasticsearch": {
                "total_documents": stats.get("elasticsearch_count", 0),
                "available": stats.get("elasticsearch_available", False)
            },
            "qdrant": {
                "total_vectors": stats.get("qdrant_count", 0),
                "available": stats.get("qdrant_available", False)
            },
            "last_update": stats.get("last_update"),
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur récupération stats user {user_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur récupération des statistiques: {str(e)}"
        )


@router.delete("/index/{user_id}")
async def delete_user_index(
    user_id: int,
    search_engine: SearchEngine = Depends(get_search_engine)
):
    """Supprime toutes les données indexées d'un utilisateur."""
    
    if user_id <= 0:
        raise HTTPException(
            status_code=400,
            detail="user_id must be a positive integer"
        )
    
    logger.info(f"🗑️ Suppression index pour user {user_id}")
    
    start_time = time.time()
    
    try:
        result = await search_engine.delete_user_data(user_id)
        
        delete_time = time.time() - start_time
        
        logger.info(f"✅ Suppression terminée en {delete_time:.3f}s")
        logger.info(f"   Elasticsearch: {result.get('elasticsearch_deleted', 0)} documents")
        logger.info(f"   Qdrant: {result.get('qdrant_deleted', 0)} vecteurs")
        
        return {
            "success": True,
            "user_id": user_id,
            "elasticsearch_deleted": result.get("elasticsearch_deleted", 0),
            "qdrant_deleted": result.get("qdrant_deleted", 0),
            "delete_time": delete_time
        }
        
    except Exception as e:
        delete_time = time.time() - start_time
        logger.error(f"❌ Erreur suppression user {user_id} après {delete_time:.3f}s: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur suppression des données: {str(e)}"
        )


@router.get("/debug/clients")
async def debug_clients():
    """Endpoint de debug pour vérifier l'état des clients."""
    global elastic_client, qdrant_client, search_engine
    
    return {
        "elastic_client": {
            "available": elastic_client is not None,
            "type": type(elastic_client).__name__ if elastic_client else None,
            "initialized": getattr(elastic_client, '_initialized', False),
            "client_type": getattr(elastic_client, 'client_type', None)
        },
        "qdrant_client": {
            "available": qdrant_client is not None,
            "type": type(qdrant_client).__name__ if qdrant_client else None,
            "initialized": getattr(qdrant_client, '_initialized', False)
        },
        "search_engine": {
            "available": search_engine is not None,
            "type": type(search_engine).__name__ if search_engine else None,
            "elasticsearch_enabled": getattr(search_engine, 'elasticsearch_enabled', False) if search_engine else False,
            "qdrant_enabled": getattr(search_engine, 'qdrant_enabled', False) if search_engine else False
        },
        "timestamp": time.time()
    }


@router.get("/debug/query-expansion")
async def debug_query_expansion(query: str = Query(..., description="Query à tester")):
    """Endpoint de debug pour tester l'expansion de requêtes."""
    
    # Validation du paramètre
    if not isinstance(query, str):
        raise HTTPException(
            status_code=400,
            detail=f"Query must be a string, got {type(query).__name__}"
        )
    
    try:
        from search_service.utils.query_expansion import expand_query_terms
        
        logger.info(f"🔍 Test expansion pour: '{query}' (type: {type(query)})")
        
        # Tester l'expansion
        expanded_terms = expand_query_terms(query)
        
        return {
            "original_query": query,
            "query_type": type(query).__name__,
            "expanded_terms": expanded_terms,
            "expanded_count": len(expanded_terms),
            "search_string": " ".join(expanded_terms),
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur test expansion: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur test expansion: {str(e)}"
        )


@router.post("/debug/search-raw")
async def debug_search_raw(
    user_id: int,
    query: str,
    client_type: str = Query("elasticsearch", description="Type de client: elasticsearch ou qdrant"),
    search_engine: SearchEngine = Depends(get_search_engine)
):
    """Endpoint de debug pour tester la recherche directe sur un client."""
    
    # Validation des paramètres
    if not isinstance(query, str):
        raise HTTPException(
            status_code=400,
            detail=f"Query must be a string, got {type(query).__name__}"
        )
    
    if not isinstance(user_id, int) or user_id <= 0:
        raise HTTPException(
            status_code=400,
            detail="user_id must be a positive integer"
        )
    
    if client_type not in ["elasticsearch", "qdrant"]:
        raise HTTPException(
            status_code=400,
            detail="client_type must be 'elasticsearch' or 'qdrant'"
        )
    
    logger.info(f"🔍 Debug recherche {client_type} pour user {user_id}: '{query}'")
    
    start_time = time.time()
    
    try:
        if client_type == "elasticsearch":
            if not search_engine.elasticsearch_enabled:
                raise HTTPException(
                    status_code=503,
                    detail="Elasticsearch non disponible"
                )
            
            results = await search_engine.lexical_search(
                user_id=user_id,
                query=query,
                limit=10
            )
            
        elif client_type == "qdrant":
            if not search_engine.qdrant_enabled:
                raise HTTPException(
                    status_code=503,
                    detail="Qdrant non disponible"
                )
            
            results = await search_engine.semantic_search(
                user_id=user_id,
                query=query,
                limit=10
            )
        
        query_time = time.time() - start_time
        
        return {
            "client_type": client_type,
            "user_id": user_id,
            "query": query,
            "query_type": type(query).__name__,
            "results_count": len(results),
            "results": results,
            "query_time": query_time,
            "timestamp": time.time()
        }
        
    except Exception as e:
        query_time = time.time() - start_time
        logger.error(f"❌ Erreur debug recherche {client_type}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur recherche {client_type}: {str(e)}"
        )


# Handlers d'erreur globaux
@router.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Gestionnaire d'erreur pour les ValueError."""
    logger.error(f"❌ ValueError: {exc}")
    return JSONResponse(
        status_code=400,
        content={
            "error": "Invalid value",
            "detail": str(exc),
            "timestamp": time.time()
        }
    )


@router.exception_handler(TypeError)
async def type_error_handler(request, exc):
    """Gestionnaire d'erreur pour les TypeError."""
    logger.error(f"❌ TypeError: {exc}")
    return JSONResponse(
        status_code=400,
        content={
            "error": "Invalid type",
            "detail": str(exc),
            "timestamp": time.time()
        }
    )


@router.exception_handler(AttributeError)
async def attribute_error_handler(request, exc):
    """Gestionnaire d'erreur pour les AttributeError (notamment le bug dict.lower())."""
    logger.error(f"❌ AttributeError: {exc}")
    
    # Détecter spécifiquement le bug dict.lower()
    if "'dict' object has no attribute 'lower'" in str(exc):
        logger.error("🚨 Bug 'dict' object has no attribute 'lower' détecté!")
        return JSONResponse(
            status_code=400,
            content={
                "error": "Query type validation error",
                "detail": "Query parameter must be a string, received a dict object",
                "bug_detected": "dict.lower() bug",
                "solution": "Ensure query parameter is properly validated as string",
                "timestamp": time.time()
            }
        )
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Attribute error",
            "detail": str(exc),
            "timestamp": time.time()
        }
    )


# Fonction d'initialisation pour les clients (appelée depuis main.py)
def set_clients(elastic=None, qdrant=None):
    """Configure les clients globaux depuis main.py."""
    global elastic_client, qdrant_client, search_engine
    
    logger.info("🔧 Configuration des clients dans routes.py")
    
    elastic_client = elastic
    qdrant_client = qdrant
    
    # Créer le moteur de recherche si possible
    if elastic_client or qdrant_client:
        success = create_search_engine()
        if success:
            logger.info("✅ Moteur de recherche configuré avec succès")
        else:
            logger.error("❌ Échec configuration moteur de recherche")
    else:
        logger.warning("⚠️ Aucun client disponible pour créer le moteur de recherche")


# Fonctions utilitaires pour les tests
async def validate_request_data(data: dict) -> dict:
    """Valide les données de requête entrantes."""
    validated_data = {}
    
    # Valider user_id
    user_id = data.get('user_id')
    if not isinstance(user_id, int):
        try:
            validated_data['user_id'] = int(user_id)
        except (ValueError, TypeError):
            raise ValueError(f"user_id must be an integer, got {type(user_id).__name__}")
    else:
        validated_data['user_id'] = user_id
    
    if validated_data['user_id'] <= 0:
        raise ValueError("user_id must be positive")
    
    # Valider query
    query = data.get('query')
    if not isinstance(query, str):
        if query is None:
            raise ValueError("query is required")
        else:
            logger.warning(f"Converting query from {type(query).__name__} to string")
            validated_data['query'] = str(query)
    else:
        validated_data['query'] = query.strip()
    
    if not validated_data['query']:
        raise ValueError("query cannot be empty")
    
    # Valider type de recherche
    search_type = data.get('type', 'hybrid')
    if not isinstance(search_type, str):
        search_type = str(search_type)
    
    if search_type not in ['lexical', 'semantic', 'hybrid']:
        raise ValueError(f"Invalid search type: {search_type}")
    
    validated_data['type'] = search_type
    
    # Valider limit
    limit = data.get('limit', 10)
    if not isinstance(limit, int):
        try:
            validated_data['limit'] = int(limit)
        except (ValueError, TypeError):
            validated_data['limit'] = 10
    else:
        validated_data['limit'] = limit
    
    validated_data['limit'] = max(1, min(validated_data['limit'], 50))
    
    # Valider use_reranking
    use_reranking = data.get('use_reranking', True)
    if not isinstance(use_reranking, bool):
        validated_data['use_reranking'] = bool(use_reranking)
    else:
        validated_data['use_reranking'] = use_reranking
    
    return validated_data


def log_request_info(endpoint: str, data: dict):
    """Log les informations de requête pour debug."""
    logger.info(f"📥 {endpoint}:")
    for key, value in data.items():
        logger.info(f"   {key}: {value} (type: {type(value).__name__})")


# Middleware personnalisé pour validation des types
async def validate_json_types(request, call_next):
    """Middleware pour valider les types JSON entrants."""
    try:
        response = await call_next(request)
        return response
    except AttributeError as e:
        if "'dict' object has no attribute 'lower'" in str(e):
            logger.error("🚨 Bug dict.lower() intercepté par middleware!")
            return JSONResponse(
                status_code=400,
                content={
                    "error": "Request validation error",
                    "detail": "Query parameter type validation failed",
                    "bug_intercepted": "dict.lower() bug prevented",
                    "timestamp": time.time()
                }
            )
        raise e


# Export des fonctions et variables importantes
__all__ = [
    'router',
    'set_clients',
    'health_check',
    'search_transactions',
    'reindex_transactions',
    'get_user_stats',
    'delete_user_index',
    'debug_clients',
    'debug_query_expansion',
    'debug_search_raw',
    'validate_request_data',
    'log_request_info'
]