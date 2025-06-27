"""
Routes API pour le service de recherche.
VERSION FINALE CORRIGÉE - Résolution complète des erreurs 500 et validation Pydantic
"""
import logging
import time
import asyncio
from typing import List, Dict, Any, Optional

from fastapi import APIRouter, HTTPException, Depends, Query, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from search_service.core.search_engine import SearchEngine
from search_service.models.requests import SearchRequest, ReindexRequest
from search_service.models.responses import SearchResponse, ReindexResponse
from search_service.models import SearchQuery, SearchType  # Import correct
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
        search_engine_status["elasticsearch_enabled"] = getattr(search_engine, 'elasticsearch_enabled', False)
        search_engine_status["qdrant_enabled"] = getattr(search_engine, 'qdrant_enabled', False)
    
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
    """
    Recherche de transactions avec validation renforcée et gestion complète des erreurs.
    VERSION FINALE - Tous les champs requis pour SearchResponse sont fournis.
    """
    
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
    search_type = getattr(request, 'search_type', SearchType.HYBRID)
    limit = min(getattr(request, 'limit', 10), 50)
    offset = getattr(request, 'offset', 0)
    use_reranking = getattr(request, 'use_reranking', True)
    
    logger.info(f"🔍 Nouvelle recherche pour user {user_id}")
    logger.info(f"   Query: '{query}' (type: {type(query)})")
    logger.info(f"   Type: {search_type}")
    logger.info(f"   Limit: {limit}")
    logger.info(f"   Offset: {offset}")
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
    
    # Convertir search_type en string pour la validation
    search_type_str = search_type.value if hasattr(search_type, 'value') else str(search_type)
    if search_type_str not in ["lexical", "semantic", "hybrid"]:
        raise HTTPException(
            status_code=400,
            detail="search_type must be one of: lexical, semantic, hybrid"
        )
    
    start_time = time.time()
    
    try:
        # Initialisation des variables pour le résultat
        formatted_results = []
        total_found = 0
        processing_time = 0.0
        search_error = None
        
        # Utiliser les nouvelles méthodes du moteur de recherche
        if hasattr(search_engine, 'search'):
            try:
                # Créer un objet SearchQuery compatible
                search_query = SearchQuery(
                    user_id=user_id,
                    query=query,
                    search_type=search_type,
                    limit=limit,
                    offset=offset,
                    use_reranking=use_reranking
                )
                
                search_result = await search_engine.search(search_query)
                
                # Convertir les résultats depuis SearchResponse du moteur
                if hasattr(search_result, 'results'):
                    for result in search_result.results:
                        formatted_result = {
                            "id": str(getattr(result, 'transaction_id', getattr(result, 'id', ''))),
                            "score": float(getattr(result, 'score', 0.0)),
                            "transaction": getattr(result, 'transaction', {}),
                            "highlights": getattr(result, 'highlights', {}),
                            "search_type": search_type_str
                        }
                        formatted_results.append(formatted_result)
                
                total_found = getattr(search_result, 'total_found', len(formatted_results))
                processing_time = getattr(search_result, 'processing_time', time.time() - start_time)
                
                logger.info(f"✅ SearchEngine.search réussi: {len(formatted_results)} résultats")
                
            except Exception as search_engine_error:
                logger.error(f"❌ Erreur SearchEngine.search: {search_engine_error}")
                # Fallback vers les anciennes méthodes
                search_error = f"SearchEngine error: {search_engine_error}"
        
        # Fallback vers les anciennes méthodes si la nouvelle méthode échoue
        if not formatted_results and hasattr(search_engine, 'lexical_search'):
            try:
                logger.info("🔄 Fallback vers anciennes méthodes")
                
                results = []
                if search_type_str == "lexical":
                    results = await search_engine.lexical_search(
                        user_id=user_id,
                        query=query,
                        limit=limit
                    )
                elif search_type_str == "semantic":
                    results = await search_engine.semantic_search(
                        user_id=user_id,
                        query=query,
                        limit=limit
                    )
                elif search_type_str == "hybrid":
                    results = await search_engine.hybrid_search(
                        user_id=user_id,
                        query=query,
                        limit=limit,
                        use_reranking=use_reranking
                    )
                
                processing_time = time.time() - start_time
                
                # Formater les résultats de l'ancien système
                for result in results:
                    formatted_result = {
                        "id": result.get("id", ""),
                        "score": float(result.get("score", 0.0)),
                        "transaction": result.get("source", {}),
                        "highlights": result.get("highlights", {}),
                        "search_type": search_type_str
                    }
                    formatted_results.append(formatted_result)
                
                total_found = len(formatted_results)
                
                logger.info(f"✅ Fallback réussi: {len(formatted_results)} résultats")
                
            except Exception as fallback_error:
                logger.error(f"❌ Erreur fallback: {fallback_error}")
                search_error = f"All methods failed: {fallback_error}"
        
        processing_time = time.time() - start_time
        
        logger.info(f"✅ Recherche terminée en {processing_time:.3f}s - {len(formatted_results)} résultats")
        
        # CONSTRUCTION FINALE CORRECTE de SearchResponse avec TOUS les champs requis
        return SearchResponse(
            # Champs OBLIGATOIRES pour éviter l'erreur Pydantic
            query=query,
            search_type=search_type_str,
            results=formatted_results,
            total_found=total_found,
            limit=limit,
            offset=offset,
            has_more=(offset + len(formatted_results)) < total_found,
            processing_time=processing_time,
            
            # Champs pour compatibilité avec le nouveau système
            total=total_found,
            query_time=processing_time,
            user_id=user_id,
            
            # Champs optionnels
            error=search_error,
            timestamp=time.time()
        )
        
    except HTTPException:
        # Re-raise HTTPException (codes 400, etc.)
        raise
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"❌ Erreur recherche après {processing_time:.3f}s: {e}")
        logger.error(f"   Query: '{query}' (type: {type(query)})")
        logger.error(f"   User ID: {user_id} (type: {type(user_id)})")
        logger.error(f"   Search type: {search_type}")
        
        # RETOURNER une réponse vide avec TOUS les champs requis
        return SearchResponse(
            # Champs OBLIGATOIRES pour éviter l'erreur Pydantic
            query=query,
            search_type=search_type_str if 'search_type_str' in locals() else "hybrid",
            results=[],
            total_found=0,
            limit=limit,
            offset=offset,
            has_more=False,
            processing_time=processing_time,
            
            # Champs pour compatibilité avec le nouveau système
            total=0,
            query_time=processing_time,
            user_id=user_id,
            
            # Champs d'erreur
            error=str(e),
            timestamp=time.time()
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
    force_refresh = getattr(request, 'force_refresh', False)
    
    if user_id <= 0:
        raise HTTPException(
            status_code=400,
            detail="user_id must be a positive integer"
        )
    
    logger.info(f"🔄 Réindexation pour user {user_id} (force_refresh: {force_refresh})")
    
    start_time = time.time()
    
    try:
        # Effectuer la réindexation si la méthode existe
        if hasattr(search_engine, 'reindex_user_transactions'):
            result = await search_engine.reindex_user_transactions(
                user_id=user_id,
                force_refresh=force_refresh
            )
        else:
            # Fallback si la méthode n'existe pas
            result = {
                'processed': 0,
                'indexed': 0,
                'errors': 1,
                'message': 'Reindex method not implemented'
            }
        
        reindex_time = time.time() - start_time
        
        logger.info(f"✅ Réindexation terminée en {reindex_time:.3f}s")
        logger.info(f"   Documents traités: {result.get('processed', 0)}")
        logger.info(f"   Documents indexés: {result.get('indexed', 0)}")
        logger.info(f"   Erreurs: {result.get('errors', 0)}")
        
        return ReindexResponse(
            success=result.get('errors', 0) == 0,
            processed=result.get('processed', 0),
            indexed=result.get('indexed', 0),
            errors=result.get('errors', 0),
            reindex_time=reindex_time,
            user_id=user_id,
            timestamp=time.time()
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
            error=str(e),
            timestamp=time.time()
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
        # Essayer d'obtenir les stats si la méthode existe
        if hasattr(search_engine, 'get_user_stats'):
            stats = await search_engine.get_user_stats(user_id)
        else:
            # Fallback avec des stats par défaut
            stats = {
                "elasticsearch_count": 0,
                "qdrant_count": 0,
                "elasticsearch_available": elastic_client is not None,
                "qdrant_available": qdrant_client is not None,
                "last_update": None
            }
        
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
        # Essayer la suppression si la méthode existe
        if hasattr(search_engine, 'delete_user_data'):
            result = await search_engine.delete_user_data(user_id)
        else:
            # Fallback
            result = {
                "elasticsearch_deleted": 0,
                "qdrant_deleted": 0,
                "message": "Delete method not implemented"
            }
        
        delete_time = time.time() - start_time
        
        logger.info(f"✅ Suppression terminée en {delete_time:.3f}s")
        logger.info(f"   Elasticsearch: {result.get('elasticsearch_deleted', 0)} documents")
        logger.info(f"   Qdrant: {result.get('qdrant_deleted', 0)} vecteurs")
        
        return {
            "success": True,
            "user_id": user_id,
            "elasticsearch_deleted": result.get("elasticsearch_deleted", 0),
            "qdrant_deleted": result.get("qdrant_deleted", 0),
            "delete_time": delete_time,
            "timestamp": time.time()
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


@router.get("/debug/injection")
async def debug_injection():
    """Endpoint de debug pour vérifier l'injection des clients."""
    global elastic_client, qdrant_client, search_engine
    
    injection_status = {
        "injection_successful": False,
        "clients_available": {
            "elastic": elastic_client is not None,
            "qdrant": qdrant_client is not None
        },
        "clients_initialized": {
            "elastic": getattr(elastic_client, '_initialized', False) if elastic_client else False,
            "qdrant": getattr(qdrant_client, '_initialized', False) if qdrant_client else False
        },
        "search_engine_created": search_engine is not None,
        "search_engine_capabilities": {},
        "recommendations": [],
        "timestamp": time.time()
    }
    
    # Vérifier les capacités du moteur de recherche
    if search_engine:
        injection_status["search_engine_capabilities"] = {
            "has_elastic": hasattr(search_engine, 'elastic_client') and search_engine.elastic_client is not None,
            "has_qdrant": hasattr(search_engine, 'qdrant_client') and search_engine.qdrant_client is not None,
            "elasticsearch_enabled": getattr(search_engine, 'elasticsearch_enabled', False),
            "qdrant_enabled": getattr(search_engine, 'qdrant_enabled', False)
        }
    
    # Déterminer le succès de l'injection
    injection_status["injection_successful"] = (
        injection_status["clients_available"]["elastic"] or 
        injection_status["clients_available"]["qdrant"]
    ) and injection_status["search_engine_created"]
    
    # Générer des recommandations
    if not injection_status["clients_available"]["elastic"] and not injection_status["clients_available"]["qdrant"]:
        injection_status["recommendations"].append("Aucun client disponible - vérifiez la configuration")
    
    if injection_status["clients_available"]["elastic"] and not injection_status["clients_initialized"]["elastic"]:
        injection_status["recommendations"].append("Client Elasticsearch disponible mais non initialisé")
    
    if injection_status["clients_available"]["qdrant"] and not injection_status["clients_initialized"]["qdrant"]:
        injection_status["recommendations"].append("Client Qdrant disponible mais non initialisé")
    
    if not injection_status["search_engine_created"]:
        injection_status["recommendations"].append("Moteur de recherche non créé - vérifiez l'injection")
    
    if injection_status["injection_successful"]:
        injection_status["recommendations"].append("Injection réussie - Search Service opérationnel")
    
    return injection_status


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
        # Essayer d'importer et tester l'expansion
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
                "expansion_available": True,
                "timestamp": time.time()
            }
        except ImportError:
            # Module d'expansion non disponible
            return {
                "original_query": query,
                "query_type": type(query).__name__,
                "expanded_terms": [query],
                "expanded_count": 1,
                "search_string": query,
                "expansion_available": False,
                "error": "Query expansion module not available",
                "timestamp": time.time()
            }
        
    except Exception as e:
        logger.error(f"❌ Erreur test expansion: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur test expansion: {str(e)}"
        )


@router.get("/debug/test-search")
async def debug_test_search(
    query: str = Query("test", description="Query de test"),
    user_id: int = Query(34, description="User ID de test"),
    search_type: str = Query("hybrid", description="Type de recherche")
):
    """Endpoint de debug pour tester une recherche complète."""
    
    logger.info(f"🧪 Test de recherche: '{query}' pour user {user_id}")
    
    try:
        # Valider les paramètres
        query = validate_string_parameter("query", query)
        user_id = validate_integer_parameter("user_id", user_id)
        
        if search_type not in ["lexical", "semantic", "hybrid"]:
            raise HTTPException(
                status_code=400,
                detail="search_type must be one of: lexical, semantic, hybrid"
            )
        
        # Créer une requête de test
        test_request = SearchRequest(
            query=query,
            user_id=user_id,
            search_type=SearchType(search_type),
            limit=5,
            offset=0,
            use_reranking=False
        )
        
        # Exécuter la recherche
        start_time = time.time()
        
        # Utiliser directement la fonction search_transactions
        search_result = await search_transactions(
            request=test_request,
            search_engine=await get_search_engine()
        )
        
        execution_time = time.time() - start_time
        
        return {
            "test_parameters": {
                "query": query,
                "user_id": user_id,
                "search_type": search_type
            },
            "execution_time": execution_time,
            "search_result": {
                "query": search_result.query,
                "results_count": len(search_result.results),
                "total_found": search_result.total_found,
                "processing_time": search_result.processing_time,
                "error": search_result.error
            },
            "test_successful": search_result.error is None,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur test de recherche: {e}")
        return {
            "test_parameters": {
                "query": query,
                "user_id": user_id,
                "search_type": search_type
            },
            "test_successful": False,
            "error": str(e),
            "timestamp": time.time()
        }


# Fonction d'initialisation pour les clients (appelée depuis heroku_app.py)
def set_clients(elastic=None, qdrant=None):
    """Configure les clients globaux depuis heroku_app.py."""
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


# Fonctions utilitaires pour validation et debug
def validate_string_parameter(param_name: str, value: Any) -> str:
    """Valide qu'un paramètre est une chaîne de caractères."""
    if not isinstance(value, str):
        raise HTTPException(
            status_code=400,
            detail=f"{param_name} must be a string, got {type(value).__name__}"
        )
    return value.strip()


def validate_integer_parameter(param_name: str, value: Any, min_value: int = 1) -> int:
    """Valide qu'un paramètre est un entier valide."""
    if not isinstance(value, int):
        try:
            value = int(value)
        except (ValueError, TypeError):
            raise HTTPException(
                status_code=400,
                detail=f"{param_name} must be an integer, got {type(value).__name__}"
            )
    
    if value < min_value:
        raise HTTPException(
            status_code=400,
            detail=f"{param_name} must be >= {min_value}, got {value}"
        )
    
    return value


def log_request_info(endpoint: str, **kwargs):
    """Log les informations de requête pour debug."""
    logger.info(f"📥 {endpoint}:")
    for key, value in kwargs.items():
        logger.info(f"   {key}: {value} (type: {type(value).__name__})")


def format_search_results(results: List[Any], search_type: str) -> List[Dict[str, Any]]:
    """Formate les résultats de recherche pour l'API."""
    formatted_results = []
    
    for result in results:
        # Gérer différents types de résultats
        if hasattr(result, 'transaction_id'):
            # Nouveau format SearchResult
            formatted_result = {
                "id": str(result.transaction_id),
                "score": getattr(result, 'score', 0.0),
                "transaction": getattr(result, 'transaction', {}),
                "highlights": getattr(result, 'highlights', {}),
                "search_type": getattr(result, 'search_type', search_type)
            }
        elif isinstance(result, dict):
            # Format dictionnaire (ancien système)
            formatted_result = {
                "id": result.get("id", result.get("_id", "")),
                "score": result.get("score", result.get("_score", 0.0)),
                "transaction": result.get("source", result.get("_source", {})),
                "highlights": result.get("highlights", {}),
                "search_type": result.get("search_type", search_type)
            }
        else:
            # Format par défaut
            formatted_result = {
                "id": str(getattr(result, 'id', '')),
                "score": getattr(result, 'score', 0.0),
                "transaction": getattr(result, 'transaction', {}),
                "highlights": getattr(result, 'highlights', {}),
                "search_type": search_type
            }
        
        formatted_results.append(formatted_result)
    
    return formatted_results


def create_error_response(
    query: str,
    user_id: int,
    search_type: str,
    limit: int,
    offset: int,
    error_message: str,
    processing_time: float
) -> SearchResponse:
    """Crée une réponse d'erreur standardisée avec tous les champs requis."""
    return SearchResponse(
        # Champs OBLIGATOIRES pour éviter l'erreur Pydantic
        query=query,
        search_type=search_type,
        results=[],
        total_found=0,
        limit=limit,
        offset=offset,
        has_more=False,
        processing_time=processing_time,
        
        # Champs pour compatibilité avec le nouveau système
        total=0,
        query_time=processing_time,
        user_id=user_id,
        
        # Champs d'erreur
        error=error_message,
        timestamp=time.time()
    )


def create_success_response(
    query: str,
    user_id: int,
    search_type: str,
    limit: int,
    offset: int,
    results: List[Dict[str, Any]],
    total_found: int,
    processing_time: float
) -> SearchResponse:
    """Crée une réponse de succès standardisée avec tous les champs requis."""
    return SearchResponse(
        # Champs OBLIGATOIRES pour éviter l'erreur Pydantic
        query=query,
        search_type=search_type,
        results=results,
        total_found=total_found,
        limit=limit,
        offset=offset,
        has_more=(offset + len(results)) < total_found,
        processing_time=processing_time,
        
        # Champs pour compatibilité avec le nouveau système
        total=total_found,
        query_time=processing_time,
        user_id=user_id,
        
        # Champs optionnels
        error=None,
        timestamp=time.time()
    )


@router.get("/debug/validation")
async def debug_validation():
    """Endpoint de debug pour tester la validation des modèles."""
    
    try:
        # Test de création d'une SearchResponse avec tous les champs
        test_response = SearchResponse(
            query="test query",
            search_type="hybrid",
            results=[],
            total_found=0,
            limit=10,
            offset=0,
            has_more=False,
            processing_time=0.1,
            total=0,
            query_time=0.1,
            user_id=34,
            timestamp=time.time()
        )
        
        return {
            "validation_successful": True,
            "test_response": {
                "query": test_response.query,
                "search_type": test_response.search_type,
                "results_count": len(test_response.results),
                "total_found": test_response.total_found,
                "limit": test_response.limit,
                "offset": test_response.offset,
                "has_more": test_response.has_more,
                "processing_time": test_response.processing_time,
                "user_id": test_response.user_id
            },
            "timestamp": time.time()
        }
        
    except Exception as e:
        return {
            "validation_successful": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "timestamp": time.time()
        }


@router.get("/debug/model-fields")
async def debug_model_fields():
    """Endpoint de debug pour afficher les champs requis des modèles."""
    
    try:
        from search_service.models.responses import SearchResponse
        from search_service.models.requests import SearchRequest
        
        # Récupérer les champs des modèles
        search_response_fields = list(SearchResponse.model_fields.keys())
        search_request_fields = list(SearchRequest.model_fields.keys())
        
        # Identifier les champs requis (compatible avec Pydantic v2)
        required_response_fields = []
        required_request_fields = []
        
        try:
            # Pydantic v2 method
            for field_name, field_info in SearchResponse.model_fields.items():
                if hasattr(field_info, 'is_required') and field_info.is_required():
                    required_response_fields.append(field_name)
                elif not hasattr(field_info, 'default') or field_info.default is ...:
                    required_response_fields.append(field_name)
            
            for field_name, field_info in SearchRequest.model_fields.items():
                if hasattr(field_info, 'is_required') and field_info.is_required():
                    required_request_fields.append(field_name)
                elif not hasattr(field_info, 'default') or field_info.default is ...:
                    required_request_fields.append(field_name)
        except Exception:
            # Fallback method
            required_response_fields = search_response_fields
            required_request_fields = search_request_fields
        
        return {
            "SearchResponse": {
                "all_fields": search_response_fields,
                "required_fields": required_response_fields,
                "total_fields": len(search_response_fields),
                "required_count": len(required_response_fields)
            },
            "SearchRequest": {
                "all_fields": search_request_fields,
                "required_fields": required_request_fields,
                "total_fields": len(search_request_fields),
                "required_count": len(required_request_fields)
            },
            "timestamp": time.time()
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "error_type": type(e).__name__,
            "timestamp": time.time()
        }


@router.get("/debug/environment")
async def debug_environment():
    """Endpoint de debug pour vérifier l'environnement."""
    
    import sys
    import platform
    
    try:
        # Informations sur l'environnement
        env_info = {
            "python_version": sys.version,
            "platform": platform.platform(),
            "architecture": platform.architecture(),
            "processor": platform.processor()
        }
        
        # Vérifier les dépendances critiques
        dependencies_status = {}
        critical_packages = [
            "fastapi", "pydantic", "elasticsearch", "qdrant_client", 
            "asyncio", "logging", "time"
        ]
        
        for package in critical_packages:
            try:
                __import__(package)
                dependencies_status[package] = "available"
            except ImportError:
                dependencies_status[package] = "missing"
        
        # Informations sur les settings
        settings_info = {
            "available": 'settings' in globals() and settings is not None
        }
        
        if 'settings' in globals() and settings is not None:
            settings_info.update({
                "bonsai_configured": bool(getattr(settings, 'BONSAI_URL', None)),
                "qdrant_configured": bool(getattr(settings, 'QDRANT_URL', None))
            })
        
        return {
            "environment": env_info,
            "dependencies": dependencies_status,
            "settings": settings_info,
            "clients_status": {
                "elastic_client": elastic_client is not None,
                "qdrant_client": qdrant_client is not None,
                "search_engine": search_engine is not None
            },
            "timestamp": time.time()
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "error_type": type(e).__name__,
            "timestamp": time.time()
        }


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
    'debug_injection',
    'debug_query_expansion',
    'debug_test_search',
    'debug_validation',
    'debug_model_fields',
    'debug_environment',
    'validate_string_parameter',
    'validate_integer_parameter',
    'log_request_info',
    'format_search_results',
    'create_error_response',
    'create_success_response'
]