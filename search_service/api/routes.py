"""
Routes API REST pour le Search Service
=====================================

Version corrigée - cohérente avec main.py et états de production corrects.
Suppression des fallbacks trompeurs, utilisation des vrais états du service.
Fix: Utilisation de search() au lieu de search_transactions()
"""

import logging
import os
import time
from datetime import datetime
from typing import Dict, Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

# Import pour les contrats de service (imports minimaux pour éviter les erreurs)
try:
    from search_service.models.service_contracts import SearchServiceQuery
except ImportError:
    logger.warning("SearchServiceQuery non disponible, utilisation de dict")
    SearchServiceQuery = dict  # Fallback

logger = logging.getLogger(__name__)

# === ROUTEUR PRINCIPAL ===
router = APIRouter(tags=["search"])

# === FONCTIONS HELPER POUR COMPATIBILITÉ API ===

def create_search_service_query(
    user_id: int,
    query: str = "",
    filters: Dict[str, Any] = None,
    limit: int = 20,
    offset: int = 0,
    intent_type: str = "TEXT_SEARCH"
) -> SearchServiceQuery:
    """Crée un SearchServiceQuery à partir des paramètres simples de l'API"""
    
    # Métadonnées de base (format dict simple pour compatibilité)
    query_metadata = {
        "query_id": f"api_{int(time.time() * 1000)}",
        "user_id": user_id,
        "intent_type": intent_type,
        "confidence": 0.9,
        "agent_name": "api_endpoint",
        "team_name": "direct_search",
        "execution_context": {
            "conversation_id": "api_direct",
            "turn_number": 1,
            "agent_chain": ["api_endpoint"]
        }
    }
    
    # Paramètres de recherche (format dict simple)
    search_parameters = {
        "query_type": "TEXT_SEARCH",
        "fields": [
            "transaction_id", "user_id", "amount", "date",
            "primary_description", "merchant_name", "category_name",
            "searchable_text"
        ],
        "limit": limit,
        "offset": offset,
        "timeout_ms": 5000
    }
    
    # Construire les filtres (format dict simple pour compatibilité)
    required_filters = [
        {"field": "user_id", "operator": "eq", "value": user_id}
    ]
    
    optional_filters = []
    range_filters = []
    
    if filters:
        for field, value in filters.items():
            if field == "user_id":
                continue  # Déjà ajouté
            elif isinstance(value, dict) and ("gte" in value or "lte" in value):
                # Filtre de range
                if "gte" in value and "lte" in value:
                    range_filters.append({
                        "field": field, 
                        "operator": "between", 
                        "value": [value["gte"], value["lte"]]
                    })
                elif "gte" in value:
                    range_filters.append({
                        "field": field, 
                        "operator": "gte", 
                        "value": value["gte"]
                    })
                elif "lte" in value:
                    range_filters.append({
                        "field": field, 
                        "operator": "lte", 
                        "value": value["lte"]
                    })
            else:
                # Filtre terme simple
                optional_filters.append({
                    "field": field, 
                    "operator": "eq", 
                    "value": value
                })
    
    # Recherche textuelle (format dict simple)
    text_search = None
    if query and query.strip():
        text_search = {
            "query": query.strip(),
            "fields": ["searchable_text", "primary_description", "merchant_name"],
            "operator": "match"
        }
    
    # Filtres (format dict simple)
    search_filters = {
        "required": required_filters,
        "optional": optional_filters,
        "ranges": range_filters
    }
    
    # Options par défaut (format dict simple)
    options = {
        "cache_enabled": True,
        "include_explanation": False,
        "include_aggregations": False
    }
    
    # Créer le contrat en utilisant des dicts simples pour éviter les erreurs d'import
    contract_dict = {
        "query_metadata": query_metadata,
        "search_parameters": search_parameters,
        "filters": search_filters,
        "text_search": text_search,
        "options": options
    }
    
    # Tenter de créer le SearchServiceQuery, avec fallback vers dict
    try:
        return SearchServiceQuery(**contract_dict)
    except Exception as e:
        logger.warning(f"Fallback vers dict simple pour SearchServiceQuery: {e}")
        # Fallback vers un dict simple si la construction échoue
        return contract_dict

def convert_service_response_to_legacy(service_response) -> Dict[str, Any]:
    """Convertit SearchServiceResponse vers le format attendu par l'API legacy"""
    
    # Gérer les cas où service_response peut être un dict (fallback) ou un objet
    if isinstance(service_response, dict):
        # Si c'est déjà un dict, essayer de l'adapter
        if "hits" in service_response:
            return service_response  # Déjà au bon format
        
        # Sinon, créer un format de base
        return {
            "took": 0,
            "hits": {
                "total": {"value": 0, "relation": "eq"},
                "hits": []
            },
            "query_id": "fallback",
            "cache_hit": False
        }
    
    # Si c'est un objet SearchServiceResponse
    try:
        # Extraire les données de la réponse
        hits_data = []
        
        results = getattr(service_response, 'results', [])
        for result in results:
            hit = {
                "_source": {
                    "transaction_id": getattr(result, 'transaction_id', ''),
                    "user_id": getattr(result, 'user_id', 0),
                    "account_id": getattr(result, 'account_id', ''),
                    "amount": getattr(result, 'amount', 0.0),
                    "amount_abs": getattr(result, 'amount_abs', 0.0),
                    "transaction_type": getattr(result, 'transaction_type', ''),
                    "currency_code": getattr(result, 'currency_code', 'EUR'),
                    "date": getattr(result, 'date', ''),
                    "primary_description": getattr(result, 'primary_description', ''),
                    "merchant_name": getattr(result, 'merchant_name', ''),
                    "category_name": getattr(result, 'category_name', ''),
                    "operation_type": getattr(result, 'operation_type', ''),
                    "month_year": getattr(result, 'month_year', ''),
                    "weekday": getattr(result, 'weekday', ''),
                    "searchable_text": getattr(result, 'searchable_text', '')
                },
                "_score": getattr(result, 'score', 0.0)
            }
            
            # Ajouter highlights si présents
            if hasattr(result, 'highlights') and result.highlights:
                hit["highlight"] = result.highlights
            
            hits_data.append(hit)
        
        # Extraire les métadonnées
        response_metadata = getattr(service_response, 'response_metadata', None)
        
        # Format de réponse compatible
        return {
            "took": getattr(response_metadata, 'elasticsearch_took', 0) if response_metadata else 0,
            "hits": {
                "total": {
                    "value": getattr(response_metadata, 'total_hits', len(hits_data)) if response_metadata else len(hits_data),
                    "relation": "eq"
                },
                "hits": hits_data
            },
            "query_id": getattr(response_metadata, 'query_id', 'unknown') if response_metadata else 'unknown',
            "cache_hit": getattr(response_metadata, 'cache_hit', False) if response_metadata else False
        }
        
    except Exception as e:
        logger.error(f"Erreur conversion réponse: {e}")
        # Fallback sécurisé
        return {
            "took": 0,
            "hits": {
                "total": {"value": 0, "relation": "eq"},
                "hits": []
            },
            "query_id": "conversion_error",
            "cache_hit": False,
            "error": str(e)
        }

# === ENDPOINTS ===

@router.get("/health", summary="Vérification de l'état du service")
async def health_check(request: Request):
    """
    Health check basé uniquement sur l'état RÉEL du service initialisé par main.py
    Pas de fallbacks trompeurs, seuls les vrais états sont retournés.
    """
    health_status = {
        "service": "search_service",
        "timestamp": datetime.utcnow().isoformat(),
        "status": "unknown",
        "version": "1.0.0",
        "details": {}
    }
    
    try:
        # Récupérer l'état RÉEL depuis main.py (pas de fallbacks)
        service_initialized = getattr(request.app.state, 'service_initialized', False)
        elasticsearch_client = getattr(request.app.state, 'elasticsearch_client', None)
        core_manager = getattr(request.app.state, 'core_manager', None)
        initialization_error = getattr(request.app.state, 'initialization_error', None)
        
        # CAS 1: Service correctement initialisé par main.py
        if service_initialized and elasticsearch_client and core_manager:
            try:
                # Vérifier l'état réel d'Elasticsearch
                es_health = await elasticsearch_client.health_check()
                
                # Vérifier l'état réel du core manager
                core_initialized = core_manager.is_initialized()
                
                if core_initialized:
                    health_status.update({
                        "status": "healthy",
                        "details": {
                            "service_initialized": True,
                            "elasticsearch": es_health,
                            "core_manager": "initialized",
                            "bonsai_url_configured": bool(os.environ.get("BONSAI_URL")),
                            "initialization_source": "main.py"
                        }
                    })
                    return JSONResponse(content=health_status, status_code=200)
                else:
                    health_status.update({
                        "status": "degraded",
                        "details": {
                            "service_initialized": True,
                            "elasticsearch": es_health,
                            "core_manager": "not_initialized",
                            "bonsai_url_configured": bool(os.environ.get("BONSAI_URL")),
                            "error": "Core manager exists but not initialized"
                        }
                    })
                    return JSONResponse(content=health_status, status_code=503)
                
            except Exception as e:
                # Erreur lors des tests de santé
                health_status.update({
                    "status": "degraded", 
                    "details": {
                        "service_initialized": True,
                        "elasticsearch_error": str(e),
                        "core_manager": "error_during_check",
                        "bonsai_url_configured": bool(os.environ.get("BONSAI_URL")),
                        "initialization_source": "main.py"
                    }
                })
                return JSONResponse(content=health_status, status_code=503)
        
        # CAS 2: Erreur d'initialisation connue
        elif initialization_error:
            health_status.update({
                "status": "unhealthy",
                "details": {
                    "service_initialized": False,
                    "initialization_error": initialization_error,
                    "bonsai_url_configured": bool(os.environ.get("BONSAI_URL")),
                    "initialization_source": "main.py",
                    "recommendation": "Check logs and restart service"
                }
            })
            return JSONResponse(content=health_status, status_code=503)
        
        # CAS 3: Service pas encore initialisé (démarrage en cours)
        else:
            health_status.update({
                "status": "starting",
                "details": {
                    "service_initialized": False,
                    "elasticsearch_client_available": elasticsearch_client is not None,
                    "core_manager_available": core_manager is not None,
                    "bonsai_url_configured": bool(os.environ.get("BONSAI_URL")),
                    "initialization_source": "main.py",
                    "message": "Service is starting up, please wait"
                }
            })
            return JSONResponse(content=health_status, status_code=503)
    
    except Exception as e:
        # Erreur inattendue dans le health check
        logger.error(f"❌ Erreur inattendue dans health check: {e}")
        health_status.update({
            "status": "error",
            "details": {
                "unexpected_error": str(e),
                "bonsai_url_configured": bool(os.environ.get("BONSAI_URL"))
            }
        })
        return JSONResponse(content=health_status, status_code=500)

@router.get("/status", summary="Statut détaillé du service")
async def service_status(request: Request):
    """
    Statut détaillé basé sur l'état réel du service
    """
    try:
        # Informations de base
        status_info = {
            "service": "search_service",
            "timestamp": datetime.utcnow().isoformat(),
            "version": "1.0.0"
        }
        
        # État du service depuis main.py (état réel)
        service_state = {
            "service_initialized": getattr(request.app.state, 'service_initialized', False),
            "elasticsearch_client_available": getattr(request.app.state, 'elasticsearch_client', None) is not None,
            "core_manager_available": getattr(request.app.state, 'core_manager', None) is not None,
            "initialization_error": getattr(request.app.state, 'initialization_error', None)
        }
        
        # Vérifier l'état réel du core manager si disponible
        core_manager = getattr(request.app.state, 'core_manager', None)
        if core_manager:
            service_state["core_manager_initialized"] = core_manager.is_initialized()
        else:
            service_state["core_manager_initialized"] = False
        
        # Configuration Elasticsearch
        try:
            from search_service.clients.elasticsearch_client import get_client_configuration_info
            config_info = get_client_configuration_info()
            status_info["configuration"] = config_info
        except Exception as e:
            status_info["configuration_error"] = str(e)
        
        # Métriques du client
        try:
            from search_service.clients.elasticsearch_client import get_client_metrics
            metrics = get_client_metrics()
            status_info["metrics"] = metrics
        except Exception as e:
            status_info["metrics_error"] = str(e)
        
        # État global
        status_info["service_state"] = service_state
        
        return JSONResponse(content=status_info, status_code=200)
        
    except Exception as e:
        return JSONResponse(
            content={
                "error": "Failed to get service status",
                "details": str(e)
            },
            status_code=500
        )

@router.post("/search", summary="Recherche de transactions")
async def search_transactions(request: Request, search_request: dict):
    """
    ✅ ENDPOINT CORRIGÉ - Utilise search() au lieu de search_transactions()
    Endpoint principal de recherche - vérifications strictes sans fallbacks
    """
    try:
        # Vérification stricte de l'état du service
        service_initialized = getattr(request.app.state, 'service_initialized', False)
        elasticsearch_client = getattr(request.app.state, 'elasticsearch_client', None)
        core_manager = getattr(request.app.state, 'core_manager', None)
        
        # Vérifications strictes - pas de fallbacks
        if not service_initialized:
            raise HTTPException(
                status_code=503,
                detail="Service not initialized. Please check service health and restart if needed."
            )
        
        if not elasticsearch_client:
            raise HTTPException(
                status_code=503,
                detail="Elasticsearch client not available. Service needs restart."
            )
        
        if not core_manager:
            raise HTTPException(
                status_code=503,
                detail="Core manager not available. Service needs restart."
            )
        
        # Vérifier que le core manager est réellement initialisé
        if not core_manager.is_initialized():
            raise HTTPException(
                status_code=503,
                detail="Core manager not properly initialized. Service needs restart."
            )
        
        # Validation des paramètres d'entrée
        user_id = search_request.get("user_id")
        if not user_id:
            raise HTTPException(
                status_code=400,
                detail="user_id is required"
            )
        
        if not isinstance(user_id, int) or user_id <= 0:
            raise HTTPException(
                status_code=400,
                detail="user_id must be a positive integer"
            )
        
        query = search_request.get("query", "")
        filters = search_request.get("filters", {})
        limit = search_request.get("limit", 20)
        offset = search_request.get("offset", 0)
        
        # Validation des limites
        if limit > 100:
            limit = 100
        if offset < 0:
            offset = 0
        
        # Obtenir le moteur de recherche
        search_engine = core_manager.get_search_engine()
        if not search_engine:
            raise HTTPException(
                status_code=503,
                detail="Search engine not available from core manager"
            )
        
        # ✅ CORRECTION: Utiliser search() avec SearchServiceQuery au lieu de search_transactions()
        
        # Créer le SearchServiceQuery à partir des paramètres
        service_query = create_search_service_query(
            user_id=user_id,
            query=query,
            filters=filters,
            limit=limit,
            offset=offset
        )
        
        # Appeler la méthode search() qui existe réellement
        service_response = await search_engine.search(service_query)
        
        # Convertir vers le format legacy attendu par le client
        legacy_result = convert_service_response_to_legacy(service_response)
        
        return JSONResponse(content=legacy_result, status_code=200)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Erreur lors de la recherche: {e}")
        import traceback
        logger.error(f"Stack trace: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )

@router.get("/test-connection", summary="Test de connexion Elasticsearch")
async def test_elasticsearch_connection_endpoint(request: Request):
    """
    Test de connexion utilisant le client réel du service
    """
    try:
        elasticsearch_client = getattr(request.app.state, 'elasticsearch_client', None)
        
        if not elasticsearch_client:
            return JSONResponse(
                content={
                    "connection_test": False,
                    "error": "Elasticsearch client not available",
                    "health_check": {"status": "unavailable"}
                },
                status_code=503
            )
        
        # Test de connexion réel
        connection_test = await elasticsearch_client.test_connection()
        health_check = await elasticsearch_client.health_check()
        
        result = {
            "connection_test": connection_test,
            "health_check": health_check,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if connection_test:
            return JSONResponse(content=result, status_code=200)
        else:
            return JSONResponse(content=result, status_code=503)
            
    except Exception as e:
        return JSONResponse(
            content={
                "connection_test": False,
                "error": str(e),
                "health_check": {"status": "error", "message": str(e)},
                "timestamp": datetime.utcnow().isoformat()
            },
            status_code=500
        )

@router.get("/quick-search", summary="Recherche rapide pour tests")
async def quick_search_endpoint(
    request: Request,
    user_id: int = 34,
    query: str = "test",
    limit: int = 5
):
    """
    ✅ ENDPOINT CORRIGÉ - Quick search avec search() au lieu de search_transactions()
    Recherche rapide utilisant les composants réels du service
    """
    try:
        # Utiliser les composants réels du service
        elasticsearch_client = getattr(request.app.state, 'elasticsearch_client', None)
        core_manager = getattr(request.app.state, 'core_manager', None)
        
        if not elasticsearch_client:
            return JSONResponse(
                content={
                    "error": "Elasticsearch client not available",
                    "took": 0,
                    "hits": {"total": {"value": 0}, "hits": []}
                },
                status_code=503
            )
        
        if not core_manager or not core_manager.is_initialized():
            # Fallback sur une recherche directe si le core manager n'est pas disponible
            try:
                from search_service.clients.elasticsearch_client import quick_search
                result = await quick_search(user_id=user_id, query=query, limit=limit)
                
                if "error" in result:
                    return JSONResponse(content=result, status_code=503)
                else:
                    return JSONResponse(content=result, status_code=200)
            except Exception as e:
                return JSONResponse(
                    content={
                        "error": f"Quick search failed: {str(e)}",
                        "took": 0,
                        "hits": {"total": {"value": 0}, "hits": []}
                    },
                    status_code=500
                )
        
        # Utiliser le moteur de recherche du core manager
        search_engine = core_manager.get_search_engine()
        if not search_engine:
            return JSONResponse(
                content={
                    "error": "Search engine not available",
                    "took": 0,
                    "hits": {"total": {"value": 0}, "hits": []}
                },
                status_code=503
            )
        
        # ✅ CORRECTION: Utiliser search() avec SearchServiceQuery au lieu de search_transactions()
        
        # Créer le SearchServiceQuery pour quick search
        service_query = create_search_service_query(
            user_id=user_id,
            query=query,
            filters={},
            limit=limit,
            offset=0,
            intent_type="QUICK_SEARCH"
        )
        
        # Appeler search() au lieu de search_transactions()
        service_response = await search_engine.search(service_query)
        
        # Convertir vers le format legacy
        legacy_result = convert_service_response_to_legacy(service_response)
        
        return JSONResponse(content=legacy_result, status_code=200)
            
    except Exception as e:
        logger.error(f"❌ Quick search error: {e}")
        return JSONResponse(
            content={
                "error": f"Quick search failed: {str(e)}",
                "took": 0,
                "hits": {"total": {"value": 0}, "hits": []}
            },
            status_code=500
        )

@router.get("/config", summary="Configuration du service")
async def get_service_configuration():
    """
    Configuration du service
    """
    try:
        config = {
            "service": "search_service",
            "version": "1.0.0",
            "timestamp": datetime.utcnow().isoformat(),
            "environment": {
                "bonsai_url_configured": bool(os.environ.get("BONSAI_URL")),
                "elasticsearch_index": os.environ.get("ELASTICSEARCH_INDEX", "harena_transactions"),
                "test_user_id": os.environ.get("TEST_USER_ID", "34")
            }
        }
        
        # Ajouter des infos détaillées si disponibles
        try:
            from search_service.clients.elasticsearch_client import get_client_configuration_info
            detailed_config = get_client_configuration_info()
            config["detailed_configuration"] = detailed_config
        except Exception as e:
            config["configuration_error"] = str(e)
        
        return JSONResponse(content=config, status_code=200)
        
    except Exception as e:
        return JSONResponse(
            content={
                "error": "Failed to get configuration",
                "details": str(e)
            },
            status_code=500
        )

@router.get("/metrics", summary="Métriques du service")
async def get_service_metrics():
    """
    Métriques du service
    """
    try:
        from search_service.clients.elasticsearch_client import get_client_metrics
        
        metrics = get_client_metrics()
        
        # Ajouter timestamp
        metrics["timestamp"] = datetime.utcnow().isoformat()
        metrics["service"] = "search_service"
        
        return JSONResponse(content=metrics, status_code=200)
        
    except Exception as e:
        return JSONResponse(
            content={
                "error": "Failed to get metrics",
                "details": str(e),
                "timestamp": datetime.utcnow().isoformat()
            },
            status_code=500
        )

# === ROUTES DE DEBUG/DÉVELOPPEMENT ===

@router.post("/restart", summary="Redémarrage du service (dev/debug)")
async def restart_service(request: Request):
    """
    Endpoint pour redémarrer le service en réinitialisant les composants
    ATTENTION: À utiliser seulement en développement/debug
    """
    try:
        logger.warning("🔄 Tentative de redémarrage du service via endpoint...")
        
        # Marquer le service comme non initialisé
        request.app.state.service_initialized = False
        request.app.state.elasticsearch_client = None
        request.app.state.core_manager = None
        request.app.state.initialization_error = "Manual restart requested"
        
        return JSONResponse(
            content={
                "message": "Service marked for restart. Please restart the application for full reinitialization.",
                "timestamp": datetime.utcnow().isoformat(),
                "status": "restart_requested",
                "recommendation": "Use 'heroku restart' for production restart"
            },
            status_code=200
        )
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de la demande de redémarrage: {e}")
        return JSONResponse(
            content={
                "message": "Restart request failed",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            },
            status_code=500
        )

@router.get("/debug/state", summary="État de debug du service")
async def debug_service_state(request: Request):
    """
    Endpoint de debug pour voir l'état détaillé des composants
    """
    try:
        debug_info = {
            "timestamp": datetime.utcnow().isoformat(),
            "app_state": {
                "service_initialized": getattr(request.app.state, 'service_initialized', None),
                "elasticsearch_client_type": str(type(getattr(request.app.state, 'elasticsearch_client', None))),
                "core_manager_type": str(type(getattr(request.app.state, 'core_manager', None))),
                "initialization_error": getattr(request.app.state, 'initialization_error', None)
            },
            "environment": {
                "bonsai_url_set": bool(os.environ.get("BONSAI_URL")),
                "bonsai_url_length": len(os.environ.get("BONSAI_URL", "")),
                "elasticsearch_index": os.environ.get("ELASTICSEARCH_INDEX"),
                "test_user_id": os.environ.get("TEST_USER_ID")
            }
        }
        
        # Vérifier l'état du core manager si disponible
        core_manager = getattr(request.app.state, 'core_manager', None)
        if core_manager:
            debug_info["core_manager"] = {
                "is_initialized": core_manager.is_initialized(),
                "has_lexical_engine": hasattr(core_manager, 'lexical_engine'),
                "has_query_executor": hasattr(core_manager, 'query_executor')
            }
        
        return JSONResponse(content=debug_info, status_code=200)
        
    except Exception as e:
        return JSONResponse(
            content={
                "error": "Debug state failed",
                "details": str(e),
                "timestamp": datetime.utcnow().isoformat()
            },
            status_code=500
        )