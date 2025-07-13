"""
Routes API REST pour le Search Service
=====================================

API REST spécialisée exclusivement dans la recherche lexicale Elasticsearch :
- POST /search/lexical - Recherche lexicale avec requête structurée
- POST /search/validate - Validation requête Elasticsearch  
- GET /search/templates - Liste templates disponibles
- GET /health - Santé service avec détails composants
- GET /metrics - Métriques performance exportables
- GET /admin/* - Endpoints d'administration (cache, config)

Architecture :
    Client → FastAPI Routes → Dependencies → Core Components → Elasticsearch
"""

import logging
import asyncio
import time
from typing import Dict, Optional, Any
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, Request, Query, Path, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from starlette.status import HTTP_200_OK, HTTP_400_BAD_REQUEST

from search_service.models.service_contracts import SearchServiceQuery, SearchServiceResponse
from search_service.models.responses import ValidationResponse, TemplateListResponse, HealthResponse
from search_service.api.dependencies import (
    get_authenticated_user,
    validate_search_request,
    validate_rate_limit,
    check_service_health,
    add_response_headers
)
from search_service.core import (
    get_lexical_engine, get_query_executor, get_result_processor,
    get_performance_optimizer, get_core_health, get_core_performance
)
from search_service.templates import template_manager
from search_service.utils import (
    get_system_metrics, get_performance_summary, 
    cleanup_old_metrics, get_cache_manager, get_utils_health,
    get_utils_performance
)
from search_service.config import settings


logger = logging.getLogger(__name__)

# === ROUTEUR PRINCIPAL ===

router = APIRouter(tags=["search"])


# === FONCTION D'AIDE POUR ADMIN ===

async def require_admin_permission(
    user_info: Dict[str, Any] = Depends(get_authenticated_user)
) -> Dict[str, Any]:
    """Vérifie les permissions admin"""
    if "admin" not in user_info.get("permissions", []):
        raise HTTPException(
            status_code=403,
            detail="Admin permission required"
        )
    return user_info


# === ENDPOINTS DE RECHERCHE ===

@router.post(
    "/lexical",
    response_model=SearchServiceResponse,
    summary="Recherche lexicale Elasticsearch",
    description="""
    Effectue une recherche lexicale haute performance sur les données financières.
    
    **Fonctionnalités :**
    - Recherche textuelle BM25 optimisée
    - Filtrage exact et par plages
    - Agrégations statistiques
    - Cache intelligent
    - Optimisations automatiques
    
    **Performance :**
    - Cible : < 50ms pour requêtes simples
    - Cible : < 200ms pour requêtes complexes avec agrégations
    
    **Sécurité :**
    - Isolation utilisateur stricte (user_id obligatoire)
    - Rate limiting par utilisateur
    - Validation complète des entrées
    """,
    responses={
        200: {"description": "Recherche exécutée avec succès"},
        400: {"description": "Requête invalide - erreurs de validation"},
        401: {"description": "Authentification requise"},
        403: {"description": "Permissions insuffisantes"},
        429: {"description": "Limite de taux dépassée"},
        503: {"description": "Service temporairement indisponible"}
    }
)
async def search_lexical(
    request: Request,
    search_query: SearchServiceQuery,
    user_info: Dict[str, Any] = Depends(get_authenticated_user),
    validated_request: SearchServiceQuery = Depends(validate_search_request),
    rate_limit_info: Dict[str, Any] = Depends(lambda r: validate_rate_limit("search", r)),
    _health_check = Depends(check_service_health)
) -> SearchServiceResponse:
    """
    Endpoint principal de recherche lexicale
    
    Traite une requête de recherche structurée et retourne les résultats
    avec métriques de performance et informations contextuelles.
    """
    
    correlation_id = getattr(request.state, 'correlation_id', 'unknown')
    start_time = time.time()
    
    try:
        # Récupérer le moteur lexical
        lexical_engine = get_lexical_engine()
        if not lexical_engine:
            raise HTTPException(
                status_code=503,
                detail="Lexical search engine not available"
            )
        
        # Exécuter la recherche avec profiling automatique
        logger.info(
            f"Executing lexical search for user {user_info['user_id']} "
            f"[{correlation_id}] - Intent: {search_query.query_metadata.intent_type}"
        )
        
        # La requête a déjà été validée par les dépendances
        search_response = await lexical_engine.search(validated_request)
        
        # Calculer les métriques de performance
        total_duration_ms = (time.time() - start_time) * 1000
        
        # Enrichir la réponse avec informations de la requête
        search_response.response_metadata.correlation_id = correlation_id
        search_response.response_metadata.request_user_id = user_info['user_id']
        search_response.response_metadata.auth_method = user_info.get('auth_method')
        
        # Ajouter headers de performance
        response_headers = {
            "X-Total-Duration": f"{total_duration_ms:.2f}ms",
            "X-Results-Count": str(search_response.response_metadata.returned_hits),
            "X-Cache-Hit": str(search_response.performance.cache_hit),
            **getattr(request.state, 'rate_limit_headers', {})
        }
        
        # Créer la réponse avec headers
        response = JSONResponse(
            content=search_response.dict(),
            status_code=HTTP_200_OK,
            headers=response_headers
        )
        
        # Ajouter headers standard
        add_response_headers(response, rate_limit_info)
        
        logger.info(
            f"Search completed for user {user_info['user_id']} [{correlation_id}] - "
            f"Duration: {total_duration_ms:.1f}ms, Results: {search_response.response_metadata.returned_hits}"
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Search failed for user {user_info['user_id']} [{correlation_id}]: {e}",
            exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail=f"Search execution failed: {str(e)}"
        )


@router.post(
    "/validate",
    response_model=ValidationResponse,
    summary="Validation de requête de recherche",
    description="""
    Valide une requête de recherche sans l'exécuter.
    
    **Utilisations :**
    - Validation côté client avant envoi
    - Tests de conformité des requêtes
    - Estimation de performance
    - Débogage de requêtes complexes
    
    **Validations effectuées :**
    - Structure des contrats Pydantic
    - Sécurité (isolation utilisateur)
    - Performance (complexité, limites)
    - Cohérence des filtres et agrégations
    """,
    responses={
        200: {"description": "Validation complète avec détails"},
        400: {"description": "Requête invalide avec erreurs détaillées"},
        401: {"description": "Authentification requise"},
        429: {"description": "Limite de taux dépassée"}
    }
)
async def validate_search_request_endpoint(
    request: Request,
    search_query: SearchServiceQuery,
    user_info: Dict[str, Any] = Depends(get_authenticated_user),
    rate_limit_info: Dict[str, Any] = Depends(lambda r: validate_rate_limit("validate", r))
) -> ValidationResponse:
    """
    Valide une requête de recherche et retourne un rapport détaillé
    """
    
    correlation_id = getattr(request.state, 'correlation_id', 'unknown')
    start_time = time.time()
    
    try:
        from search_service.utils.validators import ValidatorFactory
        
        logger.debug(f"Validating search request for user {user_info['user_id']} [{correlation_id}]")
        
        # Validation complète avec tous les validateurs
        validation_result = ValidatorFactory.validate_complete_request(search_query)
        
        # Calculer les métriques de validation
        validation_duration_ms = (time.time() - start_time) * 1000
        
        # Construire la réponse de validation
        response = ValidationResponse(
            valid=validation_result["valid"],
            errors=validation_result["errors"],
            warnings=validation_result["warnings"],
            security_check=validation_result["security_check"],
            performance_analysis={
                "complexity": validation_result["performance_check"]["complexity"],
                "estimated_time_ms": validation_result["estimated_time_ms"],
                "warnings": validation_result["performance_check"]["warnings"],
                "optimization_suggestions": [
                    "Consider adding more specific filters for better performance",
                    "Use pagination for large result sets",
                    "Cache frequently used queries"
                ] if validation_result["performance_check"]["complexity"] == "complex" else []
            },
            metadata={
                "validation_duration_ms": validation_duration_ms,
                "correlation_id": correlation_id,
                "user_id": user_info['user_id'],
                "timestamp": datetime.now().isoformat()
            }
        )
        
        # Headers de réponse
        response_headers = {
            "X-Validation-Duration": f"{validation_duration_ms:.2f}ms",
            **getattr(request.state, 'rate_limit_headers', {})
        }
        
        response_json = JSONResponse(
            content=response.dict(),
            status_code=HTTP_200_OK,
            headers=response_headers
        )
        
        add_response_headers(response_json, rate_limit_info)
        
        logger.debug(
            f"Validation completed for user {user_info['user_id']} [{correlation_id}] - "
            f"Valid: {response.valid}, Duration: {validation_duration_ms:.1f}ms"
        )
        
        return response_json
        
    except Exception as e:
        logger.error(f"Validation failed [{correlation_id}]: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Validation process failed: {str(e)}"
        )


# === ENDPOINTS DE CONFIGURATION ===

@router.get(
    "/templates",
    response_model=TemplateListResponse,
    summary="Liste des templates de requêtes",
    description="""
    Retourne la liste des templates de requêtes disponibles par intention.
    
    **Templates disponibles :**
    - Templates par intention financière (expense_analysis, merchant_search, etc.)
    - Templates d'agrégation (temporal, categorical, statistical)
    - Templates optimisés par complexité
    
    **Utilisation :**
    - Construction assistée de requêtes côté client
    - Documentation des capacités du service
    - Optimisation des requêtes fréquentes
    """,
    responses={
        200: {"description": "Liste des templates avec détails"},
        401: {"description": "Authentification requise"},
        429: {"description": "Limite de taux dépassée"}
    }
)
async def list_query_templates(
    request: Request,
    category: Optional[str] = Query(None, description="Filtrer par catégorie de template"),
    intent_type: Optional[str] = Query(None, description="Filtrer par type d'intention"),
    user_info: Dict[str, Any] = Depends(get_authenticated_user),
    rate_limit_info: Dict[str, Any] = Depends(lambda r: validate_rate_limit("validate", r))
) -> TemplateListResponse:
    """
    Liste les templates de requêtes disponibles avec filtrage optionnel
    """
    
    correlation_id = getattr(request.state, 'correlation_id', 'unknown')
    
    try:
        logger.debug(f"Listing query templates [{correlation_id}] - Category: {category}, Intent: {intent_type}")
        
        # Récupérer les templates via le gestionnaire
        all_templates = await template_manager.get_available_templates()
        
        # Filtrer selon les paramètres
        filtered_templates = all_templates
        
        if category:
            filtered_templates = {
                name: template for name, template in filtered_templates.items()
                if template.get("category") == category
            }
        
        if intent_type:
            filtered_templates = {
                name: template for name, template in filtered_templates.items()
                if template.get("intent_type") == intent_type
            }
        
        # Construire la réponse
        response = TemplateListResponse(
            templates=filtered_templates,
            total_count=len(filtered_templates),
            categories=list(set(t.get("category", "unknown") for t in all_templates.values())),
            intent_types=list(set(t.get("intent_type", "unknown") for t in all_templates.values())),
            metadata={
                "correlation_id": correlation_id,
                "filtered": bool(category or intent_type),
                "filter_category": category,
                "filter_intent_type": intent_type,
                "timestamp": datetime.now().isoformat()
            }
        )
        
        response_headers = getattr(request.state, 'rate_limit_headers', {})
        
        response_json = JSONResponse(
            content=response.dict(),
            status_code=HTTP_200_OK,
            headers=response_headers
        )
        
        add_response_headers(response_json, rate_limit_info)
        
        return response_json
        
    except Exception as e:
        logger.error(f"Failed to list templates [{correlation_id}]: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve templates: {str(e)}"
        )


# === ENDPOINTS DE SANTÉ ET MONITORING ===

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Santé détaillée du service",
    description="""
    Vérification complète de la santé du Search Service.
    
    **Composants vérifiés :**
    - Moteur lexical et ses dépendances
    - Connexion Elasticsearch
    - Gestionnaires de cache et métriques
    - Optimiseur de performance
    - Processors de résultats
    
    **Niveaux de santé :**
    - healthy : Tous composants opérationnels
    - degraded : Certains composants en difficulté
    - unhealthy : Composants critiques défaillants
    """,
    responses={
        200: {"description": "Statut de santé avec détails complets"},
        503: {"description": "Service dégradé ou indisponible"}
    }
)
async def health_check(request: Request) -> HealthResponse:
    """
    Vérification complète de la santé du service
    """
    
    correlation_id = getattr(request.state, 'correlation_id', 'unknown')
    start_time = time.time()
    
    try:
        logger.debug(f"Health check started [{correlation_id}]")
        
        # Exécuter les vérifications en série pour éviter le problème asyncio.gather
        # Initialiser les résultats
        core_health = None
        utils_health = None
        es_health = None
        cache_health = None
        
        # Vérifications individuelles avec gestion d'erreur
        try:
            core_health = await get_core_health()
        except Exception as e:
            core_health = {"status": "error", "error": str(e)}
        
        try:
            utils_health = await get_utils_health()
        except Exception as e:
            utils_health = {"status": "error", "error": str(e)}
        
        try:
            es_health = await _check_elasticsearch_connectivity()
        except Exception as e:
            es_health = {"status": "error", "error": str(e)}
        
        try:
            cache_health = await _check_cache_health()
        except Exception as e:
            cache_health = {"status": "error", "error": str(e)}
        
        # Construire les informations des composants avec sérialisation datetime
        components = []
        all_healthy = True
        current_time_iso = datetime.now().isoformat()
        
        # Composant Core
        if isinstance(core_health, dict) and core_health.get("system_status") == "healthy":
            components.append({
                "name": "core_engine",
                "status": "healthy",
                "last_check": current_time_iso,
                "response_time_ms": core_health.get("check_duration_ms", 0),
                "dependencies": ["elasticsearch", "query_executor"],
                "metrics": core_health
            })
        else:
            all_healthy = False
            components.append({
                "name": "core_engine",
                "status": "unhealthy",
                "last_check": current_time_iso,
                "error_message": core_health.get("error", "Core health check failed") if isinstance(core_health, dict) else str(core_health),
                "dependencies": ["elasticsearch", "query_executor"],
                "metrics": {}
            })
        
        # Composant Elasticsearch
        if isinstance(es_health, dict) and es_health.get("connected"):
            components.append({
                "name": "elasticsearch",
                "status": "healthy",
                "last_check": current_time_iso,
                "response_time_ms": es_health.get("response_time_ms", 0),
                "dependencies": [],
                "metrics": {
                    "cluster_name": es_health.get("cluster_name", "unknown"),
                    "nodes": es_health.get("nodes", 0),
                    "status": es_health.get("status", "unknown")
                }
            })
        else:
            all_healthy = False
            components.append({
                "name": "elasticsearch",
                "status": "unhealthy",
                "last_check": current_time_iso,
                "error_message": es_health.get("error", "Elasticsearch connection failed") if isinstance(es_health, dict) else str(es_health),
                "dependencies": [],
                "metrics": {}
            })
        
        # Composant Utils
        if isinstance(utils_health, dict) and utils_health.get("system_status") == "healthy":
            components.append({
                "name": "utils",
                "status": "healthy",
                "last_check": current_time_iso,
                "response_time_ms": utils_health.get("check_duration_ms", 0),
                "dependencies": ["cache"],
                "metrics": utils_health
            })
        else:
            components.append({
                "name": "utils",
                "status": "degraded",  # Utils non critiques
                "last_check": current_time_iso,
                "error_message": utils_health.get("error", "Utils check failed") if isinstance(utils_health, dict) else str(utils_health),
                "dependencies": ["cache"],
                "metrics": {}
            })
        
        # Composant Cache
        if isinstance(cache_health, dict) and cache_health.get("status") in ["healthy", "available"]:
            components.append({
                "name": "cache",
                "status": "healthy",
                "last_check": current_time_iso,
                "dependencies": [],
                "metrics": cache_health
            })
        else:
            components.append({
                "name": "cache",
                "status": "degraded",  # Cache non critique
                "last_check": current_time_iso,
                "error_message": cache_health.get("error", "Cache check failed") if isinstance(cache_health, dict) else str(cache_health),
                "dependencies": [],
                "metrics": {}
            })
        
        # Déterminer le statut global
        unhealthy_count = len([c for c in components if c["status"] == "unhealthy"])
        degraded_count = len([c for c in components if c["status"] == "degraded"])
        
        if unhealthy_count == 0 and degraded_count == 0:
            overall_status = "healthy"
            status_code = HTTP_200_OK
        elif unhealthy_count == 0:
            overall_status = "degraded" 
            status_code = HTTP_200_OK
        else:
            overall_status = "unhealthy"
            status_code = 503
        
        # Métriques de performance
        health_check_duration = (time.time() - start_time) * 1000
        
        # Construire la réponse avec le bon modèle et sérialisation datetime
        system_health = {
            "overall_status": overall_status,
            "uptime_seconds": time.time() - getattr(request.app.state, 'start_time', time.time()),
            "memory_usage_mb": _get_memory_usage_mb(),
            "cpu_usage_percent": _get_cpu_usage_percent(),
            "active_connections": _get_active_connections(),
            "total_requests": _get_request_count_24h(),
            "error_rate_percent": _get_error_rate_24h()
        }
        
        response_data = {
            "system": system_health,
            "components": components,
            "timestamp": datetime.now().isoformat(),
            "service_version": "1.0.0",
            "environment": getattr(settings, 'environment', 'production'),
            "metadata": {
                "correlation_id": correlation_id,
                "health_check_duration_ms": health_check_duration,
                "components_checked": len(components)
            }
        }
        
        # Créer la réponse JSON directement avec les données sérialisées
        response_json = JSONResponse(
            content=response_data,
            status_code=status_code,
            headers={"X-Health-Check-Duration": f"{health_check_duration:.2f}ms"}
        )
        
        logger.info(
            f"Health check completed [{correlation_id}] - "
            f"Status: {overall_status}, Duration: {health_check_duration:.1f}ms"
        )
        
        return response_json
        
    except Exception as e:
        logger.error(f"Health check failed [{correlation_id}]: {e}", exc_info=True)
        
        # Réponse d'erreur minimale avec sérialisation datetime correcte
        error_system = {
            "overall_status": "unhealthy",
            "uptime_seconds": 0.0,
            "memory_usage_mb": 0.0,
            "cpu_usage_percent": 0.0,
            "active_connections": 0,
            "total_requests": 0,
            "error_rate_percent": 100.0
        }
        
        error_components = [{
            "name": "system",
            "status": "unhealthy",
            "last_check": datetime.now().isoformat(),
            "error_message": f"Health check process failed: {str(e)}",
            "dependencies": [],
            "metrics": {}
        }]
        
        error_response_data = {
            "system": error_system,
            "components": error_components,
            "timestamp": datetime.now().isoformat(),
            "service_version": "1.0.0",
            "environment": "unknown",
            "metadata": {
                "correlation_id": correlation_id,
                "error_timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
        }
        
        return JSONResponse(
            content=error_response_data,
            status_code=503
        )


@router.get(
    "/metrics",
    response_model=None,
    summary="Métriques de performance exportables",
    description="""
    Exporte les métriques de performance du Search Service.
    
    **Formats supportés :**
    - json : Métriques détaillées avec historique
    - prometheus : Format compatible Prometheus/Grafana
    - summary : Résumé performance dernières 24h
    
    **Métriques incluses :**
    - Performance API (latence, throughput, erreurs)
    - Métriques Elasticsearch (temps requêtes, cache hit rate)
    - Métriques système (CPU, mémoire, I/O)
    - Métriques métier (taux succès recherches, etc.)
    """,
    responses={
        200: {
            "description": "Métriques exportées selon le format demandé",
            "content": {
                "application/json": {"example": {"service": "search-service", "metrics": "..."}},
                "text/plain": {"example": "# Prometheus metrics\n..."}
            }
        },
        401: {"description": "Authentification requise"},
        403: {"description": "Permission métriques requise"},
        429: {"description": "Limite de taux dépassée"}
    }
)
async def export_metrics(
    request: Request,
    format: str = Query("json", description="Format d'export: json, prometheus, summary"),
    hours: int = Query(24, ge=1, le=168, description="Période en heures (1-168)"),
    user_info: Dict[str, Any] = Depends(get_authenticated_user),
    rate_limit_info: Dict[str, Any] = Depends(lambda r: validate_rate_limit("metrics", r)),
    _permissions_check = Depends(lambda u: _check_metrics_permission(u))
) -> Response:
    """
    Exporte les métriques selon le format demandé
    """
    
    correlation_id = getattr(request.state, 'correlation_id', 'unknown')
    
    try:
        logger.debug(f"Exporting metrics [{correlation_id}] - Format: {format}, Hours: {hours}")
        
        if format == "json":
            # Export JSON détaillé
            metrics_data = {
                "metadata": {
                    "service": "search-service",
                    "export_timestamp": datetime.now().isoformat(),
                    "period_hours": hours,
                    "correlation_id": correlation_id
                },
                "system_metrics": get_system_metrics(),
                "performance_summary": get_performance_summary(hours=hours),
                "core_performance": await get_core_performance(),
                "utils_performance": await get_utils_performance() 
            }
            
            response = JSONResponse(
                content=metrics_data,
                headers={
                    "Content-Type": "application/json",
                    **getattr(request.state, 'rate_limit_headers', {})
                }
            )
            
        elif format == "prometheus":
            # Export format Prometheus
            from search_service.utils.metrics import metrics_collector
            prometheus_data = metrics_collector.export_metrics("prometheus")
            
            response = PlainTextResponse(
                content=prometheus_data,
                headers={
                    "Content-Type": "text/plain; version=0.0.4",
                    **getattr(request.state, 'rate_limit_headers', {})
                }
            )
            
        elif format == "summary":
            # Résumé de performance
            summary_data = {
                "service": "search-service",
                "period_hours": hours,
                "summary": get_performance_summary(hours=hours),
                "key_metrics": {
                    "avg_search_time_ms": _get_avg_search_time(hours),
                    "search_success_rate": _get_search_success_rate(hours),
                    "cache_hit_rate": _get_cache_hit_rate(),
                    "total_searches": _get_total_searches(hours)
                },
                "timestamp": datetime.now().isoformat()
            }
            
            response = JSONResponse(
                content=summary_data,
                headers=getattr(request.state, 'rate_limit_headers', {})
            )
            
        else:
            raise HTTPException(
                status_code=HTTP_400_BAD_REQUEST,
                detail=f"Unsupported format: {format}. Use: json, prometheus, summary"
            )
        
        add_response_headers(response, rate_limit_info)
        
        logger.debug(f"Metrics exported [{correlation_id}] - Format: {format}")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Metrics export failed [{correlation_id}]: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Metrics export failed: {str(e)}"
        )


# === ENDPOINTS D'ADMINISTRATION ===

admin_router = APIRouter(tags=["admin"])

@admin_router.post(
    "/cache/clear",
    summary="Vider le cache du service",
    description="Vide tous les caches du Search Service (requêtes, résultats, templates)",
    dependencies=[Depends(require_admin_permission)]
)
async def clear_cache(request: Request) -> JSONResponse:
    """Vide tous les caches du service"""
    
    correlation_id = getattr(request.state, 'correlation_id', 'unknown')
    
    try:
        logger.info(f"Clearing all caches [{correlation_id}]")
        
        # Vider les différents caches
        cleared_caches = {}
        
        # Cache du moteur lexical
        lexical_engine = get_lexical_engine()
        if lexical_engine:
            lexical_engine.clear_cache()
            cleared_caches["lexical_engine"] = "cleared"
        
        # Cache de l'exécuteur de requêtes
        query_executor = get_query_executor()
        if query_executor:
            query_executor.clear_cache()
            cleared_caches["query_executor"] = "cleared"
        
        # Cache global des utilitaires
        cache_manager = get_cache_manager()
        if cache_manager:
            await cache_manager.clear_all()
            cleared_caches["utils_cache"] = "cleared"
        
        # Nettoyer les anciennes métriques
        cleanup_old_metrics(hours=1)
        cleared_caches["old_metrics"] = "cleaned"
        
        response_data = {
            "message": "All caches cleared successfully",
            "cleared_caches": cleared_caches,
            "timestamp": datetime.now().isoformat(),
            "correlation_id": correlation_id
        }
        
        logger.info(f"All caches cleared [{correlation_id}]")
        
        return JSONResponse(content=response_data)
        
    except Exception as e:
        logger.error(f"Cache clear failed [{correlation_id}]: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Cache clear failed: {str(e)}"
        )


@admin_router.get(
    "/config",
    summary="Configuration actuelle du service",
    description="Retourne la configuration actuelle du Search Service (sans secrets)",
    dependencies=[Depends(require_admin_permission)]
)
async def get_service_config(request: Request) -> JSONResponse:
    """Retourne la configuration du service"""
    
    correlation_id = getattr(request.state, 'correlation_id', 'unknown')
    
    try:
        # Extraire la configuration publique (sans secrets)
        config_data = {
            "service": {
                "name": "search-service",
                "version": "1.0.0",
                "environment": getattr(settings, 'environment', 'unknown')
            },
            "elasticsearch": {
                "index": getattr(settings, 'elasticsearch_index', 'unknown'),
                "timeout_ms": getattr(settings, 'elasticsearch_timeout_ms', 5000),
                "max_retries": getattr(settings, 'elasticsearch_max_retries', 3)
            },
            "api": {
                "max_request_size_mb": getattr(settings, 'max_request_size_bytes', 0) / (1024*1024),
                "request_timeout_s": getattr(settings, 'request_timeout_seconds', 30),
                "enable_compression": getattr(settings, 'enable_compression', True)
            },
            "cache": {
                "enabled": True,
                "default_ttl_seconds": getattr(settings, 'cache_default_ttl_seconds', 300)
            },
            "metrics": {
                "enabled": getattr(settings, 'metrics_enabled', True),
                "retention_hours": getattr(settings, 'metrics_retention_hours', 24)
            },
            "metadata": {
                "correlation_id": correlation_id,
                "timestamp": datetime.now().isoformat()
            }
        }
        
        return JSONResponse(content=config_data)
        
    except Exception as e:
        logger.error(f"Config retrieval failed [{correlation_id}]: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Config retrieval failed: {str(e)}"
        )


# === FONCTIONS UTILITAIRES PRIVÉES ===

async def _check_elasticsearch_connectivity() -> Dict[str, Any]:
    """Vérifie la connectivité Elasticsearch"""
    
    try:
        lexical_engine = get_lexical_engine()
        if not lexical_engine:
            return {"connected": False, "error": "Lexical engine not available"}
        
        # Tester via une requête simple
        health = await lexical_engine.health_check()
        
        return {
            "connected": health.get("status") == "healthy",
            "cluster_name": health.get("cluster_name", "unknown"),
            "nodes": health.get("nodes", 0),
            "status": health.get("elasticsearch_status", "unknown"),
            "response_time_ms": health.get("response_time_ms", 0)
        }
        
    except Exception as e:
        return {"connected": False, "error": str(e)}


async def _check_cache_health() -> Dict[str, Any]:
    """Vérifie la santé du système de cache"""
    
    try:
        cache_manager = get_cache_manager()
        if not cache_manager:
            return {"status": "not_available"}
        
        health = await cache_manager.get_health_status()
        return health
        
    except Exception as e:
        return {"status": "error", "error": str(e)}


def _check_metrics_permission(user_info: Dict[str, Any]) -> bool:
    """Vérifie la permission d'accès aux métriques"""
    
    if "metrics" not in user_info.get("permissions", []):
        raise HTTPException(
            status_code=403,
            detail="Metrics permission required"
        )
    return True


def _get_memory_usage_mb() -> float:
    """Récupère l'usage mémoire en MB"""
    try:
        import psutil
        process = psutil.Process()
        return round(process.memory_info().rss / (1024 * 1024), 2)
    except:
        return 0.0


def _get_cpu_usage_percent() -> float:
    """Récupère l'usage CPU en pourcentage"""
    try:
        import psutil
        return round(psutil.cpu_percent(interval=0.1), 2)
    except:
        return 0.0


def _get_active_connections() -> int:
    """Récupère le nombre de connexions actives"""
    try:
        # À implémenter selon le système de monitoring
        return 0
    except:
        return 0


def _get_request_count_24h() -> int:
    """Récupère le nombre de requêtes des dernières 24h"""
    
    try:
        from search_service.utils.metrics import metrics_collector
        since = datetime.now() - timedelta(hours=24)
        stats = metrics_collector.get_metric_stats("api_request_count", since)
        return int(stats.get("sum", 0))
    except:
        return 0


def _get_avg_response_time() -> float:
    """Récupère le temps de réponse moyen"""
    
    try:
        from search_service.utils.metrics import metrics_collector
        since = datetime.now() - timedelta(hours=1)
        stats = metrics_collector.get_metric_stats("api_request_duration_ms", since)
        return round(stats.get("avg", 0), 2)
    except:
        return 0.0


def _get_error_rate_24h() -> float:
    """Récupère le taux d'erreur des dernières 24h"""
    
    try:
        from search_service.utils.metrics import metrics_collector
        since = datetime.now() - timedelta(hours=24)
        
        error_stats = metrics_collector.get_metric_stats("api_error_count", since)
        request_stats = metrics_collector.get_metric_stats("api_request_count", since)
        
        errors = error_stats.get("sum", 0)
        requests = request_stats.get("sum", 0)
        
        if requests > 0:
            return round((errors / requests) * 100, 2)
        return 0.0
    except:
        return 0.0


def _get_avg_search_time(hours: int) -> float:
    """Récupère le temps moyen des recherches"""
    
    try:
        from search_service.utils.metrics import metrics_collector
        since = datetime.now() - timedelta(hours=hours)
        stats = metrics_collector.get_metric_stats("lexical_search_duration_ms", since)
        return round(stats.get("avg", 0), 2)
    except:
        return 0.0


def _get_search_success_rate(hours: int) -> float:
    """Récupère le taux de succès des recherches"""
    
    try:
        from search_service.utils.metrics import metrics_collector
        since = datetime.now() - timedelta(hours=hours)
        
        success_stats = metrics_collector.get_metric_stats("lexical_search_success_count", since)
        total_stats = metrics_collector.get_metric_stats("lexical_search_count", since)
        
        successes = success_stats.get("sum", 0)
        total = total_stats.get("sum", 0)
        
        if total > 0:
            return round((successes / total) * 100, 2)
        return 100.0  # Aucune recherche = 100% par défaut
    except:
        return 0.0


def _get_cache_hit_rate() -> float:
    """Récupère le taux de cache hit"""
    
    try:
        from search_service.utils.metrics import metrics_collector
        return metrics_collector.get_current_value("lexical_cache_hit_rate", 0.0)
    except:
        return 0.0


def _get_total_searches(hours: int) -> int:
    """Récupère le nombre total de recherches"""
    
    try:
        from search_service.utils.metrics import metrics_collector
        since = datetime.now() - timedelta(hours=hours)
        stats = metrics_collector.get_metric_stats("lexical_search_count", since)
        return int(stats.get("sum", 0))
    except:
        return 0


# === ROUTEUR COMBINÉ ===

# Inclure le routeur admin dans le routeur principal
router.include_router(admin_router, prefix="/admin")


# === GESTIONNAIRE D'ERREURS SPÉCIALISÉ ===
# Note: Les exception handlers doivent être ajoutés sur l'app FastAPI principal, pas sur le router

async def custom_http_exception_handler(request: Request, exc: HTTPException):
    """Gestionnaire d'erreurs personnalisé pour les routes API"""
    
    correlation_id = getattr(request.state, 'correlation_id', 'unknown')
    
    # Enrichir la réponse d'erreur avec contexte
    error_response = {
        "error": {
            "code": exc.status_code,
            "message": exc.detail,
            "correlation_id": correlation_id,
            "timestamp": datetime.now().isoformat(),
            "path": request.url.path
        }
    }
    
    # Ajouter des détails selon le type d'erreur
    if exc.status_code == 400:
        error_response["error"]["type"] = "validation_error"
        error_response["help"] = "Check request format and required fields"
    elif exc.status_code == 401:
        error_response["error"]["type"] = "authentication_error"
        error_response["help"] = "Provide valid authentication credentials"
    elif exc.status_code == 403:
        error_response["error"]["type"] = "authorization_error"
        error_response["help"] = "Insufficient permissions for this operation"
    elif exc.status_code == 429:
        error_response["error"]["type"] = "rate_limit_error"
        error_response["help"] = "Reduce request frequency or upgrade plan"
    elif exc.status_code == 503:
        error_response["error"]["type"] = "service_unavailable"
        error_response["help"] = "Service temporarily unavailable, try again later"
    
    # Logger l'erreur
    logger.warning(
        f"API error [{correlation_id}]: {exc.status_code} - {exc.detail} "
        f"on {request.method} {request.url.path}"
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response,
        headers={"X-Correlation-ID": correlation_id}
    )


# === HOOKS DE DÉMARRAGE ET ARRÊT ===

async def initialize_routes():
    """Initialise les composants nécessaires aux routes"""
    
    logger.info("Initialisation des routes API...")
    
    try:
        # Vérifier que tous les composants core sont disponibles
        components = {
            "lexical_engine": get_lexical_engine(),
            "query_executor": get_query_executor(),
            "result_processor": get_result_processor(),
            "performance_optimizer": get_performance_optimizer()
        }
        
        missing_components = [name for name, component in components.items() if component is None]
        
        if missing_components:
            raise RuntimeError(f"Missing core components: {missing_components}")
        
        # Initialiser le gestionnaire de templates
        await template_manager.initialize()
        
        logger.info("✅ Routes API initialisées avec succès")
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'initialisation des routes: {e}")
        raise


async def shutdown_routes():
    """Nettoie les ressources des routes"""
    
    logger.info("Arrêt des routes API...")
    
    try:
        # Exporter les métriques finales si configuré
        if getattr(settings, 'export_final_metrics', False):
            try:
                final_metrics = get_performance_summary(hours=1)
                logger.info(f"Métriques finales: {final_metrics}")
            except Exception as e:
                logger.warning(f"Erreur export métriques finales: {e}")
        
        logger.info("✅ Routes API arrêtées")
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'arrêt des routes: {e}")


# === MIDDLEWARE SPÉCIALISÉ POUR LES ROUTES ===
# Note: Le middleware doit être ajouté sur l'app FastAPI principal, pas sur le router

async def routes_middleware(request: Request, call_next):
    """Middleware spécialisé pour les routes API"""
    
    # Ajouter timestamp de démarrage pour mesures de performance
    request.state.route_start_time = time.time()
    
    # Traiter la requête
    response = await call_next(request)
    
    # Ajouter des headers spécifiques aux routes API
    if hasattr(request.state, 'route_start_time'):
        route_duration = (time.time() - request.state.route_start_time) * 1000
        response.headers["X-Route-Duration"] = f"{route_duration:.2f}ms"
    
    # Ajouter informations de version et service
    response.headers["X-API-Version"] = "1.0.0"
    response.headers["X-Service-Component"] = "api-routes"
    
    return response


# === DOCUMENTATION OPENAPI PERSONNALISÉE ===

def customize_openapi_schema():
    """Personnalise le schéma OpenAPI pour une meilleure documentation"""
    
    custom_schema = {
        "info": {
            "title": "Search Service API",
            "version": "1.0.0",
            "description": """
## Search Service API - Recherche Lexicale Haute Performance

API REST spécialisée dans la recherche lexicale sur données financières avec Elasticsearch.

### Fonctionnalités Principales

- **Recherche lexicale BM25** : Recherche textuelle optimisée
- **Filtrage avancé** : Filtres exacts et par plages 
- **Agrégations statistiques** : Comptages, sommes, moyennes
- **Cache intelligent** : Optimisation des performances
- **Métriques détaillées** : Monitoring et observabilité

### Architecture

```
Client → FastAPI → Dependencies → Core Components → Elasticsearch
```

### Authentification

Trois méthodes supportées :
1. **Bearer Token** : `Authorization: Bearer <token>`
2. **API Key** : `X-User-Id: <id>` + `X-API-Key: <key>`
3. **Mode dev** : `X-User-Id: <id>` (développement uniquement)

### Rate Limiting

Limites par tier utilisateur :
- **Standard** : 100 recherches/min
- **Premium** : 500 recherches/min  
- **Enterprise** : 2000 recherches/min

### Performance

- **Cible** : < 50ms pour requêtes simples
- **Cible** : < 200ms pour requêtes complexes
- **Cache hit rate** : > 80% visé

### Support

- **Documentation** : API interactive ci-dessous
- **Correlation ID** : Inclus dans chaque réponse pour traçabilité
- **Métriques** : Endpoint `/metrics` pour monitoring
            """,
            "contact": {
                "name": "Search Service Team",
                "email": "search-service@company.com"
            },
            "license": {
                "name": "Internal Use Only"
            }
        },
        "servers": [
            {
                "url": "/api/v1",
                "description": "API Search Service v1"
            }
        ],
        "components": {
            "securitySchemes": {
                "bearerAuth": {
                    "type": "http",
                    "scheme": "bearer",
                    "description": "Token d'authentification Bearer"
                },
                "apiKeyAuth": {
                    "type": "apiKey",
                    "in": "header",
                    "name": "X-API-Key",
                    "description": "API Key avec X-User-Id header"
                }
            }
        },
        "security": [
            {"bearerAuth": []},
            {"apiKeyAuth": []}
        ],
        "tags": [
            {
                "name": "search",
                "description": "Endpoints de recherche lexicale"
            },
            {
                "name": "admin", 
                "description": "Endpoints d'administration (permissions admin requises)"
            }
        ]
    }
    
    return custom_schema


# === EXPORTS PRINCIPAUX ===

__all__ = [
    # === ROUTEUR PRINCIPAL ===
    "router",
    "admin_router",
    
    # === ENDPOINTS PUBLICS ===
    "search_lexical",
    "validate_search_request_endpoint", 
    "list_query_templates",
    "health_check",
    "export_metrics",
    
    # === ENDPOINTS ADMIN ===
    "clear_cache",
    "get_service_config",
    
    # === FONCTIONS D'AIDE ===
    "require_admin_permission",
    
    # === GESTIONNAIRES ===
    "custom_http_exception_handler",
    "routes_middleware",
    
    # === FONCTIONS D'INITIALISATION ===
    "initialize_routes",
    "shutdown_routes",
    "customize_openapi_schema",
    
    # === FONCTIONS UTILITAIRES ===
    "_get_request_count_24h",
    "_get_avg_response_time", 
    "_get_error_rate_24h",
    "_get_avg_search_time",
    "_get_search_success_rate",
    "_get_cache_hit_rate",
    "_get_total_searches"
]


# === INFORMATIONS DU MODULE ===

__version__ = "1.0.0"
__author__ = "Search Service Team"
__description__ = "Routes API REST spécialisées pour le Search Service"

logger.info(f"Module api.routes chargé - version {__version__}")