"""
Routes API pour le Search Service.

Ce module contient tous les endpoints REST pour :
- Recherche lexicale
- Validation des requêtes
- Templates de requêtes
- Métriques et monitoring
- Health checks
"""

from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Depends, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from datetime import datetime
import logging
from uuid import uuid4

from ..models.requests import (
    LexicalSearchRequest, 
    SearchOptions, 
    QueryValidationRequest
)
from ..models.responses import (
    SearchResponse, 
    ErrorResponse, 
    ValidationResponse,
    HealthResponse,
    MetricsResponse,
    TemplateListResponse
)
from ..models.service_contracts import (
    SearchServiceQuery, 
    SearchServiceResponse
)
from ..core.lexical_engine import LexicalEngine
from ..templates.query_templates import QueryTemplateManager
from ..utils.metrics import MetricsCollector
from ..config.settings import get_settings

from .dependencies import (
    CurrentUser,
    ValidatedRequest,
    LexicalEngineService,
    RequestContext,
    RateLimited,
    get_elasticsearch_client,
    format_error_response
)

logger = logging.getLogger(__name__)

# ==================== CONFIGURATION ROUTER ====================

# Router principal
router = APIRouter(
    prefix="/api/v1",
    tags=["search"],
    responses={
        400: {"description": "Requête invalide"},
        401: {"description": "Non autorisé"},
        429: {"description": "Limite de taux dépassée"},
        500: {"description": "Erreur serveur interne"}
    }
)

# Router pour les health checks (sans authentification)
health_router = APIRouter(
    tags=["health"],
    responses={
        200: {"description": "Service en bonne santé"},
        503: {"description": "Service indisponible"}
    }
)

# ==================== ENDPOINTS PRINCIPAUX ====================

@router.post(
    "/search/lexical",
    response_model=SearchResponse,
    summary="Recherche lexicale",
    description="Effectue une recherche lexicale pure via Elasticsearch"
)
async def search_lexical(
    request: ValidatedRequest,
    background_tasks: BackgroundTasks,
    lexical_engine: LexicalEngineService,
    current_user: CurrentUser,
    context: RequestContext,
    _: RateLimited
) -> SearchResponse:
    """
    Endpoint principal pour la recherche lexicale.
    
    Effectue une recherche lexicale optimisée via Elasticsearch
    avec support du cache et des métriques.
    """
    try:
        # Enrichissement du contexte
        request.request_id = request.request_id or str(uuid4())
        request.timestamp = datetime.utcnow()
        
        # Ajout des informations utilisateur
        search_context = {
            "user_id": current_user["user_id"],
            "user_role": current_user["role"],
            "request_context": context
        }
        
        # Exécution de la recherche
        logger.info(
            f"Recherche lexicale démarrée",
            extra={
                "request_id": request.request_id,
                "user_id": current_user["user_id"],
                "query": request.query[:100] + "..." if len(request.query) > 100 else request.query
            }
        )
        
        # Appel au moteur lexical
        results = await lexical_engine.search(request, search_context)
        
        # Métriques asynchrones
        background_tasks.add_task(
            _record_search_metrics,
            request.request_id,
            current_user["user_id"],
            request.query,
            len(results.documents) if results.documents else 0,
            results.total_results,
            results.search_time_ms
        )
        
        # Log du succès
        logger.info(
            f"Recherche lexicale terminée avec succès",
            extra={
                "request_id": request.request_id,
                "user_id": current_user["user_id"],
                "results_count": len(results.documents) if results.documents else 0,
                "total_results": results.total_results,
                "search_time_ms": results.search_time_ms
            }
        )
        
        return results
        
    except Exception as e:
        logger.error(
            f"Erreur lors de la recherche lexicale: {e}",
            extra={
                "request_id": request.request_id,
                "user_id": current_user["user_id"],
                "error_type": type(e).__name__
            },
            exc_info=True
        )
        
        # Métriques d'erreur
        background_tasks.add_task(
            _record_error_metrics,
            "lexical_search_error",
            current_user["user_id"],
            type(e).__name__
        )
        
        error_response = format_error_response(e, request.request_id)
        raise HTTPException(
            status_code=500,
            detail=error_response.dict()
        )

@router.post(
    "/search/validate",
    response_model=ValidationResponse,
    summary="Validation de requête",
    description="Valide une requête de recherche sans l'exécuter"
)
async def validate_query(
    validation_request: QueryValidationRequest,
    current_user: CurrentUser,
    context: RequestContext,
    _: RateLimited
) -> ValidationResponse:
    """
    Valide une requête de recherche sans l'exécuter.
    
    Utile pour vérifier la syntaxe et les paramètres avant soumission.
    """
    try:
        from ..utils.validators import RequestValidator
        
        validator = RequestValidator()
        request_id = str(uuid4())
        
        # Validation de la requête
        validation_errors = []
        warnings = []
        
        # Validation du texte de requête
        try:
            validator.validate_query_text(validation_request.query)
        except Exception as e:
            validation_errors.append({
                "field": "query",
                "message": str(e),
                "type": "invalid_query_text"
            })
        
        # Validation des filtres
        if validation_request.filters:
            try:
                validator.validate_filters(validation_request.filters)
            except Exception as e:
                validation_errors.append({
                    "field": "filters",
                    "message": str(e),
                    "type": "invalid_filters"
                })
        
        # Validation de la pagination
        try:
            validator.validate_pagination(
                validation_request.from_ or 0,
                validation_request.size or 10
            )
        except Exception as e:
            validation_errors.append({
                "field": "pagination",
                "message": str(e),
                "type": "invalid_pagination"
            })
        
        # Validation basée sur le rôle
        try:
            # Créer une requête temporaire pour validation
            temp_request = LexicalSearchRequest(
                query=validation_request.query,
                filters=validation_request.filters,
                from_=validation_request.from_,
                size=validation_request.size,
                options=validation_request.options
            )
            validator.validate_role_permissions(temp_request, current_user["role"])
        except Exception as e:
            warnings.append({
                "field": "permissions",
                "message": str(e),
                "type": "role_limitation"
            })
        
        # Estimation de la complexité
        complexity_score = _calculate_query_complexity(validation_request)
        if complexity_score > 80:
            warnings.append({
                "field": "complexity",
                "message": f"Requête complexe (score: {complexity_score}). Temps de réponse possiblement élevé.",
                "type": "high_complexity"
            })
        
        # Suggestions d'optimisation
        suggestions = _generate_optimization_suggestions(validation_request)
        
        return ValidationResponse(
            is_valid=len(validation_errors) == 0,
            errors=validation_errors,
            warnings=warnings,
            suggestions=suggestions,
            complexity_score=complexity_score,
            estimated_results=_estimate_result_count(validation_request),
            request_id=request_id,
            timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Erreur lors de la validation: {e}", exc_info=True)
        error_response = format_error_response(e)
        raise HTTPException(status_code=500, detail=error_response.dict())

@router.get(
    "/search/templates",
    response_model=TemplateListResponse,
    summary="Liste des templates",
    description="Récupère la liste des templates de requêtes disponibles"
)
async def get_query_templates(
    category: Optional[str] = None,
    current_user: CurrentUser = Depends(),
    _: RateLimited = Depends()
) -> TemplateListResponse:
    """
    Récupère la liste des templates de requêtes disponibles.
    
    Permet de filtrer par catégorie et inclut des exemples d'utilisation.
    """
    try:
        template_manager = QueryTemplateManager()
        
        # Récupération des templates
        if category:
            templates = template_manager.get_templates_by_category(category)
        else:
            templates = template_manager.get_all_templates()
        
        # Filtrage basé sur les permissions utilisateur
        user_role = current_user["role"]
        filtered_templates = []
        
        for template in templates:
            # Vérification des permissions
            if template.get("required_role"):
                required_roles = template["required_role"]
                if isinstance(required_roles, str):
                    required_roles = [required_roles]
                if user_role not in required_roles:
                    continue
            
            filtered_templates.append(template)
        
        # Catégories disponibles
        categories = list(set(t.get("category", "general") for t in filtered_templates))
        
        return TemplateListResponse(
            templates=filtered_templates,
            categories=categories,
            total_count=len(filtered_templates),
            user_role=user_role,
            request_id=str(uuid4()),
            timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des templates: {e}", exc_info=True)
        error_response = format_error_response(e)
        raise HTTPException(status_code=500, detail=error_response.dict())

@router.get(
    "/search/suggestions",
    summary="Auto-complétion",
    description="Fournit des suggestions d'auto-complétion pour les requêtes"
)
async def get_search_suggestions(
    q: str,
    limit: int = 10,
    current_user: CurrentUser = Depends(),
    lexical_engine: LexicalEngineService = Depends(),
    _: RateLimited = Depends()
):
    """
    Fournit des suggestions d'auto-complétion pour les requêtes.
    
    Utilise l'index Elasticsearch pour générer des suggestions pertinentes.
    """
    try:
        if len(q) < 2:
            return {"suggestions": [], "query": q}
        
        # Récupération des suggestions via le moteur lexical
        suggestions = await lexical_engine.get_suggestions(q, limit)
        
        return {
            "suggestions": suggestions,
            "query": q,
            "count": len(suggestions),
            "request_id": str(uuid4()),
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Erreur lors de la génération de suggestions: {e}", exc_info=True)
        error_response = format_error_response(e)
        raise HTTPException(status_code=500, detail=error_response.dict())

# ==================== ENDPOINTS DE MONITORING ====================

@router.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="Métriques du service",
    description="Récupère les métriques de performance du service"
)
async def get_metrics(
    current_user: CurrentUser,
    context: RequestContext
) -> MetricsResponse:
    """
    Récupère les métriques de performance du service.
    
    Accessible selon les permissions utilisateur.
    """
    try:
        # Vérification des permissions
        if current_user["role"] not in ["developer", "admin"]:
            raise HTTPException(
                status_code=403,
                detail="Accès aux métriques non autorisé pour ce rôle"
            )
        
        metrics_collector = MetricsCollector.get_instance()
        metrics_data = await metrics_collector.get_all_metrics()
        
        return MetricsResponse(
            metrics=metrics_data,
            timestamp=datetime.utcnow(),
            request_id=context["request_id"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des métriques: {e}", exc_info=True)
        error_response = format_error_response(e)
        raise HTTPException(status_code=500, detail=error_response.dict())

# ==================== ENDPOINTS DE SANTÉ ====================

@health_router.get(
    "/health",
    response_model=HealthResponse,
    summary="Vérification de santé",
    description="Vérifie l'état de santé du service et de ses dépendances"
)
async def health_check() -> HealthResponse:
    """
    Vérifie l'état de santé du service et de ses dépendances.
    
    Endpoint sans authentification pour les load balancers.
    """
    try:
        health_status = {}
        overall_status = "healthy"
        
        # Vérification Elasticsearch
        try:
            es_client = get_elasticsearch_client()
            es_health = await es_client.health_check()
            health_status["elasticsearch"] = {
                "status": "healthy" if es_health else "unhealthy",
                "details": es_health
            }
            if not es_health:
                overall_status = "degraded"
        except Exception as e:
            health_status["elasticsearch"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            overall_status = "unhealthy"
        
        # Vérification cache Redis
        try:
            from ..utils.cache import CacheManager
            cache_manager = CacheManager(get_settings())
            cache_health = await cache_manager.health_check()
            health_status["cache"] = {
                "status": "healthy" if cache_health else "degraded",
                "details": cache_health
            }
        except Exception as e:
            health_status["cache"] = {
                "status": "degraded",
                "error": str(e)
            }
            # Cache n'est pas critique
        
        # Métriques système
        health_status["system"] = {
            "status": "healthy",
            "timestamp": datetime.utcnow(),
            "uptime": "calculated_uptime"  # À implémenter
        }
        
        return HealthResponse(
            status=overall_status,
            timestamp=datetime.utcnow(),
            version="1.0.0",
            checks=health_status
        )
        
    except Exception as e:
        logger.error(f"Erreur lors du health check: {e}", exc_info=True)
        return HealthResponse(
            status="unhealthy",
            timestamp=datetime.utcnow(),
            version="1.0.0",
            checks={"error": str(e)}
        )

@health_router.get("/ping")
async def ping():
    """Ping simple pour vérification rapide."""
    return {"status": "ok", "timestamp": datetime.utcnow()}

@health_router.get("/ready")
async def readiness_check():
    """Vérification de préparation pour Kubernetes."""
    try:
        # Vérifications essentielles pour le démarrage
        es_client = get_elasticsearch_client()
        if not await es_client.health_check():
            raise HTTPException(status_code=503, detail="Elasticsearch not ready")
        
        return {"status": "ready", "timestamp": datetime.utcnow()}
        
    except Exception as e:
        logger.error(f"Service not ready: {e}")
        raise HTTPException(status_code=503, detail=f"Service not ready: {str(e)}")

@health_router.get("/live")
async def liveness_check():
    """Vérification de vivacité pour Kubernetes."""
    return {"status": "alive", "timestamp": datetime.utcnow()}

# ==================== FONCTIONS UTILITAIRES ====================

async def _record_search_metrics(
    request_id: str,
    user_id: str,
    query: str,
    results_count: int,
    total_results: int,
    search_time_ms: float
):
    """Enregistre les métriques de recherche de façon asynchrone."""
    try:
        metrics_collector = MetricsCollector.get_instance()
        
        # Métriques de base
        metrics_collector.increment_counter("search_requests_total")
        metrics_collector.record_histogram("search_duration_ms", search_time_ms)
        metrics_collector.record_histogram("search_results_count", results_count)
        
        # Métriques par utilisateur
        metrics_collector.increment_counter(
            "search_requests_by_user",
            {"user_id": user_id}
        )
        
        # Métriques de qualité
        if results_count == 0:
            metrics_collector.increment_counter("search_no_results_total")
        elif results_count < 5:
            metrics_collector.increment_counter("search_few_results_total")
        
        logger.debug(
            f"Métriques enregistrées pour la recherche {request_id}",
            extra={
                "request_id": request_id,
                "user_id": user_id,
                "results_count": results_count,
                "search_time_ms": search_time_ms
            }
        )
        
    except Exception as e:
        logger.warning(f"Erreur lors de l'enregistrement des métriques: {e}")

async def _record_error_metrics(error_type: str, user_id: str, error_class: str):
    """Enregistre les métriques d'erreur de façon asynchrone."""
    try:
        metrics_collector = MetricsCollector.get_instance()
        
        metrics_collector.increment_counter(
            "search_errors_total",
            {"error_type": error_type, "error_class": error_class}
        )
        
        metrics_collector.increment_counter(
            "search_errors_by_user",
            {"user_id": user_id, "error_type": error_type}
        )
        
    except Exception as e:
        logger.warning(f"Erreur lors de l'enregistrement des métriques d'erreur: {e}")

def _calculate_query_complexity(request: QueryValidationRequest) -> int:
    """Calcule un score de complexité pour une requête."""
    score = 0
    
    # Longueur de la requête
    score += min(len(request.query) // 10, 20)
    
    # Nombre de filtres
    if request.filters:
        score += len(request.filters) * 5
    
    # Taille demandée
    if request.size and request.size > 50:
        score += (request.size - 50) // 10
    
    # Offset élevé
    if request.from_ and request.from_ > 100:
        score += (request.from_ - 100) // 50
    
    # Options avancées
    if request.options:
        if hasattr(request.options, 'highlight') and request.options.highlight:
            score += 10
        if hasattr(request.options, 'aggregations') and request.options.aggregations:
            score += 15
    
    return min(score, 100)

def _estimate_result_count(request: QueryValidationRequest) -> Dict[str, int]:
    """Estime le nombre de résultats pour une requête."""
    # Estimation simple basée sur la longueur et la spécificité
    query_length = len(request.query.split())
    filter_count = len(request.filters) if request.filters else 0
    
    if query_length == 1 and filter_count == 0:
        estimated = {"min": 1000, "max": 10000}
    elif query_length <= 3 and filter_count <= 2:
        estimated = {"min": 100, "max": 1000}
    else:
        estimated = {"min": 10, "max": 100}
    
    return estimated

def _generate_optimization_suggestions(request: QueryValidationRequest) -> List[str]:
    """Génère des suggestions d'optimisation pour une requête."""
    suggestions = []
    
    # Requête trop large
    if len(request.query.split()) < 2:
        suggestions.append("Ajoutez plus de mots-clés pour affiner la recherche")
    
    # Taille de résultats trop élevée
    if request.size and request.size > 100:
        suggestions.append("Réduisez la taille des résultats pour améliorer les performances")
    
    # Offset élevé
    if request.from_ and request.from_ > 1000:
        suggestions.append("Utilisez la pagination avec search_after pour de meilleurs performances")
    
    # Pas de filtres
    if not request.filters and len(request.query.split()) < 3:
        suggestions.append("Ajoutez des filtres pour affiner la recherche")
    
    return suggestions

# ==================== ASSEMBLAGE DES ROUTERS ====================

def get_router() -> APIRouter:
    """Retourne le router principal configuré."""
    main_router = APIRouter()
    main_router.include_router(router)
    main_router.include_router(health_router)
    return main_router