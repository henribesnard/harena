# search_service/api/routes.py
"""
Routes API REST simplifiées pour le Search Service
==================================================

Version simplifiée pour démarrage - à enrichir progressivement.
Health check géré par main.py pour éviter les conflits.
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

# === ROUTEUR PRINCIPAL ===
router = APIRouter(tags=["search"])

# === ENDPOINTS DE RECHERCHE ===

@router.get(
    "/health",
    summary="Santé du service de recherche",
    description="Vérification de l'état de santé du Search Service"
)
async def health_check():
    """Health check simple qui délègue au main.py"""
    try:
        # Import dynamique pour éviter les erreurs
        from search_service.core import core_manager
        
        # Vérifier si le core manager est initialisé
        if not core_manager.is_initialized():
            return {
                "status": "unhealthy",
                "service": "search-service",
                "version": "1.0.0",
                "error": "Core manager not initialized",
                "timestamp": datetime.now().isoformat()
            }
        
        # Health check basique
        return {
            "status": "healthy",
            "service": "search-service",
            "version": "1.0.0",
            "core_manager_initialized": True,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "unhealthy", 
            "service": "search-service",
            "version": "1.0.0",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


@router.get(
    "/",
    summary="Racine du service de recherche",
    description="Informations de base sur le Search Service"
)
async def search_root():
    """Endpoint racine du service de recherche"""
    return {
        "service": "search-service",
        "version": "1.0.0",
        "status": "running",
        "description": "Service de recherche lexicale pour données financières Harena",
        "endpoints": {
            "/health": "Vérification de santé",
            "/search": "Recherche lexicale (à venir)",
            "/validate": "Validation de requêtes (à venir)",
            "/templates": "Templates de requêtes (à venir)",
            "/status": "Statut simple",
            "/info": "Informations détaillées"
        },
        "timestamp": datetime.now().isoformat()
    }


@router.get(
    "/status",
    summary="Statut simple du service",
    description="Statut rapide sans vérifications approfondies"
)
async def simple_status():
    """Statut simple et rapide"""
    return {
        "service": "search-service",
        "status": "running", 
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "message": "Service démarré - Health check disponible sur /health"
    }


@router.get(
    "/info",
    summary="Informations sur le service",
    description="Informations détaillées sur le Search Service"
)
async def service_info():
    """Informations sur le service"""
    return {
        "service": {
            "name": "search-service",
            "version": "1.0.0",
            "description": "Service de recherche lexicale haute performance pour Harena",
            "status": "development_mode"
        },
        "features": {
            "implemented": [
                "Health checks (main.py)",
                "Status endpoint",
                "Service info",
                "Logging structuré"
            ],
            "planned": [
                "Recherche lexicale Elasticsearch",
                "Cache intelligent", 
                "Métriques détaillées",
                "Validation de requêtes",
                "Templates de requêtes",
                "Recherche hybride",
                "Agrégations financières"
            ]
        },
        "endpoints": {
            "GET /": "Informations racine",
            "GET /health": "Vérification de santé",
            "GET /status": "Statut simple",
            "GET /info": "Informations service",
            "POST /search": "Recherche lexicale (à venir)",
            "POST /validate": "Validation requêtes (à venir)",
            "GET /templates": "Templates requêtes (à venir)"
        },
        "environment": "development",
        "timestamp": datetime.now().isoformat()
    }


@router.post(
    "/search",
    summary="Recherche lexicale (placeholder)",
    description="Endpoint de recherche - sera implémenté avec Elasticsearch"
)
async def search_transactions(request: Request):
    """Endpoint de recherche lexicale (placeholder)"""
    return JSONResponse(
        content={
            "message": "Endpoint de recherche en développement",
            "service": "search-service",
            "version": "1.0.0",
            "note": "Sera implémenté avec Elasticsearch/Bonsai",
            "timestamp": datetime.now().isoformat()
        },
        status_code=501
    )


@router.post(
    "/validate",
    summary="Validation de requêtes (placeholder)",
    description="Endpoint de validation - sera implémenté"
)
async def validate_query(request: Request):
    """Endpoint de validation de requêtes (placeholder)"""
    return JSONResponse(
        content={
            "message": "Endpoint de validation en développement", 
            "service": "search-service",
            "version": "1.0.0",
            "note": "Validation Elasticsearch à venir",
            "timestamp": datetime.now().isoformat()
        },
        status_code=501
    )


@router.get(
    "/templates",
    summary="Templates de requêtes (placeholder)",
    description="Endpoint pour récupérer les templates de requêtes"
)
async def get_query_templates():
    """Endpoint pour récupérer les templates de requêtes (placeholder)"""
    return JSONResponse(
        content={
            "message": "Templates de requêtes en développement",
            "service": "search-service", 
            "version": "1.0.0",
            "note": "Templates financiers à venir",
            "timestamp": datetime.now().isoformat()
        },
        status_code=501
    )


@router.get(
    "/metrics",
    summary="Métriques du service (placeholder)",
    description="Métriques de performance - sera implémenté"
)
async def get_metrics():
    """Métriques du service (placeholder)"""
    return JSONResponse(
        content={
            "message": "Métriques en développement",
            "service": "search-service",
            "version": "1.0.0", 
            "note": "Métriques Elasticsearch à venir",
            "timestamp": datetime.now().isoformat()
        },
        status_code=501
    )


# === ENDPOINTS DE TEST ===

@router.get(
    "/test/ping",
    summary="Test de connectivité",
    description="Simple test pour vérifier que l'API répond"
)
async def ping():
    """Test de connectivité simple"""
    return {
        "message": "pong",
        "service": "search-service",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }


@router.get(
    "/test/error",
    summary="Test de gestion d'erreur",
    description="Endpoint de test pour vérifier la gestion d'erreur"
)
async def test_error():
    """Endpoint de test pour vérifier la gestion d'erreur"""
    raise HTTPException(
        status_code=500, 
        detail="Test error endpoint - gestion d'erreur fonctionnelle"
    )


# === ENDPOINTS POUR DÉVELOPPEMENT ===

@router.get(
    "/dev/config",
    summary="Configuration développement",
    description="Informations de configuration pour debugging"
)
async def dev_config():
    """Informations de configuration pour debugging"""
    import os
    return {
        "environment_variables": {
            "BONSAI_URL": "SET" if os.environ.get("BONSAI_URL") else "NOT SET",
            "ELASTICSEARCH_URL": "SET" if os.environ.get("ELASTICSEARCH_URL") else "NOT SET",
            "ENABLE_METRICS": os.environ.get("ENABLE_METRICS", "false"),
            "ELASTICSEARCH_INDEX": os.environ.get("ELASTICSEARCH_INDEX", "harena_transactions")
        },
        "service": "search-service",
        "version": "1.0.0",
        "note": "Endpoint de debugging - ne pas utiliser en production",
        "timestamp": datetime.now().isoformat()
    }


# === EXPORTS ===

__all__ = [
    "router",
    "health_check",
    "search_root",
    "simple_status", 
    "service_info",
    "search_transactions",
    "validate_query",
    "get_query_templates",
    "get_metrics",
    "ping",
    "test_error",
    "dev_config"
]

logger.info("Routes API simplifiées chargées - Health check inclus")