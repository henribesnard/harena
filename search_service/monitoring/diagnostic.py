"""
Endpoints de diagnostic et monitoring pour le service de recherche.

Ce module fournit des endpoints détaillés pour diagnostiquer l'état
des services Elasticsearch et Qdrant.
"""
import logging
import time
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

logger = logging.getLogger("search_service.diagnostic")

# Instance du router
diagnostic_router = APIRouter(prefix="/diagnostic", tags=["diagnostic"])

# Référence vers le monitor (sera injectée)
search_monitor = None


class ServiceDiagnostic(BaseModel):
    """Modèle pour le diagnostic d'un service."""
    service_name: str
    status: str
    is_healthy: bool
    last_check: datetime
    response_time_ms: Optional[float] = None
    consecutive_failures: int = 0
    error_message: Optional[str] = None
    
    # Métriques additionnelles
    connection_info: Dict[str, Any] = {}
    performance_metrics: Dict[str, Any] = {}


class SystemDiagnostic(BaseModel):
    """Modèle pour le diagnostic complet du système."""
    overall_status: str
    timestamp: datetime
    uptime_seconds: float
    
    # Services
    elasticsearch: ServiceDiagnostic
    qdrant: ServiceDiagnostic
    
    # Métriques globales
    search_metrics: Dict[str, Any]
    active_alerts: List[Dict[str, Any]]
    
    # Statistiques
    uptime_stats: Dict[str, float]


@diagnostic_router.get("/health", response_model=SystemDiagnostic)
async def get_system_health():
    """
    Endpoint de diagnostic complet du système de recherche.
    
    Retourne l'état détaillé de tous les composants avec métriques.
    """
    if not search_monitor:
        raise HTTPException(status_code=503, detail="Monitor not available")
    
    logger.info("🩺 Diagnostic complet du système demandé")
    
    try:
        # Forcer une vérification de santé
        await search_monitor._perform_health_checks()
        
        # Récupérer le résumé de santé
        health_summary = search_monitor.get_health_summary()
        
        # Créer le diagnostic détaillé
        system_diagnostic = SystemDiagnostic(
            overall_status=health_summary["overall_status"],
            timestamp=datetime.now(),
            uptime_seconds=time.time() - search_monitor._start_time if hasattr(search_monitor, '_start_time') else 0,
            
            elasticsearch=ServiceDiagnostic(
                service_name="elasticsearch",
                status=health_summary["elasticsearch"]["status"],
                is_healthy=health_summary["elasticsearch"]["status"] == "healthy",
                last_check=datetime.fromisoformat(health_summary["elasticsearch"]["last_check"]),
                response_time_ms=health_summary["elasticsearch"]["response_time_ms"],
                consecutive_failures=health_summary["elasticsearch"]["consecutive_failures"],
                error_message=health_summary["elasticsearch"]["error_message"],
                connection_info=await _get_elasticsearch_info(),
                performance_metrics=await _get_elasticsearch_metrics()
            ),
            
            qdrant=ServiceDiagnostic(
                service_name="qdrant",
                status=health_summary["qdrant"]["status"],
                is_healthy=health_summary["qdrant"]["status"] == "healthy",
                last_check=datetime.fromisoformat(health_summary["qdrant"]["last_check"]),
                response_time_ms=health_summary["qdrant"]["response_time_ms"],
                consecutive_failures=health_summary["qdrant"]["consecutive_failures"],
                error_message=health_summary["qdrant"]["error_message"],
                connection_info=await _get_qdrant_info(),
                performance_metrics=await _get_qdrant_metrics()
            ),
            
            search_metrics=health_summary["search_metrics"],
            active_alerts=health_summary["active_alerts"],
            uptime_stats=search_monitor.get_uptime_stats()
        )
        
        logger.info(f"✅ Diagnostic généré: {system_diagnostic.overall_status}")
        return system_diagnostic
        
    except Exception as e:
        logger.error(f"❌ Erreur lors du diagnostic: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Diagnostic failed: {str(e)}")


@diagnostic_router.get("/elasticsearch")
async def get_elasticsearch_diagnostic():
    """Diagnostic détaillé d'Elasticsearch."""
    logger.info("🔍 Diagnostic Elasticsearch demandé")
    
    if not search_monitor or not search_monitor.elasticsearch_client:
        raise HTTPException(status_code=503, detail="Elasticsearch client not available")
    
    try:
        client = search_monitor.elasticsearch_client
        
        # Test de connexion
        start_time = time.time()
        is_healthy = await client.is_healthy()
        response_time = (time.time() - start_time) * 1000
        
        # Informations détaillées
        diagnostic = {
            "service": "elasticsearch",
            "timestamp": datetime.now().isoformat(),
            "is_healthy": is_healthy,
            "response_time_ms": response_time,
            "connection_info": await _get_elasticsearch_info(),
            "cluster_health": await _get_elasticsearch_cluster_health(),
            "index_info": await _get_elasticsearch_index_info(),
            "performance_metrics": await _get_elasticsearch_metrics()
        }
        
        logger.info(f"✅ Diagnostic Elasticsearch: {'healthy' if is_healthy else 'unhealthy'}")
        return diagnostic
        
    except Exception as e:
        logger.error(f"❌ Erreur diagnostic Elasticsearch: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@diagnostic_router.get("/qdrant")
async def get_qdrant_diagnostic():
    """Diagnostic détaillé de Qdrant."""
    logger.info("🎯 Diagnostic Qdrant demandé")
    
    if not search_monitor or not search_monitor.qdrant_client:
        raise HTTPException(status_code=503, detail="Qdrant client not available")
    
    try:
        client = search_monitor.qdrant_client
        
        # Test de connexion
        start_time = time.time()
        is_healthy = await client.is_healthy()
        response_time = (time.time() - start_time) * 1000
        
        # Informations détaillées
        diagnostic = {
            "service": "qdrant",
            "timestamp": datetime.now().isoformat(),
            "is_healthy": is_healthy,
            "response_time_ms": response_time,
            "connection_info": await _get_qdrant_info(),
            "collection_info": await _get_qdrant_collection_info(),
            "performance_metrics": await _get_qdrant_metrics()
        }
        
        logger.info(f"✅ Diagnostic Qdrant: {'healthy' if is_healthy else 'unhealthy'}")
        return diagnostic
        
    except Exception as e:
        logger.error(f"❌ Erreur diagnostic Qdrant: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@diagnostic_router.post("/test-search")
async def test_search_functionality(
    test_query: str = Query("transaction test", description="Requête de test"),
    test_user_id: int = Query(1, description="ID utilisateur pour le test")
):
    """
    Test complet de la fonctionnalité de recherche.
    
    Exécute une recherche de test pour valider que tous les composants fonctionnent.
    """
    logger.info(f"🧪 Test de recherche: '{test_query}' pour user_id={test_user_id}")
    
    test_results = {
        "test_query": test_query,
        "test_user_id": test_user_id,
        "timestamp": datetime.now().isoformat(),
        "results": {}
    }
    
    # Test Elasticsearch
    if search_monitor and search_monitor.elasticsearch_client:
        try:
            logger.info("🔍 Test recherche Elasticsearch...")
            start_time = time.time()
            
            # Construire une requête simple
            es_query = {
                "bool": {
                    "must": [
                        {"match": {"searchable_text": test_query}},
                        {"term": {"user_id": test_user_id}}
                    ]
                }
            }
            
            es_results = await search_monitor.elasticsearch_client.search(
                user_id=test_user_id,
                query=es_query,
                limit=5
            )
            
            es_time = (time.time() - start_time) * 1000
            
            test_results["results"]["elasticsearch"] = {
                "success": True,
                "response_time_ms": es_time,
                "results_count": len(es_results),
                "error": None
            }
            
            logger.info(f"✅ Elasticsearch test: {len(es_results)} résultats en {es_time:.2f}ms")
            
        except Exception as e:
            logger.error(f"❌ Elasticsearch test failed: {e}")
            test_results["results"]["elasticsearch"] = {
                "success": False,
                "response_time_ms": None,
                "results_count": 0,
                "error": str(e)
            }
    else:
        test_results["results"]["elasticsearch"] = {
            "success": False,
            "error": "Client not available"
        }
    
    # Test Qdrant (nécessite un embedding)
    if search_monitor and search_monitor.qdrant_client:
        try:
            logger.info("🎯 Test recherche Qdrant...")
            start_time = time.time()
            
            # Créer un vecteur de test (zéros pour simplicité)
            test_vector = [0.0] * 1536  # Dimension OpenAI
            
            qdrant_results = await search_monitor.qdrant_client.search(
                query_vector=test_vector,
                user_id=test_user_id,
                limit=5
            )
            
            qdrant_time = (time.time() - start_time) * 1000
            
            test_results["results"]["qdrant"] = {
                "success": True,
                "response_time_ms": qdrant_time,
                "results_count": len(qdrant_results),
                "error": None
            }
            
            logger.info(f"✅ Qdrant test: {len(qdrant_results)} résultats en {qdrant_time:.2f}ms")
            
        except Exception as e:
            logger.error(f"❌ Qdrant test failed: {e}")
            test_results["results"]["qdrant"] = {
                "success": False,
                "response_time_ms": None,
                "results_count": 0,
                "error": str(e)
            }
    else:
        test_results["results"]["qdrant"] = {
            "success": False,
            "error": "Client not available"
        }
    
    # Résumé du test
    successful_tests = sum(1 for result in test_results["results"].values() 
                          if result.get("success", False))
    total_tests = len(test_results["results"])
    
    test_results["summary"] = {
        "successful_tests": successful_tests,
        "total_tests": total_tests,
        "all_passed": successful_tests == total_tests
    }
    
    status_icon = "✅" if test_results["summary"]["all_passed"] else "❌"
    logger.info(f"{status_icon} Test search: {successful_tests}/{total_tests} composants fonctionnels")
    
    return test_results


# Fonctions utilitaires pour récupérer les informations détaillées

async def _get_elasticsearch_info() -> Dict[str, Any]:
    """Récupère les informations de connexion Elasticsearch."""
    if not search_monitor or not search_monitor.elasticsearch_client:
        return {"error": "Client not available"}
    
    try:
        client = search_monitor.elasticsearch_client.client
        if not client:
            return {"error": "Client not initialized"}
        
        info = await client.info()
        return {
            "cluster_name": info.get("cluster_name"),
            "version": info.get("version", {}).get("number"),
            "lucene_version": info.get("version", {}).get("lucene_version"),
            "cluster_uuid": info.get("cluster_uuid")
        }
    except Exception as e:
        return {"error": str(e)}


async def _get_elasticsearch_cluster_health() -> Dict[str, Any]:
    """Récupère la santé du cluster Elasticsearch."""
    if not search_monitor or not search_monitor.elasticsearch_client:
        return {"error": "Client not available"}
    
    try:
        client = search_monitor.elasticsearch_client.client
        health = await client.cluster.health()
        return {
            "status": health.get("status"),
            "number_of_nodes": health.get("number_of_nodes"),
            "number_of_data_nodes": health.get("number_of_data_nodes"),
            "active_primary_shards": health.get("active_primary_shards"),
            "active_shards": health.get("active_shards"),
            "relocating_shards": health.get("relocating_shards"),
            "initializing_shards": health.get("initializing_shards"),
            "unassigned_shards": health.get("unassigned_shards")
        }
    except Exception as e:
        return {"error": str(e)}


async def _get_elasticsearch_index_info() -> Dict[str, Any]:
    """Récupère les informations sur l'index Elasticsearch."""
    if not search_monitor or not search_monitor.elasticsearch_client:
        return {"error": "Client not available"}
    
    try:
        client = search_monitor.elasticsearch_client.client
        index_name = search_monitor.elasticsearch_client.index_name
        
        # Vérifier si l'index existe
        exists = await client.indices.exists(index=index_name)
        if not exists:
            return {"error": f"Index {index_name} does not exist"}
        
        # Statistiques de l'index
        stats = await client.indices.stats(index=index_name)
        index_stats = stats.get("indices", {}).get(index_name, {})
        
        return {
            "index_name": index_name,
            "exists": exists,
            "document_count": index_stats.get("total", {}).get("docs", {}).get("count", 0),
            "store_size_bytes": index_stats.get("total", {}).get("store", {}).get("size_in_bytes", 0),
            "primary_shards": index_stats.get("primaries", {}).get("docs", {}).get("count", 0)
        }
    except Exception as e:
        return {"error": str(e)}


async def _get_elasticsearch_metrics() -> Dict[str, Any]:
    """Récupère les métriques de performance Elasticsearch."""
    # Placeholder pour métriques de performance
    return {
        "avg_query_time_ms": 0,
        "cache_hit_ratio": 0,
        "indexing_rate": 0
    }


async def _get_qdrant_info() -> Dict[str, Any]:
    """Récupère les informations de connexion Qdrant."""
    if not search_monitor or not search_monitor.qdrant_client:
        return {"error": "Client not available"}
    
    try:
        client = search_monitor.qdrant_client.client
        if not client:
            return {"error": "Client not initialized"}
        
        collections = await client.get_collections()
        return {
            "total_collections": len(collections.collections),
            "collections": [col.name for col in collections.collections],
            "target_collection": search_monitor.qdrant_client.collection_name
        }
    except Exception as e:
        return {"error": str(e)}


async def _get_qdrant_collection_info() -> Dict[str, Any]:
    """Récupère les informations sur la collection Qdrant."""
    if not search_monitor or not search_monitor.qdrant_client:
        return {"error": "Client not available"}
    
    try:
        client = search_monitor.qdrant_client.client
        collection_name = search_monitor.qdrant_client.collection_name
        
        collection_info = await client.get_collection(collection_name)
        
        return {
            "collection_name": collection_name,
            "points_count": collection_info.points_count,
            "vectors_count": getattr(collection_info, 'vectors_count', 'unknown'),
            "indexed_vectors_count": getattr(collection_info, 'indexed_vectors_count', 'unknown'),
            "config": {
                "vector_size": collection_info.config.params.vectors.size,
                "distance": collection_info.config.params.vectors.distance.value
            } if collection_info.config else {}
        }
    except Exception as e:
        return {"error": str(e)}


async def _get_qdrant_metrics() -> Dict[str, Any]:
    """Récupère les métriques de performance Qdrant."""
    # Placeholder pour métriques de performance
    return {
        "avg_search_time_ms": 0,
        "index_efficiency": 0
    }


def setup_diagnostic_routes(app, monitor):
    """Configure les routes de diagnostic."""
    global search_monitor
    search_monitor = monitor
    
    app.include_router(diagnostic_router)
    logger.info("🔧 Routes de diagnostic configurées")