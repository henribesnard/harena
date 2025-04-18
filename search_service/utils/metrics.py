"""
Utilitaires pour la collecte et l'enregistrement des métriques.

Ce module fournit des fonctionnalités pour enregistrer des métriques
sur les performances et l'utilisation du service de recherche.
"""
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional, List

# Import conditionnel de Prometheus si disponible
try:
    from prometheus_client import Counter, Histogram, Gauge
    PROMETHEUS_AVAILABLE = True
    
    # Définir les métriques Prometheus
    SEARCH_REQUESTS = Counter(
        'search_requests_total', 
        'Total number of search requests',
        ['user_id']
    )
    
    SEARCH_LATENCY = Histogram(
        'search_latency_milliseconds',
        'Search request latency in milliseconds',
        ['step']
    )
    
    RESULTS_COUNT = Histogram(
        'search_results_count',
        'Number of search results returned',
        ['user_id']
    )
    
    CACHE_HITS = Counter(
        'search_cache_hits_total',
        'Total number of cache hits'
    )
    
    CACHE_MISSES = Counter(
        'search_cache_misses_total',
        'Total number of cache misses'
    )
    
except ImportError:
    PROMETHEUS_AVAILABLE = False

logger = logging.getLogger(__name__)

# Base de données pour les métriques (pourrait être remplacé par une vraie DB)
_search_metrics_db = []
_max_metrics_entries = 10000  # Limite pour éviter la saturation de mémoire

async def record_search_metrics(
    user_id: int,
    query_text: str,
    results_count: int,
    execution_time_ms: float,
    detailed_timings: Optional[Dict[str, float]] = None,
    cache_hit: bool = False
) -> None:
    """
    Enregistre les métriques d'une recherche pour analyse ultérieure.
    
    Args:
        user_id: ID de l'utilisateur
        query_text: Texte de la requête
        results_count: Nombre de résultats retournés
        execution_time_ms: Temps d'exécution en millisecondes
        detailed_timings: Temps détaillés par étape
        cache_hit: Si la requête a été servie depuis le cache
    """
    # Créer l'entrée de métrique
    metric_entry = {
        "timestamp": datetime.now().isoformat(),
        "user_id": user_id,
        "query_text": query_text,
        "results_count": results_count,
        "execution_time_ms": execution_time_ms,
        "detailed_timings": detailed_timings or {},
        "cache_hit": cache_hit
    }
    
    # Enregistrer localement
    global _search_metrics_db
    _search_metrics_db.append(metric_entry)
    
    # Limiter la taille de la base de données en mémoire
    if len(_search_metrics_db) > _max_metrics_entries:
        _search_metrics_db = _search_metrics_db[-_max_metrics_entries:]
    
    # Enregistrer dans Prometheus si disponible
    if PROMETHEUS_AVAILABLE:
        try:
            SEARCH_REQUESTS.labels(user_id=user_id).inc()
            SEARCH_LATENCY.labels(step='total').observe(execution_time_ms)
            RESULTS_COUNT.labels(user_id=user_id).observe(results_count)
            
            if cache_hit:
                CACHE_HITS.inc()
            else:
                CACHE_MISSES.inc()
            
            # Enregistrer les temps détaillés
            if detailed_timings:
                for step, time_ms in detailed_timings.items():
                    if isinstance(time_ms, (int, float)):
                        SEARCH_LATENCY.labels(step=step).observe(time_ms)
        except Exception as e:
            logger.warning(f"Erreur lors de l'enregistrement des métriques Prometheus: {str(e)}")
    
    logger.info(f"Requête utilisateur {user_id}: '{query_text}' - {results_count} résultats en {execution_time_ms:.2f}ms")

async def get_search_metrics(
    user_id: Optional[int] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    limit: int = 100
) -> List[Dict[str, Any]]:
    """
    Récupère les métriques de recherche enregistrées, avec filtrage optionnel.
    
    Args:
        user_id: Filtrer par ID utilisateur
        start_time: Filtrer par heure de début
        end_time: Filtrer par heure de fin
        limit: Nombre maximum d'entrées à retourner
        
    Returns:
        Liste des entrées de métriques filtrées
    """
    # Filtrer les métriques
    filtered_metrics = _search_metrics_db.copy()
    
    if user_id is not None:
        filtered_metrics = [m for m in filtered_metrics if m["user_id"] == user_id]
    
    if start_time is not None:
        start_iso = start_time.isoformat()
        filtered_metrics = [m for m in filtered_metrics if m["timestamp"] >= start_iso]
    
    if end_time is not None:
        end_iso = end_time.isoformat()
        filtered_metrics = [m for m in filtered_metrics if m["timestamp"] <= end_iso]
    
    # Trier par timestamp décroissant (plus récent d'abord) et limiter
    sorted_metrics = sorted(filtered_metrics, key=lambda m: m["timestamp"], reverse=True)
    limited_metrics = sorted_metrics[:limit]
    
    return limited_metrics

def calculate_metrics_summary(metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calcule un résumé des métriques de recherche.
    
    Args:
        metrics: Liste d'entrées de métriques
        
    Returns:
        Résumé des métriques
    """
    if not metrics:
        return {
            "count": 0,
            "avg_execution_time_ms": 0,
            "avg_results_count": 0,
            "cache_hit_ratio": 0
        }
    
    count = len(metrics)
    total_execution_time = sum(m["execution_time_ms"] for m in metrics)
    total_results_count = sum(m["results_count"] for m in metrics)
    cache_hits = sum(1 for m in metrics if m.get("cache_hit", False))
    
    return {
        "count": count,
        "avg_execution_time_ms": total_execution_time / count if count > 0 else 0,
        "avg_results_count": total_results_count / count if count > 0 else 0,
        "cache_hit_ratio": cache_hits / count if count > 0 else 0,
        "start_time": min(m["timestamp"] for m in metrics),
        "end_time": max(m["timestamp"] for m in metrics)
    }