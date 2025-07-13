"""
Utilitaires système et performance pour le Search Service
========================================================

Module centralisé contenant toutes les fonctions utilitaires système :
- Métriques système (mémoire, CPU, performance)
- Gestion du cache et nettoyage
- Santé des composants
- Résumés de performance
- Utilitaires de monitoring

Architecture :
    Routes API → system_utils → metrics_collector / cache_manager / core_components
"""

import logging
import time
import psutil
from typing import Dict, Any
from datetime import datetime, timedelta
from enum import Enum

# Imports locaux - CORRECTION: Importer cache_manager directement
from .metrics import metrics_collector, alert_manager
from .cache import cache_manager  # ← CORRECTION: au lieu de get_cache_manager


logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Statuts de santé possibles"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class ComponentType(str, Enum):
    """Types de composants surveillés"""
    ELASTICSEARCH = "elasticsearch"
    CACHE = "cache"
    METRICS = "metrics"
    CORE_ENGINE = "core_engine"
    API = "api"
    SYSTEM = "system"


# === MÉTRIQUES SYSTÈME ===

def get_system_metrics() -> Dict[str, Any]:
    """
    Retourne les métriques système complètes du Search Service
    
    Returns:
        Dict contenant les métriques système actuelles
    """
    try:
        process = psutil.Process()
        
        # Métriques mémoire
        memory_info = process.memory_info()
        memory_percent = process.memory_percent()
        
        # Métriques CPU
        cpu_percent = process.cpu_percent()
        cpu_times = process.cpu_times()
        
        # Métriques I/O si disponibles
        io_info = {}
        try:
            io_counters = process.io_counters()
            io_info = {
                "read_bytes": io_counters.read_bytes,
                "write_bytes": io_counters.write_bytes,
                "read_count": io_counters.read_count,
                "write_count": io_counters.write_count
            }
        except AttributeError:
            # I/O counters non disponibles sur certaines plateformes
            pass
        
        # Métriques réseau des connexions
        connections = len(process.connections())
        
        # Uptime du service
        start_time = getattr(metrics_collector, '_start_time', datetime.now())
        uptime_seconds = (datetime.now() - start_time).total_seconds()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": uptime_seconds,
            "process": {
                "pid": process.pid,
                "status": process.status(),
                "num_threads": process.num_threads()
            },
            "memory": {
                "rss_bytes": memory_info.rss,
                "vms_bytes": memory_info.vms,
                "rss_mb": memory_info.rss / (1024 * 1024),
                "percent": memory_percent
            },
            "cpu": {
                "percent": cpu_percent,
                "user_time": cpu_times.user,
                "system_time": cpu_times.system
            },
            "io": io_info,
            "network": {
                "connections": connections
            },
            "metrics_system": {
                "total_metrics": len(getattr(metrics_collector, '_definitions', {})),
                "active_alerts": len(getattr(alert_manager, 'active_alerts', {})),
                "collectors_active": True
            }
        }
        
    except Exception as e:
        logger.error(f"Erreur lors de la collecte des métriques système: {e}")
        return {
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "status": "error"
        }


def get_performance_summary(hours: int = 1) -> Dict[str, Any]:
    """
    Génère un résumé de performance détaillé sur une période donnée
    
    Args:
        hours: Nombre d'heures à analyser (défaut: 1)
        
    Returns:
        Dict contenant le résumé de performance
    """
    try:
        since = datetime.now() - timedelta(hours=hours)
        
        # Métriques clés de performance
        key_metrics = [
            "lexical_search_duration_ms",
            "elasticsearch_search_duration_ms", 
            "api_request_duration_ms",
            "query_execution_duration_ms",
            "result_processing_duration_ms"
        ]
        
        # Métriques de qualité
        quality_metrics = [
            "lexical_cache_hit_rate",
            "lexical_search_quality_score",
            "elasticsearch_cache_hit_rate"
        ]
        
        # Métriques de volume
        volume_metrics = [
            "api_request_count",
            "elasticsearch_search_count",
            "api_error_count"
        ]
        
        summary = {
            "period_hours": hours,
            "generated_at": datetime.now().isoformat(),
            "performance_metrics": {},
            "quality_metrics": {},
            "volume_metrics": {},
            "system_health": {}
        }
        
        # Collecter les statistiques de performance
        for metric_name in key_metrics:
            try:
                stats = metrics_collector.get_metric_stats(metric_name, since)
                summary["performance_metrics"][metric_name] = {
                    "avg_ms": round(stats.get("avg", 0), 2),
                    "min_ms": round(stats.get("min", 0), 2),
                    "max_ms": round(stats.get("max", 0), 2),
                    "p95_ms": round(stats.get("p95", 0), 2) if stats.get("p95") else None,
                    "count": stats.get("count", 0)
                }
            except Exception as e:
                logger.warning(f"Erreur collecte métrique {metric_name}: {e}")
        
        # Collecter les métriques de qualité
        for metric_name in quality_metrics:
            try:
                current_value = metrics_collector.get_current_value(metric_name, 0)
                stats = metrics_collector.get_metric_stats(metric_name, since)
                summary["quality_metrics"][metric_name] = {
                    "current": round(current_value, 2),
                    "avg": round(stats.get("avg", 0), 2),
                    "trend": _calculate_trend(stats)
                }
            except Exception as e:
                logger.warning(f"Erreur collecte qualité {metric_name}: {e}")
        
        # Collecter les métriques de volume
        for metric_name in volume_metrics:
            try:
                stats = metrics_collector.get_metric_stats(metric_name, since)
                summary["volume_metrics"][metric_name] = {
                    "total": stats.get("sum", 0),
                    "rate_per_hour": stats.get("sum", 0) / hours if hours > 0 else 0
                }
            except Exception as e:
                logger.warning(f"Erreur collecte volume {metric_name}: {e}")
        
        # Santé système
        system_metrics = get_system_metrics()
        summary["system_health"] = {
            "memory_usage_mb": system_metrics.get("memory", {}).get("rss_mb", 0),
            "cpu_percent": system_metrics.get("cpu", {}).get("percent", 0),
            "uptime_hours": system_metrics.get("uptime_seconds", 0) / 3600,
            "active_alerts": system_metrics.get("metrics_system", {}).get("active_alerts", 0)
        }
        
        # Calcul du score global de performance
        summary["overall_performance_score"] = _calculate_performance_score(summary)
        
        return summary
        
    except Exception as e:
        logger.error(f"Erreur lors de la génération du résumé de performance: {e}")
        return {
            "period_hours": hours,
            "generated_at": datetime.now().isoformat(),
            "error": str(e),
            "status": "error"
        }


def get_utils_performance() -> Dict[str, Any]:
    """
    Retourne les performances spécifiques des utilitaires (cache, métriques, etc.)
    
    Returns:
        Dict contenant les performances des utilitaires
    """
    try:
        performance_data = {
            "timestamp": datetime.now().isoformat(),
            "cache_performance": {},
            "metrics_performance": {},
            "system_performance": {}
        }
        
        # Performance du cache - CORRECTION: utiliser cache_manager directement
        try:
            cache_stats = cache_manager.get_global_stats()
            
            # Adapter la structure selon ce qui est disponible dans cache_manager
            performance_data["cache_performance"] = {
                "total_caches": cache_stats.get("total_caches", 0),
                "caches_info": cache_stats.get("caches", {}),
                "status": "operational"
            }
        except Exception as e:
            logger.warning(f"Erreur collecte performance cache: {e}")
            performance_data["cache_performance"]["error"] = str(e)
        
        # Performance des métriques
        try:
            metrics_stats = {
                "collection_active": getattr(metrics_collector, '_system_metrics_enabled', False),
                "total_metrics_defined": len(getattr(metrics_collector, '_definitions', {})),
                "total_samples": sum(len(samples) for samples in getattr(metrics_collector, '_metrics', {}).values()),
                "active_alerts": len(getattr(alert_manager, 'active_alerts', {})),
                "collection_interval_seconds": getattr(metrics_collector, '_system_metrics_interval', 30)
            }
            
            performance_data["metrics_performance"] = metrics_stats
        except Exception as e:
            logger.warning(f"Erreur collecte performance métriques: {e}")
            performance_data["metrics_performance"]["error"] = str(e)
        
        # Performance système globale
        try:
            system_perf = get_system_metrics()
            performance_data["system_performance"] = {
                "memory_efficiency": _calculate_memory_efficiency(system_perf),
                "cpu_efficiency": _calculate_cpu_efficiency(system_perf),
                "io_efficiency": _calculate_io_efficiency(system_perf)
            }
        except Exception as e:
            logger.warning(f"Erreur calcul performance système: {e}")
            performance_data["system_performance"]["error"] = str(e)
        
        return performance_data
        
    except Exception as e:
        logger.error(f"Erreur lors de la collecte des performances utilitaires: {e}")
        return {
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "status": "error"
        }


def cleanup_old_metrics(hours: int = 24) -> Dict[str, Any]:
    """
    Nettoie les métriques anciennes et retourne un rapport
    
    Args:
        hours: Nombre d'heures à conserver (défaut: 24)
        
    Returns:
        Dict contenant le rapport de nettoyage
    """
    try:
        start_time = time.time()
        
        # Nettoyage des métriques via le collecteur
        original_count = sum(len(samples) for samples in getattr(metrics_collector, '_metrics', {}).values())
        
        metrics_collector.cleanup_old_samples(hours)
        
        final_count = sum(len(samples) for samples in getattr(metrics_collector, '_metrics', {}).values())
        cleaned_count = original_count - final_count
        
        # Nettoyage du cache si nécessaire - CORRECTION: utiliser cache_manager directement
        cache_cleaned = 0
        try:
            cache_stats = cache_manager.cleanup_all_caches()
            # Calculer le total nettoyé
            cache_cleaned = sum(
                sum(cache_data.values()) if isinstance(cache_data, dict) else 0
                for cache_data in cache_stats.values()
            )
        except Exception as e:
            logger.warning(f"Erreur nettoyage cache: {e}")
        
        # Nettoyage de l'historique des alertes
        alert_cleaned = 0
        try:
            if hasattr(alert_manager, 'alert_history'):
                original_alerts = len(alert_manager.alert_history)
                # Garder seulement les alertes des dernières 48h
                cutoff_time = datetime.now() - timedelta(hours=min(hours * 2, 48))
                alert_manager.alert_history = [
                    alert for alert in alert_manager.alert_history
                    if alert.timestamp > cutoff_time
                ]
                alert_cleaned = original_alerts - len(alert_manager.alert_history)
        except Exception as e:
            logger.warning(f"Erreur nettoyage alertes: {e}")
        
        cleanup_duration = (time.time() - start_time) * 1000
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "duration_ms": round(cleanup_duration, 2),
            "retention_hours": hours,
            "metrics_cleaned": {
                "original_count": original_count,
                "final_count": final_count,
                "cleaned_count": cleaned_count
            },
            "cache_cleaned": cache_cleaned,
            "alerts_cleaned": alert_cleaned,
            "total_items_cleaned": cleaned_count + cache_cleaned + alert_cleaned,
            "success": True
        }
        
        logger.info(f"Nettoyage terminé: {report['total_items_cleaned']} éléments supprimés en {cleanup_duration:.1f}ms")
        
        return report
        
    except Exception as e:
        logger.error(f"Erreur lors du nettoyage: {e}")
        return {
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "success": False
        }


def get_utils_health() -> Dict[str, Any]:
    """
    Vérifie la santé de tous les utilitaires du Search Service
    
    Returns:
        Dict contenant l'état de santé détaillé
    """
    try:
        health_report = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": HealthStatus.HEALTHY,
            "components": {},
            "summary": {}
        }
        
        components_status = []
        
        # Santé du système de métriques
        try:
            metrics_health = _check_metrics_health()
            health_report["components"]["metrics"] = metrics_health
            components_status.append(metrics_health["status"])
        except Exception as e:
            health_report["components"]["metrics"] = {
                "status": HealthStatus.UNHEALTHY,
                "error": str(e)
            }
            components_status.append(HealthStatus.UNHEALTHY)
        
        # Santé du cache
        try:
            cache_health = _check_cache_health()
            health_report["components"]["cache"] = cache_health
            components_status.append(cache_health["status"])
        except Exception as e:
            health_report["components"]["cache"] = {
                "status": HealthStatus.UNHEALTHY,
                "error": str(e)
            }
            components_status.append(HealthStatus.UNHEALTHY)
        
        # Santé du système
        try:
            system_health = _check_system_health()
            health_report["components"]["system"] = system_health
            components_status.append(system_health["status"])
        except Exception as e:
            health_report["components"]["system"] = {
                "status": HealthStatus.UNHEALTHY,
                "error": str(e)
            }
            components_status.append(HealthStatus.UNHEALTHY)
        
        # Déterminer le statut global
        if HealthStatus.UNHEALTHY in components_status:
            health_report["overall_status"] = HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in components_status:
            health_report["overall_status"] = HealthStatus.DEGRADED
        else:
            health_report["overall_status"] = HealthStatus.HEALTHY
        
        # Résumé
        health_report["summary"] = {
            "total_components": len(components_status),
            "healthy_components": components_status.count(HealthStatus.HEALTHY),
            "degraded_components": components_status.count(HealthStatus.DEGRADED),
            "unhealthy_components": components_status.count(HealthStatus.UNHEALTHY)
        }
        
        return health_report
        
    except Exception as e:
        logger.error(f"Erreur lors de la vérification de santé: {e}")
        return {
            "timestamp": datetime.now().isoformat(),
            "overall_status": HealthStatus.UNKNOWN,
            "error": str(e)
        }


# === FONCTIONS PRIVÉES ===

def _calculate_trend(stats: Dict[str, Any]) -> str:
    """Calcule la tendance basée sur les statistiques"""
    if not stats or stats.get("count", 0) < 2:
        return "insufficient_data"
    
    avg = stats.get("avg", 0)
    last = stats.get("last", 0)
    
    if abs(last - avg) < avg * 0.1:  # Variation < 10%
        return "stable"
    elif last > avg:
        return "increasing"
    else:
        return "decreasing"


def _calculate_performance_score(summary: Dict[str, Any]) -> float:
    """Calcule un score global de performance (0-100)"""
    try:
        score_components = []
        
        # Score basé sur les temps de réponse
        api_avg = summary.get("performance_metrics", {}).get("api_request_duration_ms", {}).get("avg_ms", 1000)
        if api_avg < 100:
            score_components.append(90)
        elif api_avg < 500:
            score_components.append(70)
        else:
            score_components.append(30)
        
        # Score basé sur le cache hit rate
        cache_hit_rate = summary.get("quality_metrics", {}).get("lexical_cache_hit_rate", {}).get("current", 0)
        score_components.append(min(cache_hit_rate, 100))
        
        # Score basé sur la charge système
        cpu_percent = summary.get("system_health", {}).get("cpu_percent", 100)
        if cpu_percent < 70:
            score_components.append(90)
        elif cpu_percent < 85:
            score_components.append(60)
        else:
            score_components.append(20)
        
        # Score basé sur les erreurs
        errors = summary.get("volume_metrics", {}).get("api_error_count", {}).get("total", 0)
        requests = summary.get("volume_metrics", {}).get("api_request_count", {}).get("total", 1)
        error_rate = (errors / requests) * 100 if requests > 0 else 0
        
        if error_rate < 1:
            score_components.append(100)
        elif error_rate < 5:
            score_components.append(70)
        else:
            score_components.append(30)
        
        return round(sum(score_components) / len(score_components), 1)
        
    except Exception:
        return 50.0  # Score neutre en cas d'erreur


def _calculate_memory_efficiency(system_metrics: Dict[str, Any]) -> float:
    """Calcule l'efficacité mémoire (0-100)"""
    try:
        memory_mb = system_metrics.get("memory", {}).get("rss_mb", 0)
        
        # Considère < 512MB comme excellent, > 2GB comme problématique
        if memory_mb < 512:
            return 100.0
        elif memory_mb < 1024:
            return 80.0
        elif memory_mb < 2048:
            return 60.0
        else:
            return 30.0
    except Exception:
        return 50.0


def _calculate_cpu_efficiency(system_metrics: Dict[str, Any]) -> float:
    """Calcule l'efficacité CPU (0-100)"""
    try:
        cpu_percent = system_metrics.get("cpu", {}).get("percent", 100)
        
        if cpu_percent < 30:
            return 100.0
        elif cpu_percent < 60:
            return 80.0
        elif cpu_percent < 85:
            return 60.0
        else:
            return 30.0
    except Exception:
        return 50.0


def _calculate_io_efficiency(system_metrics: Dict[str, Any]) -> float:
    """Calcule l'efficacité I/O (0-100)"""
    try:
        io_info = system_metrics.get("io", {})
        if not io_info:
            return 100.0  # Pas de problème I/O détecté
        
        # Analyse basique basée sur le volume I/O
        read_bytes = io_info.get("read_bytes", 0)
        write_bytes = io_info.get("write_bytes", 0)
        total_io = read_bytes + write_bytes
        
        # Seuils arbitraires (à ajuster selon le contexte)
        if total_io < 100 * 1024 * 1024:  # < 100MB
            return 100.0
        elif total_io < 1024 * 1024 * 1024:  # < 1GB
            return 80.0
        else:
            return 60.0
    except Exception:
        return 100.0


def _check_metrics_health() -> Dict[str, Any]:
    """Vérifie la santé du système de métriques"""
    try:
        # Vérifier que le collecteur est actif
        is_active = getattr(metrics_collector, '_system_metrics_enabled', False)
        metrics_count = len(getattr(metrics_collector, '_definitions', {}))
        samples_count = sum(len(samples) for samples in getattr(metrics_collector, '_metrics', {}).values())
        
        if not is_active:
            status = HealthStatus.UNHEALTHY
            message = "Collecteur de métriques inactif"
        elif metrics_count == 0:
            status = HealthStatus.DEGRADED
            message = "Aucune métrique définie"
        elif samples_count == 0:
            status = HealthStatus.DEGRADED
            message = "Aucun échantillon collecté"
        else:
            status = HealthStatus.HEALTHY
            message = "Système de métriques opérationnel"
        
        return {
            "status": status,
            "message": message,
            "details": {
                "collector_active": is_active,
                "metrics_defined": metrics_count,
                "samples_collected": samples_count,
                "alerts_active": len(getattr(alert_manager, 'active_alerts', {}))
            }
        }
    except Exception as e:
        return {
            "status": HealthStatus.UNHEALTHY,
            "message": f"Erreur vérification métriques: {e}",
            "error": str(e)
        }


def _check_cache_health() -> Dict[str, Any]:
    """Vérifie la santé du système de cache - CORRECTION: utiliser cache_manager directement"""
    try:
        stats = cache_manager.get_global_stats()
        
        # Évaluer la santé du cache basée sur les stats globales
        total_caches = stats.get("total_caches", 0)
        caches_info = stats.get("caches", {})
        
        if total_caches == 0:
            status = HealthStatus.DEGRADED
            message = "Aucun cache configuré"
        elif caches_info:
            status = HealthStatus.HEALTHY
            message = "Cache opérationnel"
        else:
            status = HealthStatus.DEGRADED
            message = "Cache en état dégradé"
        
        return {
            "status": status,
            "message": message,
            "details": {
                "total_caches": total_caches,
                "caches_info": caches_info
            }
        }
    except Exception as e:
        return {
            "status": HealthStatus.UNHEALTHY,
            "message": f"Erreur vérification cache: {e}",
            "error": str(e)
        }


def _check_system_health() -> Dict[str, Any]:
    """Vérifie la santé du système"""
    try:
        system_metrics = get_system_metrics()
        
        memory_mb = system_metrics.get("memory", {}).get("rss_mb", 0)
        cpu_percent = system_metrics.get("cpu", {}).get("percent", 0)
        
        # Évaluer la santé système
        issues = []
        
        if memory_mb > 2048:  # > 2GB
            issues.append("Utilisation mémoire élevée")
        if cpu_percent > 85:  # > 85%
            issues.append("Utilisation CPU élevée")
        
        if not issues:
            status = HealthStatus.HEALTHY
            message = "Système en bon état"
        elif len(issues) == 1:
            status = HealthStatus.DEGRADED
            message = f"Problème détecté: {issues[0]}"
        else:
            status = HealthStatus.UNHEALTHY
            message = f"Problèmes multiples: {', '.join(issues)}"
        
        return {
            "status": status,
            "message": message,
            "details": {
                "memory_mb": round(memory_mb, 1),
                "cpu_percent": round(cpu_percent, 1),
                "uptime_hours": round(system_metrics.get("uptime_seconds", 0) / 3600, 1),
                "issues": issues
            }
        }
    except Exception as e:
        return {
            "status": HealthStatus.UNHEALTHY,
            "message": f"Erreur vérification système: {e}",
            "error": str(e)
        }


# === FONCTIONS UTILITAIRES AJOUTÉES ===

def initialize_utils():
    """
    Initialise tous les composants utils
    """
    # Le cache manager s'auto-initialise
    # Le metrics collector s'auto-initialise
    logger.info("Utils initialized")


def shutdown_utils():
    """
    Arrêt propre de tous les composants utils
    """
    # Nettoyer les métriques anciennes
    cleanup_old_metrics(hours=1)
    logger.info("Utils shutdown complete")


# === EXPORTS ===

__all__ = [
    # === ENUMS ===
    "HealthStatus",
    "ComponentType",
    
    # === FONCTIONS PRINCIPALES ===
    "get_system_metrics",
    "get_performance_summary", 
    "get_utils_performance",
    "cleanup_old_metrics",
    "get_utils_health",
    
    # === FONCTIONS UTILITAIRES ===
    "initialize_utils",
    "shutdown_utils"
]