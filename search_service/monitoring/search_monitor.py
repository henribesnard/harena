"""
Module de monitoring et métriques pour le service de recherche.

Ce module centralise le monitoring des performances, la santé des services
et les alertes pour Elasticsearch et Qdrant.
"""
import logging
import time
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum

logger = logging.getLogger("search_service.monitoring")


class ServiceStatus(Enum):
    """États possibles d'un service."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ServiceHealth:
    """Représente l'état de santé d'un service."""
    status: ServiceStatus
    last_check: datetime
    response_time_ms: Optional[float] = None
    error_message: Optional[str] = None
    consecutive_failures: int = 0
    uptime_percentage: float = 100.0
    
    def is_healthy(self) -> bool:
        return self.status == ServiceStatus.HEALTHY


@dataclass
class SearchMetrics:
    """Métriques de recherche."""
    total_searches: int = 0
    successful_searches: int = 0
    failed_searches: int = 0
    avg_response_time_ms: float = 0.0
    
    # Métriques par type de recherche
    lexical_searches: int = 0
    semantic_searches: int = 0
    hybrid_searches: int = 0
    
    # Performance
    p95_response_time_ms: float = 0.0
    p99_response_time_ms: float = 0.0
    
    def success_rate(self) -> float:
        if self.total_searches == 0:
            return 100.0
        return (self.successful_searches / self.total_searches) * 100.0


class SearchMonitor:
    """Monitor centralisé pour le service de recherche."""
    
    def __init__(self, check_interval: int = 60):
        self.check_interval = check_interval  # Secondes
        self.running = False
        
        # État des services
        self.elasticsearch_health = ServiceHealth(
            status=ServiceStatus.UNKNOWN,
            last_check=datetime.now()
        )
        self.qdrant_health = ServiceHealth(
            status=ServiceStatus.UNKNOWN,
            last_check=datetime.now()
        )
        
        # Métriques
        self.search_metrics = SearchMetrics()
        self.response_times: deque = deque(maxlen=1000)  # Dernier 1000 recherches
        
        # Historique pour calculs de tendances
        self.health_history: Dict[str, deque] = {
            "elasticsearch": deque(maxlen=100),
            "qdrant": deque(maxlen=100)
        }
        
        # Alertes
        self.alert_thresholds = {
            "max_response_time_ms": 5000,
            "min_success_rate": 95.0,
            "max_consecutive_failures": 3
        }
        self.active_alerts: List[Dict[str, Any]] = []
        
        # Clients pour les health checks
        self.elasticsearch_client = None
        self.qdrant_client = None
        
        # Temps de démarrage
        self._start_time = None
    
    def set_clients(self, elasticsearch_client=None, qdrant_client=None):
        """Configure les clients pour les health checks."""
        self.elasticsearch_client = elasticsearch_client
        self.qdrant_client = qdrant_client
        logger.info("🔧 Clients configurés pour le monitoring")
    
    async def start_monitoring(self):
        """Démarre le monitoring en arrière-plan."""
        if self.running:
            logger.warning("⚠️ Monitoring déjà en cours")
            return
        
        self.running = True
        self._start_time = time.time()
        logger.info(f"🚀 Démarrage monitoring (intervalle: {self.check_interval}s)")
        
        # Lancer la boucle de monitoring
        asyncio.create_task(self._monitoring_loop())
    
    async def stop_monitoring(self):
        """Arrête le monitoring."""
        self.running = False
        logger.info("🛑 Arrêt du monitoring")
    
    async def _monitoring_loop(self):
        """Boucle principale de monitoring."""
        while self.running:
            try:
                await self._perform_health_checks()
                await self._check_alerts()
                await asyncio.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"💥 Erreur dans la boucle de monitoring: {e}")
                await asyncio.sleep(5)  # Pause courte en cas d'erreur
    
    async def _perform_health_checks(self):
        """Effectue les vérifications de santé."""
        logger.debug("🩺 Vérifications de santé périodiques...")
        
        # Check Elasticsearch
        if self.elasticsearch_client:
            await self._check_elasticsearch_health()
        
        # Check Qdrant
        if self.qdrant_client:
            await self._check_qdrant_health()
        
        # Log du statut global
        self._log_overall_status()
    
    async def _check_elasticsearch_health(self):
        """Vérifie la santé d'Elasticsearch."""
        start_time = time.time()
        
        try:
            is_healthy = await self.elasticsearch_client.is_healthy()
            response_time = (time.time() - start_time) * 1000
            
            if is_healthy:
                self.elasticsearch_health.status = ServiceStatus.HEALTHY
                self.elasticsearch_health.consecutive_failures = 0
                self.elasticsearch_health.error_message = None
            else:
                self._handle_service_failure("elasticsearch", "Health check failed")
            
            self.elasticsearch_health.response_time_ms = response_time
            self.elasticsearch_health.last_check = datetime.now()
            
            # Enregistrer dans l'historique
            self.health_history["elasticsearch"].append({
                "timestamp": datetime.now(),
                "healthy": is_healthy,
                "response_time_ms": response_time
            })
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            self._handle_service_failure("elasticsearch", str(e))
            self.elasticsearch_health.response_time_ms = response_time
    
    async def _check_qdrant_health(self):
        """Vérifie la santé de Qdrant."""
        start_time = time.time()
        
        try:
            is_healthy = await self.qdrant_client.is_healthy()
            response_time = (time.time() - start_time) * 1000
            
            if is_healthy:
                self.qdrant_health.status = ServiceStatus.HEALTHY
                self.qdrant_health.consecutive_failures = 0
                self.qdrant_health.error_message = None
            else:
                self._handle_service_failure("qdrant", "Health check failed")
            
            self.qdrant_health.response_time_ms = response_time
            self.qdrant_health.last_check = datetime.now()
            
            # Enregistrer dans l'historique
            self.health_history["qdrant"].append({
                "timestamp": datetime.now(),
                "healthy": is_healthy,
                "response_time_ms": response_time
            })
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            self._handle_service_failure("qdrant", str(e))
            self.qdrant_health.response_time_ms = response_time
    
    def _handle_service_failure(self, service_name: str, error_message: str):
        """Gère les échecs de service."""
        if service_name == "elasticsearch":
            health = self.elasticsearch_health
        elif service_name == "qdrant":
            health = self.qdrant_health
        else:
            return
        
        health.consecutive_failures += 1
        health.error_message = error_message
        
        # Déterminer le nouveau statut
        if health.consecutive_failures >= self.alert_thresholds["max_consecutive_failures"]:
            health.status = ServiceStatus.UNHEALTHY
        else:
            health.status = ServiceStatus.DEGRADED
        
        logger.warning(
            f"⚠️ Service {service_name} en échec "
            f"({health.consecutive_failures} fois consécutives): {error_message}"
        )
    
    def _log_overall_status(self):
        """Log le statut global du système."""
        es_status = self.elasticsearch_health.status.value
        qdrant_status = self.qdrant_health.status.value
        
        # Déterminer le statut global
        if (self.elasticsearch_health.is_healthy() and 
            self.qdrant_health.is_healthy()):
            overall_status = "healthy"
            icon = "✅"
        elif (self.elasticsearch_health.status == ServiceStatus.UNHEALTHY or
              self.qdrant_health.status == ServiceStatus.UNHEALTHY):
            overall_status = "unhealthy"
            icon = "🚨"
        else:
            overall_status = "degraded"
            icon = "⚠️"
        
        logger.info(
            f"{icon} État système: {overall_status} "
            f"(ES: {es_status}, Qdrant: {qdrant_status})"
        )
    
    async def _check_alerts(self):
        """Vérifie et génère les alertes."""
        new_alerts = []
        
        # Alerte temps de réponse
        if self.search_metrics.avg_response_time_ms > self.alert_thresholds["max_response_time_ms"]:
            new_alerts.append({
                "type": "high_response_time",
                "message": f"Temps de réponse élevé: {self.search_metrics.avg_response_time_ms:.0f}ms",
                "severity": "warning",
                "timestamp": datetime.now()
            })
        
        # Alerte taux de succès
        if self.search_metrics.success_rate() < self.alert_thresholds["min_success_rate"]:
            new_alerts.append({
                "type": "low_success_rate",
                "message": f"Taux de succès faible: {self.search_metrics.success_rate():.1f}%",
                "severity": "critical",
                "timestamp": datetime.now()
            })
        
        # Alerte services indisponibles
        if not self.elasticsearch_health.is_healthy():
            new_alerts.append({
                "type": "elasticsearch_unhealthy",
                "message": f"Elasticsearch indisponible: {self.elasticsearch_health.error_message}",
                "severity": "critical",
                "timestamp": datetime.now()
            })
        
        if not self.qdrant_health.is_healthy():
            new_alerts.append({
                "type": "qdrant_unhealthy",
                "message": f"Qdrant indisponible: {self.qdrant_health.error_message}",
                "severity": "critical",
                "timestamp": datetime.now()
            })
        
        # Mettre à jour les alertes actives
        self.active_alerts = new_alerts
        
        # Logger les nouvelles alertes
        for alert in new_alerts:
            severity_icon = "🚨" if alert["severity"] == "critical" else "⚠️"
            logger.warning(f"{severity_icon} ALERTE {alert['type']}: {alert['message']}")
    
    def record_search(self, 
                     search_type: str, 
                     success: bool, 
                     response_time_ms: float,
                     error_type: Optional[str] = None):
        """Enregistre une opération de recherche."""
        self.search_metrics.total_searches += 1
        
        if success:
            self.search_metrics.successful_searches += 1
        else:
            self.search_metrics.failed_searches += 1
            if error_type:
                logger.debug(f"❌ Recherche échouée: {error_type}")
        
        # Enregistrer le type de recherche
        if search_type == "lexical":
            self.search_metrics.lexical_searches += 1
        elif search_type == "semantic":
            self.search_metrics.semantic_searches += 1
        elif search_type == "hybrid":
            self.search_metrics.hybrid_searches += 1
        
        # Enregistrer le temps de réponse
        self.response_times.append(response_time_ms)
        
        # Recalculer les métriques
        self._update_response_time_metrics()
    
    def _update_response_time_metrics(self):
        """Met à jour les métriques de temps de réponse."""
        if not self.response_times:
            return
        
        times = list(self.response_times)
        times.sort()
        
        # Moyenne
        self.search_metrics.avg_response_time_ms = sum(times) / len(times)
        
        # Percentiles
        p95_index = int(len(times) * 0.95)
        p99_index = int(len(times) * 0.99)
        
        if p95_index < len(times):
            self.search_metrics.p95_response_time_ms = times[p95_index]
        
        if p99_index < len(times):
            self.search_metrics.p99_response_time_ms = times[p99_index]
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Retourne un résumé de l'état de santé."""
        return {
            "overall_status": self._get_overall_status(),
            "elasticsearch": {
                "status": self.elasticsearch_health.status.value,
                "last_check": self.elasticsearch_health.last_check.isoformat(),
                "response_time_ms": self.elasticsearch_health.response_time_ms,
                "consecutive_failures": self.elasticsearch_health.consecutive_failures,
                "error_message": self.elasticsearch_health.error_message
            },
            "qdrant": {
                "status": self.qdrant_health.status.value,
                "last_check": self.qdrant_health.last_check.isoformat(),
                "response_time_ms": self.qdrant_health.response_time_ms,
                "consecutive_failures": self.qdrant_health.consecutive_failures,
                "error_message": self.qdrant_health.error_message
            },
            "search_metrics": {
                "total_searches": self.search_metrics.total_searches,
                "success_rate": self.search_metrics.success_rate(),
                "avg_response_time_ms": self.search_metrics.avg_response_time_ms,
                "p95_response_time_ms": self.search_metrics.p95_response_time_ms,
                "p99_response_time_ms": self.search_metrics.p99_response_time_ms,
                "lexical_searches": self.search_metrics.lexical_searches,
                "semantic_searches": self.search_metrics.semantic_searches,
                "hybrid_searches": self.search_metrics.hybrid_searches
            },
            "active_alerts": self.active_alerts
        }
    
    def _get_overall_status(self) -> str:
        """Calcule le statut global du système."""
        if (self.elasticsearch_health.is_healthy() and 
            self.qdrant_health.is_healthy()):
            return "healthy"
        elif (self.elasticsearch_health.status == ServiceStatus.UNHEALTHY or
              self.qdrant_health.status == ServiceStatus.UNHEALTHY):
            return "unhealthy"
        else:
            return "degraded"
    
    def get_uptime_stats(self, hours: int = 24) -> Dict[str, float]:
        """Calcule les statistiques d'uptime."""
        cutoff = datetime.now() - timedelta(hours=hours)
        
        stats = {}
        
        for service_name, history in self.health_history.items():
            recent_checks = [
                check for check in history 
                if check["timestamp"] > cutoff
            ]
            
            if recent_checks:
                healthy_checks = sum(1 for check in recent_checks if check["healthy"])
                uptime_percentage = (healthy_checks / len(recent_checks)) * 100
                stats[service_name] = uptime_percentage
            else:
                stats[service_name] = 100.0  # Pas de données = supposé healthy
        
        return stats
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Retourne un résumé des métriques pour export."""
        uptime = time.time() - self._start_time if self._start_time else 0
        
        return {
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": uptime,
            "service_health": {
                "elasticsearch": {
                    "healthy": self.elasticsearch_health.is_healthy(),
                    "status": self.elasticsearch_health.status.value,
                    "response_time_ms": self.elasticsearch_health.response_time_ms,
                    "consecutive_failures": self.elasticsearch_health.consecutive_failures
                },
                "qdrant": {
                    "healthy": self.qdrant_health.is_healthy(),
                    "status": self.qdrant_health.status.value,
                    "response_time_ms": self.qdrant_health.response_time_ms,
                    "consecutive_failures": self.qdrant_health.consecutive_failures
                }
            },
            "search_performance": {
                "total_searches": self.search_metrics.total_searches,
                "success_rate_percent": self.search_metrics.success_rate(),
                "avg_response_time_ms": self.search_metrics.avg_response_time_ms,
                "p95_response_time_ms": self.search_metrics.p95_response_time_ms,
                "p99_response_time_ms": self.search_metrics.p99_response_time_ms
            },
            "search_types": {
                "lexical": self.search_metrics.lexical_searches,
                "semantic": self.search_metrics.semantic_searches,
                "hybrid": self.search_metrics.hybrid_searches
            },
            "alerts": {
                "active_count": len(self.active_alerts),
                "critical_count": sum(1 for alert in self.active_alerts 
                                    if alert.get("severity") == "critical")
            }
        }
    
    def reset_metrics(self):
        """Remet à zéro les métriques (utile pour les tests)."""
        logger.info("🔄 Reset des métriques de monitoring")
        self.search_metrics = SearchMetrics()
        self.response_times.clear()
        self.active_alerts.clear()
    
    def set_alert_thresholds(self, thresholds: Dict[str, Any]):
        """Configure les seuils d'alerte."""
        self.alert_thresholds.update(thresholds)
        logger.info(f"🔧 Seuils d'alerte mis à jour: {self.alert_thresholds}")


# Instance globale du monitor
search_monitor = SearchMonitor()