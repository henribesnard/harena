"""
Collecteur de métriques pour conversation service
"""
import logging
import time
from typing import Dict, Any, List
from datetime import datetime, timezone
from collections import defaultdict, deque

# Configuration du logger
logger = logging.getLogger("conversation_service.metrics")

class MetricsCollector:
    """Collecteur de métriques simple pour Phase 1"""
    
    def __init__(self):
        # Compteurs simples
        self._counters: Dict[str, int] = defaultdict(int)
        
        # Histogrammes (dernières 1000 valeurs)
        self._histograms: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Gauges (dernière valeur)
        self._gauges: Dict[str, float] = {}
        
        # Timestamps pour rates
        self._timestamps: Dict[str, List[float]] = defaultdict(list)
        
        # Métriques de session
        self._session_start = datetime.now(timezone.utc)
        
        logger.info("MetricsCollector initialisé")
    
    def increment_counter(self, metric_name: str, value: int = 1) -> None:
        """Incrémenter un compteur"""
        try:
            self._counters[metric_name] += value
            
            # Log métriques importantes
            if metric_name.endswith('.errors'):
                logger.warning(f"Error counter incremented: {metric_name} = {self._counters[metric_name]}")
                
        except Exception as e:
            logger.error(f"Erreur increment counter {metric_name}: {str(e)}")
    
    def record_histogram(self, metric_name: str, value: float) -> None:
        """Enregistrer valeur dans histogramme"""
        try:
            self._histograms[metric_name].append(value)
            
            # Log valeurs extrêmes
            if value > 5000:  # > 5 secondes
                logger.warning(f"High latency recorded: {metric_name} = {value}ms")
                
        except Exception as e:
            logger.error(f"Erreur record histogram {metric_name}: {str(e)}")
    
    def record_gauge(self, metric_name: str, value: float) -> None:
        """Enregistrer valeur de gauge"""
        try:
            self._gauges[metric_name] = value
            
        except Exception as e:
            logger.error(f"Erreur record gauge {metric_name}: {str(e)}")
    
    def record_rate(self, metric_name: str) -> None:
        """Enregistrer événement pour calcul de taux"""
        try:
            current_time = time.time()
            
            # Garder seulement dernière heure
            one_hour_ago = current_time - 3600
            self._timestamps[metric_name] = [
                ts for ts in self._timestamps[metric_name] 
                if ts > one_hour_ago
            ]
            
            self._timestamps[metric_name].append(current_time)
            
        except Exception as e:
            logger.error(f"Erreur record rate {metric_name}: {str(e)}")
    
    def get_counter(self, metric_name: str) -> int:
        """Récupérer valeur compteur"""
        return self._counters.get(metric_name, 0)
    
    def get_histogram_stats(self, metric_name: str) -> Dict[str, float]:
        """Statistiques histogramme"""
        try:
            values = list(self._histograms.get(metric_name, []))
            
            if not values:
                return {"count": 0}
            
            values.sort()
            count = len(values)
            
            return {
                "count": count,
                "min": values[0],
                "max": values[-1],
                "avg": sum(values) / count,
                "p50": values[count // 2],
                "p95": values[int(count * 0.95)] if count > 20 else values[-1],
                "p99": values[int(count * 0.99)] if count > 100 else values[-1]
            }
            
        except Exception as e:
            logger.error(f"Erreur histogram stats {metric_name}: {str(e)}")
            return {"count": 0, "error": str(e)}
    
    def get_gauge(self, metric_name: str) -> float:
        """Récupérer valeur gauge"""
        return self._gauges.get(metric_name, 0.0)
    
    def get_rate(self, metric_name: str, window_seconds: int = 60) -> float:
        """Calcul taux sur fenêtre temporelle"""
        try:
            current_time = time.time()
            window_start = current_time - window_seconds
            
            timestamps = self._timestamps.get(metric_name, [])
            recent_events = [ts for ts in timestamps if ts > window_start]
            
            return len(recent_events) / window_seconds  # événements par seconde
            
        except Exception as e:
            logger.error(f"Erreur rate calculation {metric_name}: {str(e)}")
            return 0.0
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Export complet des métriques"""
        try:
            # Uptime
            uptime_seconds = (datetime.now(timezone.utc) - self._session_start).total_seconds()
            
            metrics = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "uptime_seconds": uptime_seconds,
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "histograms": {},
                "rates": {}
            }
            
            # Stats histogrammes
            for name in self._histograms.keys():
                metrics["histograms"][name] = self.get_histogram_stats(name)
            
            # Rates courantes
            for name in self._timestamps.keys():
                metrics["rates"][f"{name}_per_second"] = self.get_rate(name, 60)
                metrics["rates"][f"{name}_per_minute"] = self.get_rate(name, 60) * 60
            
            return metrics
            
        except Exception as e:
            logger.error(f"Erreur export métriques: {str(e)}")
            return {"error": str(e)}
    
    def get_health_metrics(self) -> Dict[str, Any]:
        """Métriques santé service"""
        try:
            # Compteurs d'erreurs
            total_requests = self.get_counter("conversation.requests.total")
            total_errors = (
                self.get_counter("conversation.errors.technical") +
                self.get_counter("conversation.errors.auth") +
                self.get_counter("conversation.errors.validation")
            )
            
            error_rate = (total_errors / max(total_requests, 1)) * 100
            
            # Latence
            latency_stats = self.get_histogram_stats("conversation.processing_time")
            
            # Statut santé
            health_status = "healthy"
            if error_rate > 5.0:
                health_status = "degraded"
            if error_rate > 15.0 or latency_stats.get("p95", 0) > 5000:
                health_status = "unhealthy"
            
            return {
                "status": health_status,
                "total_requests": total_requests,
                "total_errors": total_errors,
                "error_rate_percent": round(error_rate, 2),
                "latency_p95_ms": latency_stats.get("p95", 0),
                "uptime_seconds": (datetime.now(timezone.utc) - self._session_start).total_seconds()
            }
            
        except Exception as e:
            logger.error(f"Erreur health metrics: {str(e)}")
            return {"status": "unknown", "error": str(e)}
    
    def reset_metrics(self) -> None:
        """Reset toutes les métriques"""
        try:
            self._counters.clear()
            self._histograms.clear()
            self._gauges.clear()
            self._timestamps.clear()
            self._session_start = datetime.now(timezone.utc)
            
            logger.info("Métriques réinitialisées")
            
        except Exception as e:
            logger.error(f"Erreur reset métriques: {str(e)}")
    
    def log_metrics_summary(self) -> None:
        """Log résumé métriques importantes"""
        try:
            health = self.get_health_metrics()
            
            logger.info(
                f"Metrics Summary - Status: {health['status']}, "
                f"Requests: {health['total_requests']}, "
                f"Errors: {health['total_errors']}, "
                f"Error Rate: {health['error_rate_percent']}%, "
                f"P95 Latency: {health['latency_p95_ms']}ms"
            )
            
        except Exception as e:
            logger.error(f"Erreur log metrics summary: {str(e)}")

# Instance globale pour faciliter l'usage
metrics_collector = MetricsCollector()