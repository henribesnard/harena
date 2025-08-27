"""
Collecteur de métriques optimisé pour conversation service avec agrégation avancée
"""
import logging
import time
import threading
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timezone
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Configuration du logger
logger = logging.getLogger("conversation_service.metrics")


class MetricType(str, Enum):
    """Types de métriques supportées"""
    COUNTER = "counter"
    HISTOGRAM = "histogram"
    GAUGE = "gauge"
    RATE = "rate"
    TIMER = "timer"


@dataclass
class MetricValue:
    """Valeur métrique avec métadonnées"""
    value: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HistogramStats:
    """Statistiques calculées pour histogramme"""
    count: int = 0
    sum: float = 0.0
    min: float = float('inf')
    max: float = float('-inf')
    mean: float = 0.0
    p50: float = 0.0
    p90: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Conversion en dictionnaire"""
        return {
            "count": self.count,
            "sum": self.sum,
            "min": self.min if self.min != float('inf') else 0.0,
            "max": self.max if self.max != float('-inf') else 0.0,
            "mean": self.mean,
            "p50": self.p50,
            "p90": self.p90,
            "p95": self.p95,
            "p99": self.p99
        }


class AdvancedMetricsCollector:
    """Collecteur de métriques avancé avec agrégation temps réel et thread-safety"""
    
    def __init__(self, max_history: int = 10000, aggregation_interval: int = 60):
        # Configuration
        self.max_history = max_history
        self.aggregation_interval = aggregation_interval
        
        # Stockage métriques thread-safe
        self._lock = threading.RLock()
        self._counters: Dict[str, int] = defaultdict(int)
        self._gauges: Dict[str, MetricValue] = {}
        self._histograms: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self._rates: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))  # Dernière heure
        self._timers: Dict[str, List[float]] = defaultdict(list)
        
        # Cache des statistiques calculées
        self._histogram_cache: Dict[str, HistogramStats] = {}
        self._cache_timestamps: Dict[str, datetime] = {}
        self._cache_ttl = 30  # 30 secondes
        
        # Métriques de session
        self._session_start = datetime.now(timezone.utc)
        self._total_operations = 0
        
        # Labels et contexte
        self._global_labels: Dict[str, str] = {}
        self._metric_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Agrégation asynchrone
        self._aggregation_task: Optional[asyncio.Task] = None
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="metrics")
        
        logger.info(f"AdvancedMetricsCollector initialisé - Histoire: {max_history}, Agrégation: {aggregation_interval}s")
    
    def set_global_labels(self, labels: Dict[str, str]) -> None:
        """Définit des labels globaux pour toutes les métriques.

        Ajoute automatiquement la version du service si elle n'est pas
        fournie, en la récupérant depuis la configuration quand c'est
        possible. Fallback sur "1.1.0" si aucune configuration disponible.
        """
        try:
            if "version" not in labels:
                from config_service.config import settings  # type: ignore
                labels["version"] = getattr(
                    settings,
                    "CONVERSATION_SERVICE_VERSION",
                    getattr(settings, "APP_VERSION", "1.1.0"),
                )
        except Exception:
            labels.setdefault("version", "1.1.0")

        with self._lock:
            self._global_labels.update(labels)

        logger.info(f"Labels globaux mis à jour: {labels}")
    
    def increment_counter(
        self, 
        metric_name: str, 
        value: int = 1,
        labels: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Incrémente un compteur avec labels et métadonnées"""
        try:
            with self._lock:
                # Création de la clé avec labels
                key = self._create_metric_key(metric_name, labels)
                self._counters[key] += value
                self._total_operations += 1
                
                # Stockage métadonnées
                if metadata:
                    self._metric_metadata[key] = metadata
                
                # Log métriques critiques
                if "error" in metric_name.lower() and value > 0:
                    logger.warning(f"Métrique erreur incrémentée: {key} (+{value}) = {self._counters[key]}")
                elif "critical" in metric_name.lower():
                    logger.error(f"Métrique critique: {key} (+{value}) = {self._counters[key]}")
                    
        except Exception as e:
            logger.error(f"Erreur increment counter {metric_name}: {str(e)}")
    
    def record_histogram(
        self, 
        metric_name: str, 
        value: float,
        labels: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Enregistre une valeur dans un histogramme"""
        try:
            with self._lock:
                key = self._create_metric_key(metric_name, labels)
                metric_value = MetricValue(
                    value=value,
                    timestamp=datetime.now(timezone.utc),
                    labels=labels or {},
                    metadata=metadata or {}
                )
                self._histograms[key].append(metric_value)
                
                # Invalide le cache pour ce métrique
                if key in self._histogram_cache:
                    del self._histogram_cache[key]
                    del self._cache_timestamps[key]
                
                # Log valeurs extrêmes
                if value > 10000:  # > 10 secondes
                    logger.warning(f"Valeur élevée enregistrée: {key} = {value}")
                elif value < 0:
                    logger.warning(f"Valeur négative enregistrée: {key} = {value}")
                    
        except Exception as e:
            logger.error(f"Erreur record histogram {metric_name}: {str(e)}")
    
    def record_gauge(
        self, 
        metric_name: str, 
        value: float,
        labels: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Enregistre une valeur de jauge (dernière valeur)"""
        try:
            with self._lock:
                key = self._create_metric_key(metric_name, labels)
                self._gauges[key] = MetricValue(
                    value=value,
                    timestamp=datetime.now(timezone.utc),
                    labels=labels or {},
                    metadata=metadata or {}
                )
                
        except Exception as e:
            logger.error(f"Erreur record gauge {metric_name}: {str(e)}")
    
    def record_rate(
        self, 
        metric_name: str,
        labels: Optional[Dict[str, str]] = None,
        timestamp: Optional[datetime] = None
    ) -> None:
        """Enregistre un événement pour calcul de taux"""
        try:
            current_time = timestamp or datetime.now(timezone.utc)
            
            with self._lock:
                key = self._create_metric_key(metric_name, labels)
                
                # Nettoyage automatique (garde dernière heure)
                one_hour_ago = current_time.timestamp() - 3600
                while self._rates[key] and self._rates[key][0] < one_hour_ago:
                    self._rates[key].popleft()
                
                self._rates[key].append(current_time.timestamp())
                
        except Exception as e:
            logger.error(f"Erreur record rate {metric_name}: {str(e)}")
    
    def start_timer(self, metric_name: str) -> Callable[[], None]:
        """Démarre un timer et retourne une fonction pour l'arrêter"""
        start_time = time.time()
        
        def stop_timer(labels: Optional[Dict[str, str]] = None) -> float:
            """Arrête le timer et enregistre la durée"""
            duration = time.time() - start_time
            self.record_histogram(f"{metric_name}.duration", duration * 1000, labels)  # en ms
            return duration
        
        return stop_timer
    
    def _create_metric_key(self, metric_name: str, labels: Optional[Dict[str, str]]) -> str:
        """Crée une clé unique pour la métrique avec labels"""
        if not labels and not self._global_labels:
            return metric_name
        
        # Fusion labels globaux et spécifiques
        all_labels = {**self._global_labels, **(labels or {})}
        
        if not all_labels:
            return metric_name
        
        # Tri des labels pour cohérence
        label_str = ",".join(f"{k}={v}" for k, v in sorted(all_labels.items()))
        return f"{metric_name}{{${label_str}}}"
    
    def get_counter(self, metric_name: str, labels: Optional[Dict[str, str]] = None) -> int:
        """Récupère la valeur d'un compteur"""
        key = self._create_metric_key(metric_name, labels)
        with self._lock:
            return self._counters.get(key, 0)
    
    def get_histogram_stats(
        self, 
        metric_name: str, 
        labels: Optional[Dict[str, str]] = None,
        force_recalculate: bool = False
    ) -> Dict[str, float]:
        """Statistiques histogramme avec cache intelligent"""
        try:
            key = self._create_metric_key(metric_name, labels)
            
            # Vérification cache
            if not force_recalculate and key in self._histogram_cache:
                cache_age = (datetime.now(timezone.utc) - self._cache_timestamps.get(key, datetime.min)).total_seconds()
                if cache_age < self._cache_ttl:
                    return self._histogram_cache[key].to_dict()
            
            with self._lock:
                values = self._histograms.get(key, deque())
                
                if not values:
                    return {"count": 0}
                
                # Extraction valeurs numériques
                numeric_values = []
                for item in values:
                    if isinstance(item, MetricValue):
                        numeric_values.append(item.value)
                    else:
                        numeric_values.append(float(item))
                
                # Calcul statistiques
                stats = self._calculate_histogram_stats(numeric_values)
                
                # Mise en cache
                self._histogram_cache[key] = stats
                self._cache_timestamps[key] = datetime.now(timezone.utc)
                
                return stats.to_dict()
                
        except Exception as e:
            logger.error(f"Erreur histogram stats {metric_name}: {str(e)}")
            return {"count": 0, "error": str(e)}
    
    def _calculate_histogram_stats(self, values: List[float]) -> HistogramStats:
        """Calcule les statistiques pour un histogramme"""
        if not values:
            return HistogramStats()
        
        sorted_values = sorted(values)
        count = len(sorted_values)
        
        stats = HistogramStats()
        stats.count = count
        stats.sum = sum(sorted_values)
        stats.min = sorted_values[0]
        stats.max = sorted_values[-1]
        stats.mean = stats.sum / count
        
        # Percentiles
        if count >= 2:
            stats.p50 = self._percentile(sorted_values, 0.50)
            stats.p90 = self._percentile(sorted_values, 0.90)
            stats.p95 = self._percentile(sorted_values, 0.95)
            stats.p99 = self._percentile(sorted_values, 0.99)
        else:
            stats.p50 = stats.p90 = stats.p95 = stats.p99 = sorted_values[0]
        
        return stats
    
    def _percentile(self, sorted_values: List[float], percentile: float) -> float:
        """Calcule un percentile sur des valeurs triées"""
        if not sorted_values:
            return 0.0
        
        index = percentile * (len(sorted_values) - 1)
        lower = int(index)
        upper = min(lower + 1, len(sorted_values) - 1)
        
        if lower == upper:
            return sorted_values[lower]
        
        # Interpolation linéaire
        weight = index - lower
        return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight
    
    def get_gauge(self, metric_name: str, labels: Optional[Dict[str, str]] = None) -> float:
        """Récupère la valeur d'une jauge"""
        key = self._create_metric_key(metric_name, labels)
        with self._lock:
            metric_value = self._gauges.get(key)
            return metric_value.value if metric_value else 0.0
    
    def get_rate(
        self, 
        metric_name: str, 
        window_seconds: int = 60,
        labels: Optional[Dict[str, str]] = None
    ) -> float:
        """Calcul taux sur fenêtre temporelle glissante"""
        try:
            key = self._create_metric_key(metric_name, labels)
            current_time = time.time()
            window_start = current_time - window_seconds
            
            with self._lock:
                timestamps = self._rates.get(key, deque())
                recent_events = [ts for ts in timestamps if ts > window_start]
                
                return len(recent_events) / window_seconds if window_seconds > 0 else 0.0
                
        except Exception as e:
            logger.error(f"Erreur rate calculation {metric_name}: {str(e)}")
            return 0.0
    
    def get_all_metrics(self, include_metadata: bool = True) -> Dict[str, Any]:
        """Export complet des métriques avec agrégation avancée"""
        try:
            current_time = datetime.now(timezone.utc)
            uptime_seconds = (current_time - self._session_start).total_seconds()
            
            with self._lock:
                metrics = {
                    "timestamp": current_time.isoformat(),
                    "uptime_seconds": uptime_seconds,
                    "total_operations": self._total_operations,
                    "global_labels": self._global_labels.copy(),
                    "counters": dict(self._counters),
                    "gauges": {},
                    "histograms": {},
                    "rates": {}
                }
                
                # Export jauges avec timestamps
                for key, metric_value in self._gauges.items():
                    metrics["gauges"][key] = {
                        "value": metric_value.value,
                        "timestamp": metric_value.timestamp.isoformat()
                    }
                    if include_metadata and metric_value.labels:
                        metrics["gauges"][key]["labels"] = metric_value.labels
                
                # Export histogrammes avec stats
                for key in self._histograms.keys():
                    metrics["histograms"][key] = self.get_histogram_stats(key)
                
                # Export taux pour différentes fenêtres
                for key in self._rates.keys():
                    metrics["rates"][key] = {
                        "per_second_1m": self.get_rate(key, 60),
                        "per_second_5m": self.get_rate(key, 300),
                        "per_second_1h": self.get_rate(key, 3600),
                        "per_minute_1m": self.get_rate(key, 60) * 60,
                        "per_minute_5m": self.get_rate(key, 300) * 60
                    }
                
                # Métadonnées optionnelles
                if include_metadata:
                    metrics["metadata"] = {
                        "cache_stats": {
                            "histogram_cache_size": len(self._histogram_cache),
                            "cache_hit_ratio": self._calculate_cache_hit_ratio()
                        },
                        "storage_stats": {
                            "counters_count": len(self._counters),
                            "gauges_count": len(self._gauges),
                            "histograms_count": len(self._histograms),
                            "rates_count": len(self._rates)
                        }
                    }
                
                return metrics
                
        except Exception as e:
            logger.error(f"Erreur export métriques: {str(e)}")
            return {"error": str(e), "timestamp": datetime.now(timezone.utc).isoformat()}
    
    def _calculate_cache_hit_ratio(self) -> float:
        """Calcule le ratio de hit du cache des histogrammes"""
        if not hasattr(self, '_cache_requests'):
            return 0.0
        return getattr(self, '_cache_hits', 0) / max(getattr(self, '_cache_requests', 1), 1)
    
    def get_health_metrics(self) -> Dict[str, Any]:
        """Métriques santé service optimisées"""
        try:
            # Calcul des métriques critiques
            total_requests = self.get_counter("conversation.requests.total")
            
            technical_errors = self.get_counter("conversation.errors.technical")
            auth_errors = self.get_counter("conversation.errors.auth")
            validation_errors = self.get_counter("conversation.errors.validation")
            total_errors = technical_errors + auth_errors + validation_errors
            
            error_rate = (total_errors / max(total_requests, 1)) * 100
            
            # Statistiques latence
            latency_stats = self.get_histogram_stats("conversation.processing_time")
            avg_latency = latency_stats.get("mean", 0)
            p95_latency = latency_stats.get("p95", 0)
            
            # Détermination statut santé avec règles dynamiques
            health_status = "healthy"
            health_score = 100.0
            
            # Règles de dégradation
            if error_rate > 5.0:
                health_status = "degraded"
                health_score -= min(error_rate * 2, 40)
            
            if error_rate > 20.0 or p95_latency > 10000:  # 10s
                health_status = "unhealthy" 
                health_score = min(health_score, 30)
            
            if total_requests == 0:
                health_status = "unknown"
                health_score = 0
            
            uptime_seconds = (datetime.now(timezone.utc) - self._session_start).total_seconds()
            
            return {
                "status": health_status,
                "score": max(0, health_score),
                "total_requests": total_requests,
                "total_errors": total_errors,
                "error_rate_percent": round(error_rate, 2),
                "latency_avg_ms": round(avg_latency, 1),
                "latency_p95_ms": round(p95_latency, 1),
                "uptime_seconds": round(uptime_seconds, 1),
                "operations_per_second": round(self._total_operations / max(uptime_seconds, 1), 2),
                "health_details": {
                    "technical_errors": technical_errors,
                    "auth_errors": auth_errors,
                    "validation_errors": validation_errors,
                    "requests_count": total_requests,
                    "avg_latency_ms": round(avg_latency, 1)
                }
            }
            
        except Exception as e:
            logger.error(f"Erreur health metrics: {str(e)}")
            return {"status": "error", "error": str(e), "score": 0}
    
    def reset_metrics(self, preserve_session: bool = False) -> None:
        """Reset métriques avec option de préserver la session"""
        try:
            with self._lock:
                self._counters.clear()
                self._gauges.clear()
                self._histograms.clear()
                self._rates.clear()
                self._timers.clear()
                self._histogram_cache.clear()
                self._cache_timestamps.clear()
                self._metric_metadata.clear()
                
                if not preserve_session:
                    self._session_start = datetime.now(timezone.utc)
                    self._total_operations = 0
                
                logger.info(f"Métriques réinitialisées - Session préservée: {preserve_session}")
                
        except Exception as e:
            logger.error(f"Erreur reset métriques: {str(e)}")
    
    def export_prometheus_format(self) -> str:
        """Export format Prometheus/OpenMetrics"""
        try:
            lines = []
            lines.append("# Harena Conversation Service Metrics")
            lines.append(f"# Exported at {datetime.now(timezone.utc).isoformat()}")
            
            with self._lock:
                # Compteurs
                for key, value in self._counters.items():
                    metric_name, labels = self._parse_metric_key(key)
                    lines.append(f"# TYPE {metric_name} counter")
                    lines.append(f"{metric_name}{labels} {value}")
                
                # Jauges
                for key, metric_value in self._gauges.items():
                    metric_name, labels = self._parse_metric_key(key)
                    lines.append(f"# TYPE {metric_name} gauge")
                    lines.append(f"{metric_name}{labels} {metric_value.value}")
                
                # Histogrammes (format simplifié)
                for key in self._histograms.keys():
                    metric_name, labels = self._parse_metric_key(key)
                    stats = self.get_histogram_stats(key)
                    
                    lines.append(f"# TYPE {metric_name} histogram")
                    lines.append(f"{metric_name}_count{labels} {stats['count']}")
                    lines.append(f"{metric_name}_sum{labels} {stats['sum']}")
                    
                    for percentile in ['p50', 'p90', 'p95', 'p99']:
                        quantile = percentile[1:] if percentile != 'p50' else '50'
                        lines.append(f"{metric_name}_bucket{{le=\"{quantile}\"}}{labels[1:-1] + ',' if labels != '{}' else ''} {stats[percentile]}")
            
            return "\n".join(lines) + "\n"
            
        except Exception as e:
            logger.error(f"Erreur export Prometheus: {str(e)}")
            return f"# Export error: {str(e)}\n"
    
    def _parse_metric_key(self, key: str) -> tuple[str, str]:
        """Parse une clé métrique pour extraire nom et labels"""
        if "{$" in key:
            name, label_part = key.split("{$", 1)
            labels = "{" + label_part
            return name, labels
        return key, "{}"
    
    async def start_background_aggregation(self) -> None:
        """Démarre l'agrégation en arrière-plan"""
        if self._aggregation_task and not self._aggregation_task.done():
            logger.warning("Tâche d'agrégation déjà en cours")
            return
        
        self._aggregation_task = asyncio.create_task(self._background_aggregation())
        logger.info(f"Tâche d'agrégation démarrée - Intervalle: {self.aggregation_interval}s")
    
    async def _background_aggregation(self) -> None:
        """Tâche d'agrégation en arrière-plan"""
        try:
            while True:
                await asyncio.sleep(self.aggregation_interval)
                
                # Nettoyage périodique
                await self._cleanup_old_data()
                
                # Log métriques importantes
                self._log_key_metrics()
                
        except asyncio.CancelledError:
            logger.info("Tâche d'agrégation annulée")
        except Exception as e:
            logger.error(f"Erreur tâche d'agrégation: {str(e)}")
    
    async def _cleanup_old_data(self) -> None:
        """Nettoyage des données anciennes"""
        try:
            current_time = datetime.now(timezone.utc)
            cleaned_items = 0
            
            with self._lock:
                # Nettoyage cache histogrammes
                expired_keys = [
                    key for key, timestamp in self._cache_timestamps.items()
                    if (current_time - timestamp).total_seconds() > self._cache_ttl * 2
                ]
                
                for key in expired_keys:
                    if key in self._histogram_cache:
                        del self._histogram_cache[key]
                    if key in self._cache_timestamps:
                        del self._cache_timestamps[key]
                    cleaned_items += 1
                
                # Nettoyage rates (garde dernières 2h)
                two_hours_ago = current_time.timestamp() - 7200
                for key, timestamps in self._rates.items():
                    original_length = len(timestamps)
                    while timestamps and timestamps[0] < two_hours_ago:
                        timestamps.popleft()
                    cleaned_items += original_length - len(timestamps)
            
            if cleaned_items > 0:
                logger.debug(f"Nettoyage métríques: {cleaned_items} éléments supprimés")
                
        except Exception as e:
            logger.error(f"Erreur nettoyage données: {str(e)}")
    
    def _log_key_metrics(self) -> None:
        """Log des métriques clés périodiquement"""
        try:
            health = self.get_health_metrics()
            uptime_hours = health.get("uptime_seconds", 0) / 3600
            
            logger.info(
                f"Métriques clés - Status: {health['status']}, "
                f"Requêtes: {health['total_requests']}, "
                f"Erreurs: {health['total_errors']}, "
                f"Taux erreur: {health['error_rate_percent']}%, "
                f"Latence P95: {health['latency_p95_ms']}ms, "
                f"Uptime: {uptime_hours:.1f}h"
            )
            
        except Exception as e:
            logger.debug(f"Erreur log métriques: {str(e)}")
    
    async def stop_background_aggregation(self) -> None:
        """Arrête l'agrégation en arrière-plan"""
        if self._aggregation_task:
            self._aggregation_task.cancel()
            try:
                await self._aggregation_task
            except asyncio.CancelledError:
                pass
            
            logger.info("Tâche d'agrégation arrêtée")
    
    def __del__(self):
        """Nettoyage final"""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)


# Instance globale optimisée
metrics_collector = AdvancedMetricsCollector(
    max_history=10000,
    aggregation_interval=60
)

# Configuration labels globaux si disponible
try:
    from config_service.config import settings
    environment = getattr(settings, "ENVIRONMENT", "production")
    service_version = getattr(
        settings,
        "CONVERSATION_SERVICE_VERSION",
        getattr(settings, "APP_VERSION", "1.1.0"),
    )
    metrics_collector.set_global_labels(
        {
            "service": "conversation_service",
            "phase": "1",
            "environment": environment,
            "version": service_version,
        }
    )
except Exception:
    metrics_collector.set_global_labels(
        {
            "service": "conversation_service",
            "phase": "1",
            "version": "1.1.0",
        }
    )
    logger.warning("Configuration non disponible pour labels globaux")

logger.info("MetricsCollector global initialisé et configuré")
