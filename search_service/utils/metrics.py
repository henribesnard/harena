"""
Métriques spécialisées pour le Search Service.

Ce module fournit un système de métriques complet pour monitorer les performances
du service de recherche : latence des requêtes, taux de succès, utilisation du cache,
qualité des résultats, et patterns d'usage des utilisateurs.

MÉTRIQUES COLLECTÉES:
- Performance : latence, throughput, erreurs
- Qualité : scores de pertinence, taux de clics
- Cache : hit rate, miss rate, évictions
- Usage : requêtes populaires, patterns utilisateurs
- Elasticsearch : temps de réponse, index stats

FONCTIONNALITÉS:
- Collecte en temps réel avec buckets temporels
- Agrégations par période (minute, heure, jour)
- Alertes sur seuils configurables
- Export Prometheus/Grafana
- Métriques business (conversion, engagement)

USAGE:
    collector = MetricsCollector("search_service")
    
    # Enregistrer une requête
    with collector.time_query() as timer:
        results = await search_engine.search(query)
        timer.record_results(len(results), max_score)
    
    # Métriques du cache
    collector.record_cache_hit("search_results")
    collector.record_cache_miss("search_results")
    
    # Export
    prometheus_metrics = collector.export_prometheus()
"""

import asyncio
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Callable, NamedTuple
from contextlib import contextmanager
import statistics
import json

# Configuration centralisée
from config_service.config import settings

logger = logging.getLogger(__name__)

# ==================== ENUMS ET CONSTANTES ====================

class MetricType(str, Enum):
    """Types de métriques."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"

class AlertLevel(str, Enum):
    """Niveaux d'alerte."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

# Buckets pour histogrammes de latence (en secondes)
LATENCY_BUCKETS = [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]

# Buckets pour tailles de résultats
RESULT_SIZE_BUCKETS = [0, 1, 5, 10, 20, 50, 100, 200, 500, 1000]

# ==================== STRUCTURES DE DONNÉES ====================

@dataclass
class MetricValue:
    """Valeur d'une métrique avec métadonnées."""
    value: Union[int, float]
    timestamp: float
    labels: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.timestamp == 0:
            self.timestamp = time.time()

@dataclass 
class TimerResult:
    """Résultat d'un timer."""
    duration_seconds: float
    success: bool = True
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

class HistogramBucket(NamedTuple):
    """Bucket d'histogramme."""
    le: float  # less than or equal
    count: int

@dataclass
class HistogramData:
    """Données d'histogramme."""
    buckets: List[HistogramBucket] = field(default_factory=list)
    count: int = 0
    sum: float = 0.0
    
    def add_observation(self, value: float):
        """Ajoute une observation."""
        self.count += 1
        self.sum += value
        
        # Mettre à jour les buckets
        for i, bucket in enumerate(self.buckets):
            if value <= bucket.le:
                self.buckets[i] = HistogramBucket(bucket.le, bucket.count + 1)

@dataclass
class SearchMetrics:
    """Métriques spécifiques aux recherches."""
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    avg_latency_seconds: float = 0.0
    avg_result_count: float = 0.0
    avg_relevance_score: float = 0.0
    cache_hit_rate: float = 0.0
    popular_queries: Dict[str, int] = field(default_factory=dict)
    user_activity: Dict[int, int] = field(default_factory=dict)
    
    @property
    def success_rate(self) -> float:
        """Taux de succès des requêtes."""
        total = self.total_queries
        return self.successful_queries / total if total > 0 else 0.0
    
    @property
    def error_rate(self) -> float:
        """Taux d'erreur des requêtes."""
        return 1.0 - self.success_rate

@dataclass
class QueryMetrics:
    """Métriques d'une requête individuelle."""
    query: str
    user_id: int
    latency_seconds: float
    result_count: int
    max_score: float
    cache_hit: bool
    timestamp: float = field(default_factory=time.time)
    filters_applied: List[str] = field(default_factory=list)
    elasticsearch_time: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Conversion en dictionnaire."""
        return {
            "query": self.query,
            "user_id": self.user_id,
            "latency_seconds": self.latency_seconds,
            "result_count": self.result_count,
            "max_score": self.max_score,
            "cache_hit": self.cache_hit,
            "timestamp": self.timestamp,
            "filters_applied": self.filters_applied,
            "elasticsearch_time": self.elasticsearch_time
        }

@dataclass
class PerformanceMetrics:
    """Métriques de performance système."""
    requests_per_second: float = 0.0
    avg_response_time: float = 0.0
    p95_response_time: float = 0.0
    p99_response_time: float = 0.0
    active_connections: int = 0
    memory_usage_mb: float = 0.0
    elasticsearch_health: str = "unknown"
    cache_size: int = 0
    cache_memory_mb: float = 0.0

# ==================== COLLECTEUR DE MÉTRIQUES ====================

class MetricsCollector:
    """
    Collecteur principal de métriques pour le Search Service.
    
    Responsabilités:
    - Collecte des métriques en temps réel
    - Agrégation et calculs statistiques
    - Alertes sur seuils
    - Export dans différents formats
    """
    
    def __init__(
        self,
        service_name: str = "search_service",
        include_query_metrics: bool = True,
        include_performance_metrics: bool = True,
        include_cache_metrics: bool = True,
        history_retention_hours: int = 24
    ):
        self.service_name = service_name
        self.include_query_metrics = include_query_metrics
        self.include_performance_metrics = include_performance_metrics
        self.include_cache_metrics = include_cache_metrics
        
        # Stockage des métriques
        self._counters: Dict[str, int] = defaultdict(int)
        self._gauges: Dict[str, float] = defaultdict(float)
        self._histograms: Dict[str, HistogramData] = defaultdict(
            lambda: HistogramData(
                buckets=[HistogramBucket(bucket, 0) for bucket in LATENCY_BUCKETS]
            )
        )
        self._timers: Dict[str, List[float]] = defaultdict(list)
        
        # Métriques spécialisées
        self.search_metrics = SearchMetrics()
        self.performance_metrics = PerformanceMetrics()
        
        # Historique des requêtes
        self._query_history: deque = deque(maxlen=10000)
        self._response_times: deque = deque(maxlen=1000)  # Pour percentiles
        
        # Configuration des alertes
        self._alert_thresholds = {
            "error_rate": 0.05,  # 5% max
            "avg_latency": 2.0,  # 2s max
            "cache_hit_rate": 0.7  # 70% min
        }
        self._alert_callbacks: List[Callable] = []
        
        # Nettoyage périodique
        self._cleanup_task: Optional[asyncio.Task] = None
        self._retention_hours = history_retention_hours
        
        logger.info(f"Metrics collector initialized for {service_name}")
        self._start_cleanup_task()
    
    def _start_cleanup_task(self):
        """Démarre la tâche de nettoyage périodique."""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def _cleanup_loop(self):
        """Boucle de nettoyage des anciennes métriques."""
        while True:
            try:
                await asyncio.sleep(3600)  # Toutes les heures
                await self._cleanup_old_metrics()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Metrics cleanup error: {e}")
    
    async def _cleanup_old_metrics(self):
        """Nettoie les métriques anciennes."""
        cutoff_time = time.time() - (self._retention_hours * 3600)
        
        # Nettoyer l'historique des requêtes
        initial_size = len(self._query_history)
        self._query_history = deque(
            [q for q in self._query_history if q.timestamp > cutoff_time],
            maxlen=self._query_history.maxlen
        )
        
        cleaned = initial_size - len(self._query_history)
        if cleaned > 0:
            logger.debug(f"Cleaned {cleaned} old query metrics")
    
    # ==================== ENREGISTREMENT MÉTRIQUES ====================
    
    def increment_counter(self, name: str, value: int = 1, labels: Optional[Dict[str, str]] = None):
        """Incrémente un compteur."""
        key = self._make_key(name, labels)
        self._counters[key] += value
    
    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Définit la valeur d'une jauge."""
        key = self._make_key(name, labels)
        self._gauges[key] = value
    
    def observe_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Ajoute une observation à un histogramme."""
        key = self._make_key(name, labels)
        self._histograms[key].add_observation(value)
    
    def record_timer(self, name: str, duration: float, labels: Optional[Dict[str, str]] = None):
        """Enregistre une durée."""
        key = self._make_key(name, labels)
        self._timers[key].append(duration)
        self.observe_histogram(f"{name}_duration", duration, labels)
    
    def _make_key(self, name: str, labels: Optional[Dict[str, str]]) -> str:
        """Crée une clé unique pour une métrique avec labels."""
        if not labels:
            return name
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}#{label_str}"
    
    # ==================== MÉTRIQUES RECHERCHE ====================
    
    def record_query(self, metrics: QueryMetrics):
        """Enregistre les métriques d'une requête."""
        if not self.include_query_metrics:
            return
        
        # Métriques globales
        self.search_metrics.total_queries += 1
        
        if metrics.result_count >= 0:  # Succès si pas d'erreur
            self.search_metrics.successful_queries += 1
        else:
            self.search_metrics.failed_queries += 1
        
        # Moyennes mobiles
        self._update_moving_average("avg_latency_seconds", metrics.latency_seconds)
        self._update_moving_average("avg_result_count", metrics.result_count)
        self._update_moving_average("avg_relevance_score", metrics.max_score)
        
        # Requêtes populaires
        query_key = metrics.query.lower().strip()
        self.search_metrics.popular_queries[query_key] = \
            self.search_metrics.popular_queries.get(query_key, 0) + 1
        
        # Activité utilisateur
        self.search_metrics.user_activity[metrics.user_id] = \
            self.search_metrics.user_activity.get(metrics.user_id, 0) + 1
        
        # Historique
        self._query_history.append(metrics)
        self._response_times.append(metrics.latency_seconds)
        
        # Métriques Prometheus
        self.increment_counter("search_queries_total", labels={
            "status": "success" if metrics.result_count >= 0 else "error"
        })
        self.observe_histogram("search_latency_seconds", metrics.latency_seconds)
        self.observe_histogram("search_result_count", metrics.result_count)
        self.set_gauge("search_max_score", metrics.max_score)
        
        # Cache metrics
        if metrics.cache_hit:
            self.increment_counter("search_cache_hits_total")
        else:
            self.increment_counter("search_cache_misses_total")
        
        # Vérifier les alertes
        self._check_alerts()
    
    def _update_moving_average(self, field: str, new_value: float, alpha: float = 0.1):
        """Met à jour une moyenne mobile exponentielle."""
        current = getattr(self.search_metrics, field)
        if current == 0:
            setattr(self.search_metrics, field, new_value)
        else:
            updated = current * (1 - alpha) + new_value * alpha
            setattr(self.search_metrics, field, updated)
    
    # ==================== MÉTRIQUES CACHE ====================
    
    def record_cache_hit(self, cache_name: str = "default"):
        """Enregistre un hit de cache."""
        if not self.include_cache_metrics:
            return
        
        self.increment_counter("cache_hits_total", labels={"cache": cache_name})
        self._update_cache_hit_rate()
    
    def record_cache_miss(self, cache_name: str = "default"):
        """Enregistre un miss de cache."""
        if not self.include_cache_metrics:
            return
        
        self.increment_counter("cache_misses_total", labels={"cache": cache_name})
        self._update_cache_hit_rate()
    
    def record_cache_eviction(self, cache_name: str = "default"):
        """Enregistre une éviction de cache."""
        if not self.include_cache_metrics:
            return
        
        self.increment_counter("cache_evictions_total", labels={"cache": cache_name})
    
    def _update_cache_hit_rate(self):
        """Met à jour le taux de hit du cache."""
        hits = self._counters.get("cache_hits_total", 0)
        misses = self._counters.get("cache_misses_total", 0)
        total = hits + misses
        
        if total > 0:
            self.search_metrics.cache_hit_rate = hits / total
        else:
            self.search_metrics.cache_hit_rate = 0.0
    
    # ==================== MÉTRIQUES PERFORMANCE ====================
    
    def update_performance_metrics(self, metrics: PerformanceMetrics):
        """Met à jour les métriques de performance."""
        if not self.include_performance_metrics:
            return
        
        self.performance_metrics = metrics
        
        # Métriques Prometheus
        self.set_gauge("requests_per_second", metrics.requests_per_second)
        self.set_gauge("response_time_avg", metrics.avg_response_time)
        self.set_gauge("response_time_p95", metrics.p95_response_time)
        self.set_gauge("response_time_p99", metrics.p99_response_time)
        self.set_gauge("active_connections", metrics.active_connections)
        self.set_gauge("memory_usage_mb", metrics.memory_usage_mb)
        self.set_gauge("cache_size", metrics.cache_size)
        self.set_gauge("cache_memory_mb", metrics.cache_memory_mb)
    
    def calculate_percentiles(self) -> Dict[str, float]:
        """Calcule les percentiles des temps de réponse."""
        if not self._response_times:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
        
        sorted_times = sorted(self._response_times)
        
        return {
            "p50": statistics.median(sorted_times),
            "p95": self._percentile(sorted_times, 0.95),
            "p99": self._percentile(sorted_times, 0.99)
        }
    
    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calcule un percentile."""
        if not data:
            return 0.0
        index = int(len(data) * percentile)
        return data[min(index, len(data) - 1)]
    
    # ==================== CONTEXTE TIMER ====================
    
    @contextmanager
    def time_query(self, query: str = "", user_id: int = 0):
        """Context manager pour mesurer le temps d'une requête."""
        start_time = time.time()
        timer = QueryTimer(self, query, user_id, start_time)
        
        try:
            yield timer
        finally:
            timer.finish()
    
    # ==================== ALERTES ====================
    
    def add_alert_callback(self, callback: Callable[[str, AlertLevel, Dict[str, Any]], None]):
        """Ajoute un callback d'alerte."""
        self._alert_callbacks.append(callback)
    
    def _check_alerts(self):
        """Vérifie les seuils d'alerte."""
        # Taux d'erreur
        if self.search_metrics.error_rate > self._alert_thresholds["error_rate"]:
            self._trigger_alert(
                "high_error_rate",
                AlertLevel.WARNING,
                {"error_rate": self.search_metrics.error_rate}
            )
        
        # Latence moyenne
        if self.search_metrics.avg_latency_seconds > self._alert_thresholds["avg_latency"]:
            self._trigger_alert(
                "high_latency",
                AlertLevel.WARNING,
                {"avg_latency": self.search_metrics.avg_latency_seconds}
            )
        
        # Taux de hit du cache
        if self.search_metrics.cache_hit_rate < self._alert_thresholds["cache_hit_rate"]:
            self._trigger_alert(
                "low_cache_hit_rate",
                AlertLevel.INFO,
                {"cache_hit_rate": self.search_metrics.cache_hit_rate}
            )
    
    def _trigger_alert(self, alert_type: str, level: AlertLevel, context: Dict[str, Any]):
        """Déclenche une alerte."""
        for callback in self._alert_callbacks:
            try:
                callback(alert_type, level, context)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")
    
    # ==================== EXPORT ====================
    
    def get_summary(self) -> Dict[str, Any]:
        """Retourne un résumé des métriques."""
        percentiles = self.calculate_percentiles()
        
        return {
            "service": self.service_name,
            "timestamp": datetime.utcnow().isoformat(),
            "search": {
                "total_queries": self.search_metrics.total_queries,
                "success_rate": self.search_metrics.success_rate,
                "avg_latency_seconds": self.search_metrics.avg_latency_seconds,
                "avg_result_count": self.search_metrics.avg_result_count,
                "cache_hit_rate": self.search_metrics.cache_hit_rate,
                "top_queries": list(self.search_metrics.popular_queries.items())[:10],
                "active_users": len(self.search_metrics.user_activity)
            },
            "performance": {
                "requests_per_second": self.performance_metrics.requests_per_second,
                "response_time_p50": percentiles["p50"],
                "response_time_p95": percentiles["p95"],
                "response_time_p99": percentiles["p99"],
                "memory_usage_mb": self.performance_metrics.memory_usage_mb,
                "elasticsearch_health": self.performance_metrics.elasticsearch_health
            }
        }
    
    def export_prometheus(self) -> str:
        """Exporte les métriques au format Prometheus."""
        lines = []
        
        # Commentaires
        lines.append(f"# HELP search_queries_total Total number of search queries")
        lines.append(f"# TYPE search_queries_total counter")
        
        # Compteurs
        for key, value in self._counters.items():
            metric_name, labels = self._parse_key(key)
            label_str = self._format_labels(labels) if labels else ""
            lines.append(f"{metric_name}{label_str} {value}")
        
        # Jauges
        for key, value in self._gauges.items():
            metric_name, labels = self._parse_key(key)
            label_str = self._format_labels(labels) if labels else ""
            lines.append(f"{metric_name}{label_str} {value}")
        
        # Histogrammes
        for key, histogram in self._histograms.items():
            metric_name, labels = self._parse_key(key)
            base_label_str = self._format_labels(labels) if labels else ""
            
            # Buckets
            for bucket in histogram.buckets:
                bucket_labels = (labels or {}).copy()
                bucket_labels["le"] = str(bucket.le)
                label_str = self._format_labels(bucket_labels)
                lines.append(f"{metric_name}_bucket{label_str} {bucket.count}")
            
            # Count et sum
            lines.append(f"{metric_name}_count{base_label_str} {histogram.count}")
            lines.append(f"{metric_name}_sum{base_label_str} {histogram.sum}")
        
        return "\n".join(lines)
    
    def _parse_key(self, key: str) -> tuple[str, Optional[Dict[str, str]]]:
        """Parse une clé de métrique avec labels."""
        if "#" not in key:
            return key, None
        
        name, label_str = key.split("#", 1)
        labels = {}
        
        for pair in label_str.split(","):
            if "=" in pair:
                k, v = pair.split("=", 1)
                labels[k] = v
        
        return name, labels
    
    def _format_labels(self, labels: Dict[str, str]) -> str:
        """Formate les labels pour Prometheus."""
        if not labels:
            return ""
        
        formatted = []
        for k, v in sorted(labels.items()):
            formatted.append(f'{k}="{v}"')
        
        return "{" + ",".join(formatted) + "}"
    
    async def shutdown(self):
        """Arrête le collecteur de métriques."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        logger.info(f"Metrics collector for {self.service_name} shutdown")

# ==================== TIMER ====================

class QueryTimer:
    """Timer pour mesurer les requêtes."""
    
    def __init__(self, collector: MetricsCollector, query: str, user_id: int, start_time: float):
        self.collector = collector
        self.query = query
        self.user_id = user_id
        self.start_time = start_time
        self.result_count = 0
        self.max_score = 0.0
        self.cache_hit = False
        self.elasticsearch_time: Optional[float] = None
        self.filters_applied: List[str] = []
    
    def record_results(self, result_count: int, max_score: float = 0.0):
        """Enregistre les résultats de la requête."""
        self.result_count = result_count
        self.max_score = max_score
    
    def record_cache_hit(self):
        """Marque que cette requête a utilisé le cache."""
        self.cache_hit = True
    
    def record_elasticsearch_time(self, duration: float):
        """Enregistre le temps Elasticsearch."""
        self.elasticsearch_time = duration
    
    def add_filter(self, filter_name: str):
        """Ajoute un filtre appliqué."""
        self.filters_applied.append(filter_name)
    
    def finish(self):
        """Termine le timer et enregistre les métriques."""
        duration = time.time() - self.start_time
        
        metrics = QueryMetrics(
            query=self.query,
            user_id=self.user_id,
            latency_seconds=duration,
            result_count=self.result_count,
            max_score=self.max_score,
            cache_hit=self.cache_hit,
            elasticsearch_time=self.elasticsearch_time,
            filters_applied=self.filters_applied
        )
        
        self.collector.record_query(metrics)

# ==================== FACTORY ====================

def create_metrics_collector(
    service_name: str = "search_service",
    **kwargs
) -> MetricsCollector:
    """Crée un collecteur de métriques configuré."""
    return MetricsCollector(service_name=service_name, **kwargs)

# ==================== EXPORTER ====================

class MetricsExporter:
    """Exporteur de métriques vers différents systèmes."""
    
    def __init__(self, collector: MetricsCollector):
        self.collector = collector
    
    async def export_to_file(self, filepath: str, format: str = "json"):
        """Exporte vers un fichier."""
        if format == "json":
            data = self.collector.get_summary()
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        elif format == "prometheus":
            data = self.collector.export_prometheus()
            with open(filepath, 'w') as f:
                f.write(data)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    async def export_to_http(self, url: str, format: str = "json"):
        """Exporte vers un endpoint HTTP."""
        import aiohttp
        
        if format == "json":
            data = self.collector.get_summary()
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=data) as response:
                    return response.status == 200
        else:
            raise ValueError(f"Unsupported format for HTTP: {format}")