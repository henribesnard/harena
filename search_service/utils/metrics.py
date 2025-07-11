"""
Système de métriques spécialisées pour le Search Service - Version Complète.

Ce module fournit une collecte de métriques complète pour :
- Performance des recherches
- Utilisation des ressources
- Qualité des résultats
- Monitoring système
- Alertes et seuils
- Export et analyse
"""

import time
import psutil
import threading
import json
import pickle
import gzip
from typing import Dict, Any, List, Optional, Union, Callable
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
import asyncio
from contextlib import contextmanager
from pathlib import Path
import statistics

logger = logging.getLogger(__name__)

# ==================== TYPES ET ENUMS ====================

class MetricType(Enum):
    """Types de métriques supportées."""
    COUNTER = "counter"
    HISTOGRAM = "histogram"
    GAUGE = "gauge"
    TIMER = "timer"
    RATE = "rate"
    SUMMARY = "summary"

class AlertLevel(Enum):
    """Niveaux d'alerte."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class TimeWindow(Enum):
    """Fenêtres temporelles pour les métriques."""
    MINUTE = 60
    HOUR = 3600
    DAY = 86400
    WEEK = 604800

@dataclass
class MetricPoint:
    """Point de métrique individuel."""
    value: Union[int, float]
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire."""
        return {
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "labels": self.labels
        }

@dataclass
class MetricSummary:
    """Résumé d'une métrique."""
    name: str
    type: MetricType
    current_value: Union[int, float]
    total_points: int
    labels: Dict[str, str] = field(default_factory=dict)
    statistics: Dict[str, float] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire."""
        return asdict(self)

@dataclass
class Alert:
    """Alerte système."""
    name: str
    level: AlertLevel
    message: str
    timestamp: datetime
    metric_name: str
    current_value: Union[int, float]
    threshold: Union[int, float]
    resolved: bool = False
    resolution_timestamp: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire."""
        return {
            "name": self.name,
            "level": self.level.value,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "metric_name": self.metric_name,
            "current_value": self.current_value,
            "threshold": self.threshold,
            "resolved": self.resolved,
            "resolution_timestamp": self.resolution_timestamp.isoformat() if self.resolution_timestamp else None
        }

@dataclass
class PerformanceProfile:
    """Profil de performance pour analyse."""
    name: str
    search_count: int
    avg_duration_ms: float
    p95_duration_ms: float
    p99_duration_ms: float
    error_rate: float
    cache_hit_rate: float
    avg_results_count: float
    timestamp: datetime = field(default_factory=datetime.utcnow)

# ==================== CLASSES MÉTRIQUES SPÉCIALISÉES ====================

class SearchMetrics:
    """
    Classe spécialisée pour les métriques de recherche.
    
    Interface simplifiée pour les opérations de recherche fréquentes.
    """
    
    def __init__(self, collector: Optional['MetricsCollector'] = None):
        self._collector = collector or MetricsCollector.get_instance()
        self._start_time = datetime.utcnow()
    
    def record_search_request(self, 
                            query: str,
                            user_id: str,
                            search_type: str = "lexical",
                            labels: Optional[Dict[str, str]] = None):
        """Enregistre une requête de recherche."""
        search_labels = {
            "search_type": search_type,
            "user_id": user_id,
            **(labels or {})
        }
        
        self._collector.increment_counter("search_requests_total", search_labels)
        self._collector.record_histogram("search_query_length", len(query.split()), search_labels)
    
    def record_search_duration(self, 
                             duration_ms: float,
                             user_id: str,
                             search_type: str = "lexical",
                             labels: Optional[Dict[str, str]] = None):
        """Enregistre la durée d'une recherche."""
        search_labels = {
            "search_type": search_type,
            "user_id": user_id,
            **(labels or {})
        }
        
        self._collector.record_histogram("search_duration_ms", duration_ms, search_labels)
        
        # Catégorisation des performances
        if duration_ms > 2000:
            self._collector.increment_counter("search_very_slow", search_labels)
        elif duration_ms > 1000:
            self._collector.increment_counter("search_slow", search_labels)
        else:
            self._collector.increment_counter("search_fast", search_labels)
    
    def record_search_results(self,
                            results_count: int,
                            total_available: int,
                            user_id: str,
                            search_type: str = "lexical",
                            labels: Optional[Dict[str, str]] = None):
        """Enregistre les résultats d'une recherche."""
        search_labels = {
            "search_type": search_type,
            "user_id": user_id,
            **(labels or {})
        }
        
        self._collector.record_histogram("search_results_count", results_count, search_labels)
        self._collector.record_histogram("search_total_available", total_available, search_labels)
        
        # Catégorisation de la qualité
        if results_count == 0:
            self._collector.increment_counter("search_no_results", search_labels)
        elif results_count < 5:
            self._collector.increment_counter("search_few_results", search_labels)
        else:
            self._collector.increment_counter("search_good_results", search_labels)
    
    def record_cache_operation(self,
                             cache_type: str,
                             operation: str,
                             hit: bool,
                             latency_ms: float = 0,
                             labels: Optional[Dict[str, str]] = None):
        """Enregistre une opération de cache."""
        cache_labels = {
            "cache_type": cache_type,
            "operation": operation,
            "hit": str(hit),
            **(labels or {})
        }
        
        if hit:
            self._collector.increment_counter("cache_hits", cache_labels)
        else:
            self._collector.increment_counter("cache_misses", cache_labels)
        
        if latency_ms > 0:
            self._collector.record_histogram("cache_latency_ms", latency_ms, cache_labels)
    
    def record_error(self,
                    error_type: str,
                    error_message: str,
                    user_id: str,
                    search_type: str = "lexical",
                    labels: Optional[Dict[str, str]] = None):
        """Enregistre une erreur de recherche."""
        error_labels = {
            "error_type": error_type,
            "search_type": search_type,
            "user_id": user_id,
            **(labels or {})
        }
        
        self._collector.increment_counter("search_errors_total", error_labels)
        logger.error(f"Search error: {error_type} - {error_message}", extra=error_labels)
    
    def get_search_stats(self, hours: int = 1) -> Dict[str, Any]:
        """Récupère les statistiques de recherche."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        # Cette méthode nécessiterait l'accès aux métriques internes
        # Pour simplifier, on retourne des stats de base
        return {
            "period_hours": hours,
            "timestamp": datetime.utcnow().isoformat(),
            "message": "Use MetricsCollector.get_all_metrics() for detailed stats"
        }

class QueryMetrics(SearchMetrics):
    """Alias pour compatibilité - metrics spécifiques aux requêtes."""
    pass

class PerformanceMetrics:
    """
    Métriques de performance spécialisées.
    """
    
    def __init__(self, collector: Optional['MetricsCollector'] = None):
        self._collector = collector or MetricsCollector.get_instance()
    
    def record_elasticsearch_query(self,
                                  query_type: str,
                                  index_name: str,
                                  execution_time_ms: float,
                                  shard_count: int = 1,
                                  success: bool = True,
                                  error_type: Optional[str] = None):
        """Enregistre une requête Elasticsearch."""
        labels = {
            "query_type": query_type,
            "index": index_name,
            "success": str(success)
        }
        
        if error_type:
            labels["error_type"] = error_type
        
        self._collector.record_histogram("elasticsearch_query_duration_ms", execution_time_ms, labels)
        self._collector.record_histogram("elasticsearch_shards", shard_count, labels)
        
        if success:
            self._collector.increment_counter("elasticsearch_queries_success", labels)
        else:
            self._collector.increment_counter("elasticsearch_queries_error", labels)
    
    def record_api_request(self,
                          endpoint: str,
                          method: str,
                          status_code: int,
                          duration_ms: float,
                          user_id: Optional[str] = None):
        """Enregistre une requête API."""
        labels = {
            "endpoint": endpoint,
            "method": method,
            "status_code": str(status_code)
        }
        
        if user_id:
            labels["user_id"] = user_id
        
        self._collector.record_histogram("api_request_duration_ms", duration_ms, labels)
        self._collector.increment_counter("api_requests_total", labels)
        
        if status_code >= 400:
            self._collector.increment_counter("api_requests_error", labels)

class MetricsExporter:
    """
    Exporteur de métriques vers différents formats.
    """
    
    def __init__(self, collector: Optional['MetricsCollector'] = None):
        self._collector = collector or MetricsCollector.get_instance()
    
    async def export_prometheus(self) -> str:
        """Exporte les métriques au format Prometheus."""
        metrics = await self._collector.get_all_metrics()
        
        prometheus_output = []
        prometheus_output.append("# HELP search_service_metrics Search Service Metrics")
        prometheus_output.append("# TYPE search_service_metrics gauge")
        
        # Export des compteurs
        for name, value in metrics.get("counters", {}).items():
            prometheus_output.append(f"search_service_{name} {value}")
        
        # Export des jauges
        for name, value in metrics.get("gauges", {}).items():
            prometheus_output.append(f"search_service_{name} {value}")
        
        return "\n".join(prometheus_output)
    
    async def export_json(self) -> str:
        """Exporte les métriques au format JSON."""
        metrics = await self._collector.get_all_metrics()
        return json.dumps(metrics, indent=2, default=str)
    
    async def export_csv(self) -> str:
        """Exporte les métriques au format CSV."""
        metrics = await self._collector.get_all_metrics()
        
        csv_lines = ["metric_name,metric_type,current_value,timestamp"]
        
        for name, summary in metrics.get("metrics_summary", {}).items():
            csv_lines.append(
                f"{name},{summary.get('type', 'unknown')},"
                f"{summary.get('current', 0)},{summary.get('last_updated', '')}"
            )
        
        return "\n".join(csv_lines)

# ==================== COLLECTEUR DE MÉTRIQUES ====================

class MetricsCollector:
    """
    Collecteur de métriques thread-safe avec support d'alertes et d'export.
    
    Implémente le pattern Singleton pour une instance globale.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __init__(self, 
                 retention_hours: int = 24, 
                 max_points_per_metric: int = 10000,
                 auto_export: bool = True,
                 export_interval_minutes: int = 60):
        if MetricsCollector._instance is not None:
            raise RuntimeError("MetricsCollector est un singleton. Utilisez get_instance()")
        
        self.retention_hours = retention_hours
        self.max_points_per_metric = max_points_per_metric
        self.auto_export = auto_export
        self.export_interval_minutes = export_interval_minutes
        
        # Stockage des métriques
        self._metrics: Dict[str, List[MetricPoint]] = defaultdict(list)
        self._metric_types: Dict[str, MetricType] = {}
        self._counters: Dict[str, int] = defaultdict(int)
        self._gauges: Dict[str, float] = {}
        self._timers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._rates: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Système d'alertes
        self._alert_rules: Dict[str, Dict] = {}
        self._active_alerts: List[Alert] = []
        self._alert_history: List[Alert] = []
        self._alert_callbacks: List[Callable[[Alert], None]] = []
        
        # Export et persistance
        self._export_directory = Path("metrics_exports")
        self._export_directory.mkdir(exist_ok=True)
        self._last_export = datetime.utcnow()
        
        # Thread de maintenance
        self._cleanup_thread = None
        self._running = True
        self._start_maintenance_thread()
        
        # Métriques système
        self._last_system_check = 0
        self._system_metrics_interval = 60  # 1 minute
        
        # Profils de performance
        self._performance_profiles: Dict[str, PerformanceProfile] = {}
        
        logger.info("MetricsCollector initialisé avec persistance")
    
    @classmethod
    def get_instance(cls, **kwargs) -> 'MetricsCollector':
        """Récupère l'instance singleton."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(**kwargs)
        return cls._instance
    
    def _start_maintenance_thread(self):
        """Démarre le thread de maintenance automatique."""
        def maintenance_worker():
            while self._running:
                try:
                    self._cleanup_old_metrics()
                    self._collect_system_metrics()
                    self._check_alerts()
                    self._update_performance_profiles()
                    
                    # Export automatique
                    if self.auto_export and self._should_export():
                        self._auto_export_metrics()
                    
                    time.sleep(60)  # 1 minute
                except Exception as e:
                    logger.error(f"Erreur dans le thread de maintenance: {e}")
        
        self._maintenance_thread = threading.Thread(target=maintenance_worker, daemon=True)
        self._maintenance_thread.start()
    
    # ==================== MÉTRIQUES DE BASE ====================
    
    def increment_counter(self, name: str, labels: Optional[Dict[str, str]] = None, value: int = 1):
        """Incrémente un compteur."""
        with self._lock:
            key = self._get_metric_key(name, labels)
            self._counters[key] += value
            self._metric_types[name] = MetricType.COUNTER
            
            point = MetricPoint(
                value=self._counters[key],
                timestamp=datetime.utcnow(),
                labels=labels or {}
            )
            self._add_metric_point(name, point)
    
    def decrement_counter(self, name: str, labels: Optional[Dict[str, str]] = None, value: int = 1):
        """Décrémente un compteur."""
        self.increment_counter(name, labels, -value)
    
    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Définit la valeur d'une jauge."""
        with self._lock:
            key = self._get_metric_key(name, labels)
            self._gauges[key] = value
            self._metric_types[name] = MetricType.GAUGE
            
            point = MetricPoint(
                value=value,
                timestamp=datetime.utcnow(),
                labels=labels or {}
            )
            self._add_metric_point(name, point)
    
    def record_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Enregistre une valeur dans un histogramme."""
        with self._lock:
            self._metric_types[name] = MetricType.HISTOGRAM
            
            point = MetricPoint(
                value=value,
                timestamp=datetime.utcnow(),
                labels=labels or {}
            )
            self._add_metric_point(name, point)
    
    def record_rate(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Enregistre un taux (événements par seconde)."""
        with self._lock:
            key = self._get_metric_key(name, labels)
            self._rates[key].append((time.time(), value))
            self._metric_types[name] = MetricType.RATE
            
            point = MetricPoint(
                value=value,
                timestamp=datetime.utcnow(),
                labels=labels or {}
            )
            self._add_metric_point(name, point)
    
    @contextmanager
    def timer(self, name: str, labels: Optional[Dict[str, str]] = None):
        """Context manager pour mesurer le temps d'exécution."""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.record_timer(name, duration, labels)
    
    def record_timer(self, name: str, duration: float, labels: Optional[Dict[str, str]] = None):
        """Enregistre une durée."""
        with self._lock:
            key = self._get_metric_key(name, labels)
            self._timers[key].append(duration)
            self._metric_types[name] = MetricType.TIMER
            
            point = MetricPoint(
                value=duration,
                timestamp=datetime.utcnow(),
                labels=labels or {}
            )
            self._add_metric_point(name, point)
    
    # ==================== MÉTRIQUES SPÉCIALISÉES SEARCH ====================
    
    def record_search_operation(self, 
                              query: str,
                              results_count: int,
                              total_results: int,
                              search_time_ms: float,
                              user_id: str,
                              cache_hit: bool = False,
                              error: Optional[str] = None):
        """Enregistre une opération de recherche complète."""
        labels = {
            "user_id": user_id, 
            "cache_hit": str(cache_hit),
            "has_error": str(error is not None)
        }
        
        # Métriques de base
        self.increment_counter("search_requests_total", labels)
        self.record_histogram("search_duration_ms", search_time_ms, labels)
        self.record_histogram("search_results_count", results_count, labels)
        self.record_histogram("search_total_results", total_results, labels)
        
        # Métriques de qualité
        if results_count == 0:
            self.increment_counter("search_no_results", labels)
        elif results_count < 5:
            self.increment_counter("search_few_results", labels)
        
        # Métriques de performance
        if search_time_ms > 1000:
            self.increment_counter("search_slow_requests", labels)
        elif search_time_ms > 2000:
            self.increment_counter("search_very_slow_requests", labels)
        
        # Longueur de requête
        query_length = len(query.split())
        self.record_histogram("search_query_length", query_length, labels)
        
        # Cache
        if cache_hit:
            self.increment_counter("search_cache_hits", labels)
        else:
            self.increment_counter("search_cache_misses", labels)
        
        # Erreurs
        if error:
            error_labels = {**labels, "error_type": error}
            self.increment_counter("search_errors_total", error_labels)
        
        # Taux de recherche par utilisateur
        self.record_rate(f"search_rate_user_{user_id}", 1, labels)
    
    def record_elasticsearch_operation(self,
                                     operation: str,
                                     index_name: str,
                                     query_time_ms: float,
                                     shard_count: int,
                                     success: bool = True,
                                     error_type: Optional[str] = None):
        """Enregistre une opération Elasticsearch."""
        labels = {
            "operation": operation,
            "index": index_name, 
            "success": str(success)
        }
        
        if error_type:
            labels["error_type"] = error_type
        
        self.record_histogram("elasticsearch_query_time_ms", query_time_ms, labels)
        self.record_histogram("elasticsearch_shard_count", shard_count, labels)
        
        if success:
            self.increment_counter("elasticsearch_operations_success", labels)
        else:
            self.increment_counter("elasticsearch_operations_error", labels)
    
    def record_cache_operation(self,
                             cache_type: str,
                             operation: str,
                             hit: bool,
                             latency_ms: float,
                             key_size: Optional[int] = None,
                             value_size: Optional[int] = None):
        """Enregistre une opération de cache."""
        labels = {
            "cache_type": cache_type, 
            "operation": operation,
            "hit": str(hit)
        }
        
        self.record_histogram("cache_operation_latency_ms", latency_ms, labels)
        
        if key_size:
            self.record_histogram("cache_key_size_bytes", key_size, labels)
        if value_size:
            self.record_histogram("cache_value_size_bytes", value_size, labels)
        
        if hit:
            self.increment_counter("cache_hits", labels)
        else:
            self.increment_counter("cache_misses", labels)
    
    # ==================== MÉTRIQUES SYSTÈME ====================
    
    def _collect_system_metrics(self):
        """Collecte les métriques système."""
        try:
            now = time.time()
            if now - self._last_system_check < self._system_metrics_interval:
                return
            
            # CPU et processus
            cpu_percent = psutil.cpu_percent(interval=None)
            self.set_gauge("system_cpu_percent", cpu_percent)
            
            load_avg = psutil.getloadavg()
            self.set_gauge("system_load_1m", load_avg[0])
            self.set_gauge("system_load_5m", load_avg[1])
            self.set_gauge("system_load_15m", load_avg[2])
            
            # Mémoire système
            memory = psutil.virtual_memory()
            self.set_gauge("system_memory_percent", memory.percent)
            self.set_gauge("system_memory_available_gb", memory.available / 1024**3)
            self.set_gauge("system_memory_used_gb", memory.used / 1024**3)
            
            # Mémoire swap
            swap = psutil.swap_memory()
            self.set_gauge("system_swap_percent", swap.percent)
            
            # Disque
            disk = psutil.disk_usage('/')
            self.set_gauge("system_disk_percent", disk.percent)
            self.set_gauge("system_disk_free_gb", disk.free / 1024**3)
            self.set_gauge("system_disk_used_gb", disk.used / 1024**3)
            
            # Réseau
            net_io = psutil.net_io_counters()
            self.set_gauge("system_network_bytes_sent", net_io.bytes_sent)
            self.set_gauge("system_network_bytes_recv", net_io.bytes_recv)
            
            # Processus courant
            process = psutil.Process()
            proc_memory = process.memory_info()
            self.set_gauge("process_memory_rss_mb", proc_memory.rss / 1024**2)
            self.set_gauge("process_memory_vms_mb", proc_memory.vms / 1024**2)
            self.set_gauge("process_cpu_percent", process.cpu_percent())
            
            # Threads et descripteurs de fichiers
            self.set_gauge("process_threads", process.num_threads())
            try:
                self.set_gauge("process_open_files", len(process.open_files()))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            
            # Connexions réseau
            try:
                connections = len(process.connections())
                self.set_gauge("process_network_connections", connections)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            
            self._last_system_check = now
            
        except Exception as e:
            logger.warning(f"Erreur lors de la collecte des métriques système: {e}")
    
    # ==================== PROFILS DE PERFORMANCE ====================
    
    def _update_performance_profiles(self):
        """Met à jour les profils de performance."""
        try:
            # Profil général des recherches
            search_points = self._metrics.get("search_duration_ms", [])
            if len(search_points) > 10:  # Au moins 10 points pour un profil valide
                durations = [p.value for p in search_points[-1000:]]  # Derniers 1000 points
                
                profile = PerformanceProfile(
                    name="search_general",
                    search_count=len(durations),
                    avg_duration_ms=statistics.mean(durations),
                    p95_duration_ms=statistics.quantiles(durations, n=20)[18],  # 95e percentile
                    p99_duration_ms=statistics.quantiles(durations, n=100)[98],  # 99e percentile
                    error_rate=self._calculate_error_rate(),
                    cache_hit_rate=self._calculate_cache_hit_rate(),
                    avg_results_count=self._calculate_avg_results_count()
                )
                
                self._performance_profiles["search_general"] = profile
                
        except Exception as e:
            logger.warning(f"Erreur lors de la mise à jour des profils: {e}")
    
    def _calculate_error_rate(self) -> float:
        """Calcule le taux d'erreur des recherches."""
        total_searches = self._counters.get("search_requests_total", 0)
        total_errors = self._counters.get("search_errors_total", 0)
        
        if total_searches == 0:
            return 0.0
        
        return (total_errors / total_searches) * 100
    
    def _calculate_cache_hit_rate(self) -> float:
        """Calcule le taux de succès du cache."""
        cache_hits = self._counters.get("search_cache_hits", 0)
        cache_misses = self._counters.get("search_cache_misses", 0)
        
        total_cache_ops = cache_hits + cache_misses
        if total_cache_ops == 0:
            return 0.0
        
        return (cache_hits / total_cache_ops) * 100
    
    def _calculate_avg_results_count(self) -> float:
        """Calcule le nombre moyen de résultats."""
        result_points = self._metrics.get("search_results_count", [])
        if not result_points:
            return 0.0
        
        recent_points = result_points[-1000:]  # Derniers 1000 points
        return statistics.mean([p.value for p in recent_points])
    
    # ==================== SYSTÈME D'ALERTES AVANCÉ ====================
    
    def add_alert_rule(self,
                      metric_name: str,
                      threshold: Union[int, float],
                      condition: str = "greater_than",
                      level: AlertLevel = AlertLevel.WARNING,
                      message_template: str = None,
                      window_minutes: int = 5,
                      min_points: int = 3):
        """Ajoute une règle d'alerte avancée."""
        rule = {
            "threshold": threshold,
            "condition": condition,
            "level": level,
            "message_template": message_template or f"{metric_name} {condition} {threshold}",
            "window_minutes": window_minutes,
            "min_points": min_points,
            "enabled": True,
            "last_triggered": None
        }
        self._alert_rules[metric_name] = rule
        logger.info(f"Règle d'alerte avancée ajoutée pour {metric_name}")
    
    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """Ajoute un callback pour les alertes."""
        self._alert_callbacks.append(callback)
    
    def _check_alerts(self):
        """Vérifie les règles d'alerte avec fenêtres temporelles."""
        try:
            current_time = datetime.utcnow()
            
            for metric_name, rule in self._alert_rules.items():
                if not rule["enabled"]:
                    continue
                
                # Évite les alertes trop fréquentes
                if (rule["last_triggered"] and 
                    (current_time - rule["last_triggered"]).total_seconds() < 300):  # 5 minutes
                    continue
                
                should_alert = self._evaluate_windowed_alert(metric_name, rule, current_time)
                
                if should_alert:
                    self._trigger_alert(metric_name, rule, 0)  # Valeur sera calculée dans _trigger_alert
                    rule["last_triggered"] = current_time
                else:
                    self._resolve_alert(metric_name)
                    
        except Exception as e:
            logger.error(f"Erreur lors de la vérification des alertes: {e}")
    
    def _evaluate_windowed_alert(self, metric_name: str, rule: Dict, current_time: datetime) -> bool:
        """Évalue une alerte sur une fenêtre temporelle."""
        window_start = current_time - timedelta(minutes=rule["window_minutes"])
        
        # Récupère les points dans la fenêtre
        points = [
            p for p in self._metrics.get(metric_name, [])
            if p.timestamp >= window_start
        ]
        
        if len(points) < rule["min_points"]:
            return False
        
        # Calcule la valeur pour la fenêtre (moyenne par défaut)
        values = [p.value for p in points]
        window_value = statistics.mean(values)
        
        return self._evaluate_alert_condition(window_value, rule["threshold"], rule["condition"])
    
    def _trigger_alert(self, metric_name: str, rule: Dict, current_value: float):
        """Déclenche une alerte avec callbacks."""
        existing_alert = next(
            (a for a in self._active_alerts if a.metric_name == metric_name and not a.resolved),
            None
        )
        
        if existing_alert:
            return
        
        alert = Alert(
            name=f"alert_{metric_name}_{int(time.time())}",
            level=rule["level"],
            message=rule["message_template"].format(
                metric_name=metric_name,
                current_value=current_value,
                threshold=rule["threshold"]
            ),
            timestamp=datetime.utcnow(),
            metric_name=metric_name,
            current_value=current_value,
            threshold=rule["threshold"]
        )
        
        self._active_alerts.append(alert)
        self._alert_history.append(alert)
        
        # Callbacks
        for callback in self._alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Erreur dans callback d'alerte: {e}")
        
        logger.warning(
            f"ALERTE {rule['level'].value.upper()}: {alert.message}",
            extra=alert.to_dict()
        )
    
    # ==================== EXPORT ET PERSISTANCE ====================
    
    def _should_export(self) -> bool:
        """Détermine s'il faut faire un export automatique."""
        time_since_export = datetime.utcnow() - self._last_export
        return time_since_export.total_seconds() > (self.export_interval_minutes * 60)
    
    def _auto_export_metrics(self):
        """Export automatique des métriques."""
        try:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"metrics_export_{timestamp}.json.gz"
            filepath = self._export_directory / filename
            
            self.export_to_file(str(filepath), compress=True)
            self._last_export = datetime.utcnow()
            
            # Nettoyage des anciens exports (garde 7 jours)
            self._cleanup_old_exports()
            
        except Exception as e:
            logger.error(f"Erreur lors de l'export automatique: {e}")
    
    def export_to_file(self, filepath: str, compress: bool = True):
        """Exporte les métriques vers un fichier."""
        try:
            data = asyncio.run(self.get_all_metrics())
            
            if compress:
                with gzip.open(filepath, 'wt', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, default=str)
            else:
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, default=str)
            
            logger.info(f"Métriques exportées vers {filepath}")
            
        except Exception as e:
            logger.error(f"Erreur lors de l'export vers {filepath}: {e}")
    
    def _cleanup_old_exports(self, days_to_keep: int = 7):
        """Nettoie les anciens fichiers d'export."""
        try:
            cutoff_time = datetime.utcnow() - timedelta(days=days_to_keep)
            
            for file_path in self._export_directory.glob("metrics_export_*.json*"):
                if file_path.stat().st_mtime < cutoff_time.timestamp():
                    file_path.unlink()
                    logger.debug(f"Ancien export supprimé: {file_path}")
                    
        except Exception as e:
            logger.warning(f"Erreur lors du nettoyage des exports: {e}")
    
    # ==================== API PUBLIQUE ÉTENDUE ====================
    
    async def get_all_metrics(self) -> Dict[str, Any]:
        """Récupère toutes les métriques avec données enrichies."""
        with self._lock:
            result = {
                "timestamp": datetime.utcnow().isoformat(),
                "service": "search_service",
                "version": "1.0.0",
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "metrics_summary": {},
                "performance_profiles": {
                    name: asdict(profile) for name, profile in self._performance_profiles.items()
                },
                "active_alerts": [alert.to_dict() for alert in self._active_alerts if not alert.resolved],
                "alert_summary": {
                    "total_active": len([a for a in self._active_alerts if not a.resolved]),
                    "by_level": self._get_alerts_by_level(),
                    "last_24h_count": len([
                        a for a in self._alert_history 
                        if (datetime.utcnow() - a.timestamp).total_seconds() < 86400
                    ])
                },
                "system_info": {
                    "retention_hours": self.retention_hours,
                    "total_metrics": len(self._metrics),
                    "total_points": sum(len(points) for points in self._metrics.values()),
                    "uptime_seconds": self._get_uptime_seconds(),
                    "memory_usage_mb": self._get_memory_usage_mb()
                }
            }
            
            # Résumés de métriques avec statistiques avancées
            for metric_name, points in self._metrics.items():
                if not points:
                    continue
                
                values = [p.value for p in points]
                summary = self._calculate_advanced_statistics(values)
                summary.update({
                    "type": self._metric_types.get(metric_name, MetricType.HISTOGRAM).value,
                    "count": len(values),
                    "last_updated": points[-1].timestamp.isoformat() if points else None,
                    "first_recorded": points[0].timestamp.isoformat() if points else None
                })
                
                result["metrics_summary"][metric_name] = summary
            
            return result
    
    def _calculate_advanced_statistics(self, values: List[float]) -> Dict[str, float]:
        """Calcule des statistiques avancées pour une série de valeurs."""
        if not values:
            return {}
        
        try:
            sorted_values = sorted(values)
            n = len(values)
            
            stats = {
                "current": values[-1],
                "min": min(values),
                "max": max(values),
                "avg": statistics.mean(values),
                "median": statistics.median(values),
                "std_dev": statistics.stdev(values) if n > 1 else 0,
                "sum": sum(values)
            }
            
            # Percentiles
            if n >= 4:
                stats.update({
                    "p25": statistics.quantiles(sorted_values, n=4)[0],
                    "p75": statistics.quantiles(sorted_values, n=4)[2],
                    "p90": statistics.quantiles(sorted_values, n=10)[8],
                    "p95": statistics.quantiles(sorted_values, n=20)[18],
                    "p99": statistics.quantiles(sorted_values, n=100)[98] if n >= 100 else sorted_values[-1]
                })
            
            # Tendance (pente de régression linéaire simple)
            if n >= 2:
                x_values = list(range(n))
                slope = self._calculate_slope(x_values, values)
                stats["trend_slope"] = slope
                stats["trend_direction"] = "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable"
            
            return stats
            
        except Exception as e:
            logger.warning(f"Erreur lors du calcul des statistiques: {e}")
            return {
                "current": values[-1] if values else 0,
                "min": min(values) if values else 0,
                "max": max(values) if values else 0,
                "avg": statistics.mean(values) if values else 0,
                "count": len(values)
            }
    
    def _calculate_slope(self, x_values: List[int], y_values: List[float]) -> float:
        """Calcule la pente d'une régression linéaire simple."""
        try:
            n = len(x_values)
            sum_x = sum(x_values)
            sum_y = sum(y_values)
            sum_xy = sum(x * y for x, y in zip(x_values, y_values))
            sum_x2 = sum(x * x for x in x_values)
            
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            return slope
        except ZeroDivisionError:
            return 0.0
    
    def _get_alerts_by_level(self) -> Dict[str, int]:
        """Compte les alertes actives par niveau."""
        counts = {level.value: 0 for level in AlertLevel}
        for alert in self._active_alerts:
            if not alert.resolved:
                counts[alert.level.value] += 1
        return counts
    
    def _get_uptime_seconds(self) -> float:
        """Calcule le temps de fonctionnement en secondes."""
        # Approximation basée sur la première métrique enregistrée
        if self._metrics:
            first_timestamps = []
            for points in self._metrics.values():
                if points:
                    first_timestamps.append(points[0].timestamp)
            
            if first_timestamps:
                start_time = min(first_timestamps)
                return (datetime.utcnow() - start_time).total_seconds()
        
        return 0.0
    
    def _get_memory_usage_mb(self) -> float:
        """Calcule l'utilisation mémoire du collecteur en MB."""
        try:
            import sys
            
            # Taille approximative des structures de données
            metrics_size = sys.getsizeof(self._metrics)
            counters_size = sys.getsizeof(self._counters)
            gauges_size = sys.getsizeof(self._gauges)
            
            # Taille des points de métriques
            points_size = 0
            for points_list in self._metrics.values():
                points_size += sys.getsizeof(points_list)
                for point in points_list:
                    points_size += sys.getsizeof(point)
            
            total_bytes = metrics_size + counters_size + gauges_size + points_size
            return total_bytes / (1024 * 1024)  # Conversion en MB
            
        except Exception:
            return 0.0
    
    # ==================== UTILITAIRES ====================
    
    def _get_metric_key(self, name: str, labels: Optional[Dict[str, str]]) -> str:
        """Génère une clé unique pour une métrique avec labels."""
        if not labels:
            return name
        
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}:{label_str}"
    
    def _add_metric_point(self, name: str, point: MetricPoint):
        """Ajoute un point de métrique avec rotation automatique."""
        self._metrics[name].append(point)
        
        # Limite le nombre de points
        if len(self._metrics[name]) > self.max_points_per_metric:
            self._metrics[name] = self._metrics[name][-self.max_points_per_metric:]
    
    def _cleanup_old_metrics(self):
        """Nettoie les anciennes métriques et optimise la mémoire."""
        cutoff_time = datetime.utcnow() - timedelta(hours=self.retention_hours)
        cleaned_count = 0
        
        with self._lock:
            for metric_name in list(self._metrics.keys()):
                original_count = len(self._metrics[metric_name])
                
                self._metrics[metric_name] = [
                    point for point in self._metrics[metric_name]
                    if point.timestamp > cutoff_time
                ]
                
                cleaned_count += original_count - len(self._metrics[metric_name])
                
                # Supprime les métriques complètement vides
                if not self._metrics[metric_name]:
                    del self._metrics[metric_name]
                    if metric_name in self._metric_types:
                        del self._metric_types[metric_name]
        
        # Nettoie l'historique des alertes
        original_alert_count = len(self._alert_history)
        self._alert_history = [
            alert for alert in self._alert_history
            if alert.timestamp > cutoff_time
        ]
        
        if cleaned_count > 0 or original_alert_count != len(self._alert_history):
            logger.debug(
                f"Nettoyage: {cleaned_count} points de métriques et "
                f"{original_alert_count - len(self._alert_history)} alertes supprimés"
            )
    
    def _evaluate_alert_condition(self, value: float, threshold: float, condition: str) -> bool:
        """Évalue une condition d'alerte."""
        conditions = {
            "greater_than": value > threshold,
            "less_than": value < threshold,
            "equal": value == threshold,
            "greater_equal": value >= threshold,
            "less_equal": value <= threshold,
            "not_equal": value != threshold
        }
        return conditions.get(condition, False)
    
    def _resolve_alert(self, metric_name: str):
        """Résout une alerte active."""
        for alert in self._active_alerts:
            if alert.metric_name == metric_name and not alert.resolved:
                alert.resolved = True
                alert.resolution_timestamp = datetime.utcnow()
                logger.info(f"Alerte résolue: {alert.name}")
    
    def reset_metrics(self, metric_names: Optional[List[str]] = None):
        """Remet à zéro des métriques spécifiques ou toutes."""
        with self._lock:
            if metric_names is None:
                # Reset complet
                self._metrics.clear()
                self._counters.clear()
                self._gauges.clear()
                self._timers.clear()
                self._rates.clear()
                self._active_alerts.clear()
                self._performance_profiles.clear()
                logger.info("Toutes les métriques ont été remises à zéro")
            else:
                # Reset sélectif
                for metric_name in metric_names:
                    self._metrics.pop(metric_name, None)
                    self._metric_types.pop(metric_name, None)
                    
                    # Reset des compteurs
                    keys_to_remove = [k for k in self._counters.keys() if k.startswith(metric_name)]
                    for key in keys_to_remove:
                        del self._counters[key]
                    
                    # Reset des jauges
                    keys_to_remove = [k for k in self._gauges.keys() if k.startswith(metric_name)]
                    for key in keys_to_remove:
                        del self._gauges[key]
                
                logger.info(f"Métriques remises à zéro: {metric_names}")
    
    def shutdown(self):
        """Arrête proprement le collecteur de métriques."""
        logger.info("Arrêt du MetricsCollector en cours...")
        
        self._running = False
        
        # Export final des métriques
        if self.auto_export:
            try:
                final_export_path = self._export_directory / f"final_export_{int(time.time())}.json.gz"
                self.export_to_file(str(final_export_path), compress=True)
                logger.info("Export final des métriques effectué")
            except Exception as e:
                logger.error(f"Erreur lors de l'export final: {e}")
        
        # Attendre l'arrêt du thread de maintenance
        if self._maintenance_thread and self._maintenance_thread.is_alive():
            self._maintenance_thread.join(timeout=10)
        
        logger.info("MetricsCollector arrêté proprement")

# ==================== DÉCORATEURS UTILITAIRES ====================

def measure_time(metric_name: str, labels: Optional[Dict[str, str]] = None):
    """Décorateur pour mesurer le temps d'exécution d'une fonction."""
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                collector = MetricsCollector.get_instance()
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    return result
                finally:
                    duration = (time.time() - start_time) * 1000  # en ms
                    collector.record_histogram(metric_name, duration, labels)
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                collector = MetricsCollector.get_instance()
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    duration = (time.time() - start_time) * 1000  # en ms
                    collector.record_histogram(metric_name, duration, labels)
            return sync_wrapper
    return decorator

def count_calls(metric_name: str, labels: Optional[Dict[str, str]] = None):
    """Décorateur pour compter les appels à une fonction."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            collector = MetricsCollector.get_instance()
            collector.increment_counter(metric_name, labels)
            return func(*args, **kwargs)
        return wrapper
    return decorator

def track_errors(metric_name: str, labels: Optional[Dict[str, str]] = None):
    """Décorateur pour traquer les erreurs d'une fonction."""
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                collector = MetricsCollector.get_instance()
                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception as e:
                    error_labels = {**(labels or {}), "error_type": type(e).__name__}
                    collector.increment_counter(metric_name, error_labels)
                    raise
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                collector = MetricsCollector.get_instance()
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    error_labels = {**(labels or {}), "error_type": type(e).__name__}
                    collector.increment_counter(metric_name, error_labels)
                    raise
            return sync_wrapper
    return decorator

# ==================== CONFIGURATION DES ALERTES PAR DÉFAUT ====================

def setup_default_alerts(collector: MetricsCollector):
    """Configure les alertes par défaut pour le service de recherche."""
    
    # Alertes de performance des recherches
    collector.add_alert_rule(
        "search_duration_ms",
        threshold=1000,
        condition="greater_than",
        level=AlertLevel.WARNING,
        message_template="Temps de recherche élevé: {current_value:.2f}ms (seuil: {threshold}ms)",
        window_minutes=5,
        min_points=3
    )
    
    collector.add_alert_rule(
        "search_duration_ms",
        threshold=2000,
        condition="greater_than",
        level=AlertLevel.ERROR,
        message_template="Temps de recherche très élevé: {current_value:.2f}ms (seuil: {threshold}ms)",
        window_minutes=3,
        min_points=2
    )
    
    # Alertes de qualité
    collector.add_alert_rule(
        "search_no_results",
        threshold=50,
        condition="greater_than",
        level=AlertLevel.WARNING,
        message_template="Trop de recherches sans résultats: {current_value} (seuil: {threshold})",
        window_minutes=10,
        min_points=5
    )
    
    # Alertes système
    collector.add_alert_rule(
        "system_memory_percent",
        threshold=85,
        condition="greater_than",
        level=AlertLevel.WARNING,
        message_template="Utilisation mémoire élevée: {current_value:.1f}% (seuil: {threshold}%)",
        window_minutes=5,
        min_points=3
    )
    
    collector.add_alert_rule(
        "system_cpu_percent",
        threshold=80,
        condition="greater_than",
        level=AlertLevel.WARNING,
        message_template="Utilisation CPU élevée: {current_value:.1f}% (seuil: {threshold}%)",
        window_minutes=5,
        min_points=3
    )
    
    # Alertes de cache
    collector.add_alert_rule(
        "search_cache_hits",
        threshold=0.7,  # 70% de cache hit rate minimum
        condition="less_than",
        level=AlertLevel.WARNING,
        message_template="Taux de cache faible: {current_value:.1%} (seuil: {threshold:.1%})",
        window_minutes=15,
        min_points=10
    )
    
    logger.info("Alertes par défaut configurées")

# ==================== CALLBACKS D'ALERTE PAR DÉFAUT ====================

def log_alert_callback(alert: Alert):
    """Callback par défaut pour logger les alertes."""
    level_mapping = {
        AlertLevel.INFO: logging.INFO,
        AlertLevel.WARNING: logging.WARNING,
        AlertLevel.ERROR: logging.ERROR,
        AlertLevel.CRITICAL: logging.CRITICAL
    }
    
    logger.log(
        level_mapping.get(alert.level, logging.WARNING),
        f"ALERTE {alert.level.value.upper()}: {alert.message}",
        extra={
            "alert_name": alert.name,
            "metric_name": alert.metric_name,
            "current_value": alert.current_value,
            "threshold": alert.threshold,
            "timestamp": alert.timestamp.isoformat()
        }
    )

def email_alert_callback(alert: Alert):
    """Callback pour envoyer des alertes par email (à implémenter)."""
    # Implémentation d'envoi d'email selon vos besoins
    if alert.level in [AlertLevel.ERROR, AlertLevel.CRITICAL]:
        logger.info(f"Email d'alerte envoyé pour: {alert.name}")

# ==================== FONCTIONS FACTORY ====================

def create_metrics_collector(**kwargs) -> MetricsCollector:
    """Crée ou récupère l'instance du collecteur de métriques."""
    return MetricsCollector.get_instance(**kwargs)

def create_search_metrics(collector: Optional[MetricsCollector] = None) -> SearchMetrics:
    """Crée une instance de SearchMetrics."""
    return SearchMetrics(collector)

def create_performance_metrics(collector: Optional[MetricsCollector] = None) -> PerformanceMetrics:
    """Crée une instance de PerformanceMetrics."""
    return PerformanceMetrics(collector)

def create_metrics_exporter(collector: Optional[MetricsCollector] = None) -> MetricsExporter:
    """Crée une instance de MetricsExporter."""
    return MetricsExporter(collector)

# ==================== INSTANCE GLOBALE ====================

def get_metrics_collector() -> MetricsCollector:
    """Récupère l'instance globale du collecteur de métriques."""
    collector = MetricsCollector.get_instance()
    
    # Configuration par défaut si première utilisation
    if not hasattr(collector, '_default_setup_done'):
        setup_default_alerts(collector)
        collector.add_alert_callback(log_alert_callback)
        collector._default_setup_done = True
    
    return collector

# ==================== EXPORTS PRINCIPAUX ====================

__all__ = [
    # Classes principales
    'SearchMetrics',
    'QueryMetrics', 
    'PerformanceMetrics',
    'MetricsCollector',
    'MetricsExporter',
    
    # Types et structures
    'MetricType',
    'AlertLevel',
    'TimeWindow',
    'MetricPoint',
    'MetricSummary',
    'Alert',
    'PerformanceProfile',
    
    # Décorateurs
    'measure_time',
    'count_calls',
    'track_errors',
    
    # Fonctions factory
    'create_metrics_collector',
    'create_search_metrics',
    'create_performance_metrics',
    'create_metrics_exporter',
    'get_metrics_collector',
    
    # Configuration
    'setup_default_alerts',
    'log_alert_callback',
    'email_alert_callback'
]