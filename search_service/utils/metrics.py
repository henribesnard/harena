"""
Métriques spécialisées pour le Search Service
===========================================

Module de collecte, agrégation et analyse des métriques de performance
pour tous les composants du Search Service :
- Métriques Elasticsearch (latence, throughput, erreurs)
- Métriques lexicales (pertinence, qualité, cache)
- Métriques d'exécution (requêtes, optimisations)
- Métriques de traitement (formatage, enrichissement)
- Métriques système (mémoire, CPU, I/O)

Architecture :
    Component → MetricsCollector → MetricsAggregator → MetricsReporter → Dashboard/Alerts
"""

import logging
import time
import threading
from typing import Dict, List, Optional, Any, Union, Callable, Set
from enum import Enum
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
import statistics
import json
import hashlib
from contextlib import contextmanager
import psutil

from config import settings


logger = logging.getLogger(__name__)


class MetricType(str, Enum):
    """Types de métriques collectées"""
    COUNTER = "counter"           # Compteur incrémental (requêtes totales)
    GAUGE = "gauge"              # Valeur instantanée (requêtes actives)
    HISTOGRAM = "histogram"      # Distribution de valeurs (temps de réponse)
    TIMER = "timer"              # Durée d'opérations
    RATE = "rate"                # Taux par seconde
    PERCENTAGE = "percentage"    # Pourcentage (0-100)
    BYTES = "bytes"              # Tailles en octets
    THROUGHPUT = "throughput"    # Débit (ops/sec)


class MetricCategory(str, Enum):
    """Catégories de métriques"""
    ELASTICSEARCH = "elasticsearch"     # Métriques ES
    LEXICAL_SEARCH = "lexical_search"  # Métriques recherche lexicale
    QUERY_EXECUTION = "query_execution" # Métriques exécution
    RESULT_PROCESSING = "result_processing" # Métriques traitement résultats
    CACHE = "cache"                    # Métriques cache
    PERFORMANCE = "performance"        # Métriques performance globale
    SYSTEM = "system"                  # Métriques système
    API = "api"                        # Métriques API REST
    BUSINESS = "business"              # Métriques métier


class AlertLevel(str, Enum):
    """Niveaux d'alerte"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class MetricDefinition:
    """Définition d'une métrique"""
    name: str
    type: MetricType
    category: MetricCategory
    description: str
    unit: str = ""
    tags: Set[str] = field(default_factory=set)
    warning_threshold: Optional[float] = None
    error_threshold: Optional[float] = None
    critical_threshold: Optional[float] = None
    retention_hours: int = 24
    aggregation_functions: List[str] = field(default_factory=lambda: ["avg", "count"])


@dataclass
class MetricValue:
    """Valeur d'une métrique avec métadonnées"""
    name: str
    value: Union[int, float]
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricSample:
    """Échantillon de métrique pour agrégation"""
    timestamp: datetime
    value: Union[int, float]
    tags: Dict[str, str] = field(default_factory=dict)


class MetricAlert:
    """Alerte basée sur une métrique"""
    
    def __init__(self, metric_name: str, level: AlertLevel, 
                 threshold: float, current_value: float, message: str):
        self.metric_name = metric_name
        self.level = level
        self.threshold = threshold
        self.current_value = current_value
        self.message = message
        self.timestamp = datetime.now()
        self.alert_id = self._generate_alert_id()
    
    def _generate_alert_id(self) -> str:
        """Génère un ID unique pour l'alerte"""
        content = f"{self.metric_name}_{self.level}_{self.timestamp.isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:8]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit l'alerte en dictionnaire"""
        return {
            "alert_id": self.alert_id,
            "metric_name": self.metric_name,
            "level": self.level.value,
            "threshold": self.threshold,
            "current_value": self.current_value,
            "message": self.message,
            "timestamp": self.timestamp.isoformat()
        }


class MetricsCollector:
    """Collecteur de métriques thread-safe"""
    
    def __init__(self, max_samples_per_metric: int = 10000):
        self._metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_samples_per_metric))
        self._definitions: Dict[str, MetricDefinition] = {}
        self._lock = threading.RLock()
        self._start_time = datetime.now()
        
        # Métriques système automatiques
        self._system_metrics_enabled = True
        self._system_metrics_interval = 30  # secondes
        
        self._register_default_metrics()
        self._start_system_metrics_collection()
    
    def register_metric(self, definition: MetricDefinition):
        """Enregistre une nouvelle métrique"""
        with self._lock:
            self._definitions[definition.name] = definition
            logger.debug(f"Métrique enregistrée: {definition.name}")
    
    def record(self, name: str, value: Union[int, float], 
               tags: Optional[Dict[str, str]] = None):
        """Enregistre une valeur de métrique"""
        
        if name not in self._definitions:
            logger.warning(f"Métrique non définie: {name}")
            return
        
        sample = MetricSample(
            timestamp=datetime.now(),
            value=value,
            tags=tags or {}
        )
        
        with self._lock:
            self._metrics[name].append(sample)
    
    def increment(self, name: str, amount: Union[int, float] = 1,
                  tags: Optional[Dict[str, str]] = None):
        """Incrémente un compteur"""
        
        # Récupérer la valeur actuelle et l'incrémenter
        current = self.get_current_value(name, default=0)
        self.record(name, current + amount, tags)
    
    def set_gauge(self, name: str, value: Union[int, float],
                  tags: Optional[Dict[str, str]] = None):
        """Définit la valeur d'une gauge"""
        self.record(name, value, tags)
    
    @contextmanager
    def timer(self, name: str, tags: Optional[Dict[str, str]] = None):
        """Context manager pour mesurer la durée"""
        start_time = time.time()
        try:
            yield
        finally:
            duration_ms = (time.time() - start_time) * 1000
            self.record(name, duration_ms, tags)
    
    def get_current_value(self, name: str, default: Any = None) -> Any:
        """Récupère la valeur actuelle d'une métrique"""
        with self._lock:
            samples = self._metrics.get(name)
            if samples:
                return samples[-1].value
            return default
    
    def get_samples(self, name: str, since: Optional[datetime] = None,
                   limit: Optional[int] = None) -> List[MetricSample]:
        """Récupère les échantillons d'une métrique"""
        
        with self._lock:
            samples = list(self._metrics.get(name, []))
        
        # Filtrer par date si spécifiée
        if since:
            samples = [s for s in samples if s.timestamp >= since]
        
        # Limiter le nombre de résultats
        if limit:
            samples = samples[-limit:]
        
        return samples
    
    def get_metric_stats(self, name: str, since: Optional[datetime] = None) -> Dict[str, Any]:
        """Calcule les statistiques d'une métrique"""
        
        samples = self.get_samples(name, since)
        if not samples:
            return {"count": 0}
        
        values = [s.value for s in samples]
        
        stats = {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "avg": statistics.mean(values),
            "sum": sum(values),
            "first": values[0],
            "last": values[-1]
        }
        
        if len(values) > 1:
            stats["median"] = statistics.median(values)
            stats["std_dev"] = statistics.stdev(values)
            
            # Percentiles
            if len(values) >= 10:
                sorted_values = sorted(values)
                stats["p50"] = statistics.median(sorted_values)
                stats["p90"] = sorted_values[int(0.9 * len(sorted_values))]
                stats["p95"] = sorted_values[int(0.95 * len(sorted_values))]
                stats["p99"] = sorted_values[int(0.99 * len(sorted_values))]
        
        return stats
    
    def _register_default_metrics(self):
        """Enregistre les métriques par défaut"""
        
        default_metrics = [
            # Métriques Elasticsearch
            MetricDefinition(
                "elasticsearch_search_duration_ms",
                MetricType.HISTOGRAM,
                MetricCategory.ELASTICSEARCH,
                "Durée des requêtes Elasticsearch en millisecondes",
                "ms",
                warning_threshold=100,
                error_threshold=500,
                critical_threshold=2000
            ),
            MetricDefinition(
                "elasticsearch_search_count",
                MetricType.COUNTER,
                MetricCategory.ELASTICSEARCH,
                "Nombre total de requêtes Elasticsearch"
            ),
            MetricDefinition(
                "elasticsearch_error_count",
                MetricType.COUNTER,
                MetricCategory.ELASTICSEARCH,
                "Nombre d'erreurs Elasticsearch"
            ),
            MetricDefinition(
                "elasticsearch_cache_hit_rate",
                MetricType.PERCENTAGE,
                MetricCategory.ELASTICSEARCH,
                "Taux de cache hit Elasticsearch",
                "%"
            ),
            
            # Métriques recherche lexicale
            MetricDefinition(
                "lexical_search_duration_ms",
                MetricType.HISTOGRAM,
                MetricCategory.LEXICAL_SEARCH,
                "Durée totale recherche lexicale",
                "ms",
                warning_threshold=50,
                error_threshold=200,
                critical_threshold=1000
            ),
            MetricDefinition(
                "lexical_search_quality_score",
                MetricType.HISTOGRAM,
                MetricCategory.LEXICAL_SEARCH,
                "Score de qualité des résultats de recherche"
            ),
            MetricDefinition(
                "lexical_cache_hit_rate",
                MetricType.PERCENTAGE,
                MetricCategory.CACHE,
                "Taux de cache hit du moteur lexical",
                "%"
            ),
            
            # Métriques exécution de requêtes
            MetricDefinition(
                "query_execution_duration_ms",
                MetricType.HISTOGRAM,
                MetricCategory.QUERY_EXECUTION,
                "Durée d'exécution des requêtes",
                "ms"
            ),
            MetricDefinition(
                "query_optimization_count",
                MetricType.COUNTER,
                MetricCategory.QUERY_EXECUTION,
                "Nombre d'optimisations appliquées"
            ),
            
            # Métriques traitement résultats
            MetricDefinition(
                "result_processing_duration_ms",
                MetricType.HISTOGRAM,
                MetricCategory.RESULT_PROCESSING,
                "Durée de traitement des résultats",
                "ms"
            ),
            MetricDefinition(
                "results_processed_count",
                MetricType.COUNTER,
                MetricCategory.RESULT_PROCESSING,
                "Nombre de résultats traités"
            ),
            
            # Métriques API
            MetricDefinition(
                "api_request_duration_ms",
                MetricType.HISTOGRAM,
                MetricCategory.API,
                "Durée des requêtes API",
                "ms"
            ),
            MetricDefinition(
                "api_request_count",
                MetricType.COUNTER,
                MetricCategory.API,
                "Nombre de requêtes API"
            ),
            MetricDefinition(
                "api_error_count",
                MetricType.COUNTER,
                MetricCategory.API,
                "Nombre d'erreurs API"
            ),
            
            # Métriques système
            MetricDefinition(
                "system_memory_usage_bytes",
                MetricType.GAUGE,
                MetricCategory.SYSTEM,
                "Usage mémoire du processus",
                "bytes",
                warning_threshold=512*1024*1024,  # 512MB
                error_threshold=1024*1024*1024,   # 1GB
                critical_threshold=2048*1024*1024  # 2GB
            ),
            MetricDefinition(
                "system_cpu_usage_percent",
                MetricType.GAUGE,
                MetricCategory.SYSTEM,
                "Usage CPU du processus",
                "%",
                warning_threshold=70,
                error_threshold=85,
                critical_threshold=95
            ),
            
            # Métriques métier
            MetricDefinition(
                "search_intent_success_rate",
                MetricType.PERCENTAGE,
                MetricCategory.BUSINESS,
                "Taux de succès de détection d'intention",
                "%"
            ),
            MetricDefinition(
                "user_satisfaction_score",
                MetricType.HISTOGRAM,
                MetricCategory.BUSINESS,
                "Score de satisfaction utilisateur (1-5)"
            ),
            MetricDefinition(
                "api_search_duration_ms",
                MetricType.HISTOGRAM,
                MetricCategory.API,
                "Durée des appels API de recherche",
                "ms",
                warning_threshold=200,
                error_threshold=1000,
                critical_threshold=5000
            ),
            MetricDefinition(
                "api_search_count",
                MetricType.COUNTER,
                MetricCategory.API,
                "Nombre d'appels API de recherche"
            ),
            MetricDefinition(
                "api_search_success_count",
                MetricType.COUNTER,
                MetricCategory.API,
                "Nombre d'appels API de recherche réussis"
            ),
            MetricDefinition(
                "api_search_error_count",
                MetricType.COUNTER,
                MetricCategory.API,
                "Nombre d'erreurs API de recherche"
            ),
            MetricDefinition(
                "api_search_results_count",
                MetricType.HISTOGRAM,
                MetricCategory.API,
                "Nombre de résultats retournés par recherche"
            ),
            MetricDefinition(
                "api_validation_duration_ms",
                MetricType.HISTOGRAM,
                MetricCategory.API,
                "Durée des appels API de validation",
                "ms"
            ),
            MetricDefinition(
                "api_validation_count",
                MetricType.COUNTER,
                MetricCategory.API,
                "Nombre d'appels API de validation"
            ),
            MetricDefinition(
                "api_validation_success_count",
                MetricType.COUNTER,
                MetricCategory.API,
                "Nombre de validations réussies"
            ),
            MetricDefinition(
                "api_validation_error_count",
                MetricType.COUNTER,
                MetricCategory.API,
                "Nombre d'erreurs de validation"
            ),
            MetricDefinition(
                "api_health_percentage",
                MetricType.GAUGE,
                MetricCategory.API,
                "Pourcentage de santé globale de l'API",
                "%",
                warning_threshold=90,
                error_threshold=75,
                critical_threshold=50
            ),
            MetricDefinition(
                "api_rate_limit_exceeded",
                MetricType.COUNTER,
                MetricCategory.API,
                "Nombre de dépassements de limite de taux"
            ),
            MetricDefinition(
                "api_auth_attempts",
                MetricType.COUNTER,
                MetricCategory.API,
                "Nombre de tentatives d'authentification"
            ),
            MetricDefinition(
                "api_auth_success_count",
                MetricType.COUNTER,
                MetricCategory.API,
                "Nombre d'authentifications réussies"
            ),
            MetricDefinition(
                "api_auth_failure_count",
                MetricType.COUNTER,
                MetricCategory.API,
                "Nombre d'échecs d'authentification"
            )
        ]
        
        for metric in default_metrics:
            self.register_metric(metric)
    
    def _start_system_metrics_collection(self):
        """Démarre la collecte automatique des métriques système"""
        
        if not self._system_metrics_enabled:
            return
        
        def collect_system_metrics():
            """Collecte les métriques système périodiquement"""
            
            while self._system_metrics_enabled:
                try:
                    # Métriques mémoire
                    process = psutil.Process()
                    memory_info = process.memory_info()
                    self.set_gauge("system_memory_usage_bytes", memory_info.rss)
                    
                    # Métriques CPU
                    cpu_percent = process.cpu_percent()
                    self.set_gauge("system_cpu_usage_percent", cpu_percent)
                    
                    # Métriques disque I/O
                    io_counters = process.io_counters()
                    self.set_gauge("system_disk_read_bytes", io_counters.read_bytes)
                    self.set_gauge("system_disk_write_bytes", io_counters.write_bytes)
                    
                    time.sleep(self._system_metrics_interval)
                    
                except Exception as e:
                    logger.error(f"Erreur lors de la collecte des métriques système: {e}")
                    time.sleep(self._system_metrics_interval)
        
        # Démarrer le thread de collecte
        thread = threading.Thread(target=collect_system_metrics, daemon=True)
        thread.start()
        logger.info("Collecte automatique des métriques système démarrée")
    
    def cleanup_old_samples(self, hours: int = 24):
        """Nettoie les échantillons anciens"""
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        cleaned_count = 0
        
        with self._lock:
            for metric_name, samples in self._metrics.items():
                original_count = len(samples)
                
                # Filtrer les échantillons récents
                recent_samples = deque([
                    sample for sample in samples 
                    if sample.timestamp > cutoff_time
                ], maxlen=samples.maxlen)
                
                self._metrics[metric_name] = recent_samples
                cleaned_count += original_count - len(recent_samples)
        
        if cleaned_count > 0:
            logger.info(f"Nettoyage des métriques: {cleaned_count} échantillons supprimés")
    
    def export_metrics(self, format: str = "json") -> str:
        """Exporte les métriques dans différents formats"""
        
        if format == "json":
            return self._export_json()
        elif format == "prometheus":
            return self._export_prometheus()
        else:
            raise ValueError(f"Format d'export non supporté: {format}")
    
    def _export_json(self) -> str:
        """Exporte les métriques au format JSON"""
        
        export_data = {
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": (datetime.now() - self._start_time).total_seconds(),
            "metrics": {}
        }
        
        for name, definition in self._definitions.items():
            stats = self.get_metric_stats(name)
            export_data["metrics"][name] = {
                "definition": {
                    "type": definition.type.value,
                    "category": definition.category.value,
                    "description": definition.description,
                    "unit": definition.unit
                },
                "stats": stats,
                "current_value": self.get_current_value(name)
            }
        
        return json.dumps(export_data, indent=2, default=str)
    
    def _export_prometheus(self) -> str:
        """Exporte les métriques au format Prometheus"""
        
        lines = []
        
        for name, definition in self._definitions.items():
            # Nom de métrique compatible Prometheus
            prom_name = f"{settings.metrics_prefix}_{name}".replace("-", "_")
            
            # Commentaire de description
            lines.append(f"# HELP {prom_name} {definition.description}")
            lines.append(f"# TYPE {prom_name} {self._prometheus_type(definition.type)}")
            
            # Valeur actuelle
            current_value = self.get_current_value(name, 0)
            lines.append(f"{prom_name} {current_value}")
        
        return "\n".join(lines)
    
    def _prometheus_type(self, metric_type: MetricType) -> str:
        """Convertit le type de métrique vers Prometheus"""
        
        mapping = {
            MetricType.COUNTER: "counter",
            MetricType.GAUGE: "gauge",
            MetricType.HISTOGRAM: "histogram",
            MetricType.TIMER: "histogram"
        }
        
        return mapping.get(metric_type, "gauge")


class AlertManager:
    """Gestionnaire d'alertes basées sur les métriques"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.active_alerts: Dict[str, MetricAlert] = {}
        self.alert_history: deque = deque(maxlen=1000)
        self.alert_callbacks: Dict[AlertLevel, List[Callable]] = defaultdict(list)
        self.check_interval = 60  # secondes
        
        self._start_alert_monitoring()
    
    def add_alert_callback(self, level: AlertLevel, callback: Callable[[MetricAlert], None]):
        """Ajoute un callback pour un niveau d'alerte"""
        self.alert_callbacks[level].append(callback)
    
    def check_alerts(self):
        """Vérifie toutes les métriques pour les seuils d'alerte"""
        
        for name, definition in self.metrics_collector._definitions.items():
            current_value = self.metrics_collector.get_current_value(name)
            
            if current_value is None:
                continue
            
            # Vérifier les seuils dans l'ordre de gravité
            alert_level = None
            threshold = None
            
            if (definition.critical_threshold is not None and 
                current_value >= definition.critical_threshold):
                alert_level = AlertLevel.CRITICAL
                threshold = definition.critical_threshold
                
            elif (definition.error_threshold is not None and 
                  current_value >= definition.error_threshold):
                alert_level = AlertLevel.ERROR
                threshold = definition.error_threshold
                
            elif (definition.warning_threshold is not None and 
                  current_value >= definition.warning_threshold):
                alert_level = AlertLevel.WARNING
                threshold = definition.warning_threshold
            
            # Créer ou mettre à jour l'alerte
            if alert_level:
                self._trigger_alert(name, alert_level, threshold, current_value)
            else:
                # Résoudre l'alerte si elle existait
                self._resolve_alert(name)
    
    def _trigger_alert(self, metric_name: str, level: AlertLevel, 
                      threshold: float, current_value: float):
        """Déclenche une alerte"""
        
        alert_key = f"{metric_name}_{level.value}"
        
        # Éviter les alertes en double
        if alert_key in self.active_alerts:
            return
        
        message = (f"Métrique '{metric_name}' au niveau {level.value}: "
                  f"{current_value} >= {threshold}")
        
        alert = MetricAlert(metric_name, level, threshold, current_value, message)
        
        self.active_alerts[alert_key] = alert
        self.alert_history.append(alert)
        
        # Exécuter les callbacks
        for callback in self.alert_callbacks[level]:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Erreur dans callback d'alerte: {e}")
        
        logger.warning(f"🚨 ALERTE {level.value.upper()}: {message}")
    
    def _resolve_alert(self, metric_name: str):
        """Résout une alerte"""
        
        resolved_keys = []
        for alert_key in self.active_alerts:
            if alert_key.startswith(metric_name):
                resolved_keys.append(alert_key)
        
        for key in resolved_keys:
            alert = self.active_alerts.pop(key)
            logger.info(f"✅ Alerte résolue: {alert.metric_name}")
    
    def _start_alert_monitoring(self):
        """Démarre la surveillance automatique des alertes"""
        
        def monitor_alerts():
            while True:
                try:
                    self.check_alerts()
                    time.sleep(self.check_interval)
                except Exception as e:
                    logger.error(f"Erreur lors de la vérification des alertes: {e}")
                    time.sleep(self.check_interval)
        
        thread = threading.Thread(target=monitor_alerts, daemon=True)
        thread.start()
        logger.info("Surveillance automatique des alertes démarrée")
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Retourne les alertes actives"""
        return [alert.to_dict() for alert in self.active_alerts.values()]
    
    def get_alert_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Retourne l'historique des alertes"""
        recent_alerts = list(self.alert_history)[-limit:]
        return [alert.to_dict() for alert in recent_alerts]


class SearchMetrics:
    """Métriques spécialisées pour la recherche"""
    
    def __init__(self, collector: MetricsCollector):
        self.collector = collector
    
    def record_search_request(self, user_id: int, intent_type: str, 
                            query_complexity: str):
        """Enregistre une demande de recherche"""
        
        tags = {
            "user_id": str(user_id),
            "intent_type": intent_type,
            "complexity": query_complexity
        }
        
        self.collector.increment("api_request_count", tags=tags)
        self.collector.increment("lexical_search_count", tags=tags)
    
    def record_search_execution(self, duration_ms: float, cache_hit: bool,
                              results_count: int, quality_score: float):
        """Enregistre l'exécution d'une recherche"""
        
        tags = {
            "cache_hit": str(cache_hit),
            "has_results": str(results_count > 0)
        }
        
        self.collector.record("lexical_search_duration_ms", duration_ms, tags)
        self.collector.record("lexical_search_quality_score", quality_score, tags)
        self.collector.record("results_processed_count", results_count, tags)
        
        # Mettre à jour le taux de cache hit
        if cache_hit:
            self.collector.increment("lexical_cache_hits")
        self.collector.increment("lexical_cache_requests")
        
        # Calculer le taux
        hits = self.collector.get_current_value("lexical_cache_hits", 0)
        requests = self.collector.get_current_value("lexical_cache_requests", 1)
        hit_rate = (hits / requests) * 100
        self.collector.set_gauge("lexical_cache_hit_rate", hit_rate)
    
    def record_elasticsearch_execution(self, duration_ms: float, 
                                     success: bool, error_type: str = None):
        """Enregistre l'exécution Elasticsearch"""
        
        tags = {
            "success": str(success),
            "error_type": error_type or "none"
        }
        
        self.collector.record("elasticsearch_search_duration_ms", duration_ms, tags)
        
        if success:
            self.collector.increment("elasticsearch_search_count", tags=tags)
        else:
            self.collector.increment("elasticsearch_error_count", tags=tags)
    
    def record_optimization_applied(self, optimization_type: str, 
                                  improvement_percent: float):
        """Enregistre une optimisation appliquée"""
        
        tags = {"optimization_type": optimization_type}
        
        self.collector.increment("query_optimization_count", tags=tags)
        self.collector.record("optimization_improvement_percent", 
                            improvement_percent, tags)


class PerformanceProfiler:
    """Profileur de performance pour opérations critiques"""
    
    def __init__(self, collector: MetricsCollector):
        self.collector = collector
        self.active_profiles: Dict[str, datetime] = {}
    
    @contextmanager
    def profile_operation(self, operation_name: str, 
                         tags: Optional[Dict[str, str]] = None):
        """Profile une opération avec métriques détaillées"""
        
        profile_id = f"{operation_name}_{time.time()}"
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss
        
        self.active_profiles[profile_id] = datetime.now()
        
        try:
            yield profile_id
            
        finally:
            # Calculer les métriques
            duration_ms = (time.time() - start_time) * 1000
            end_memory = psutil.Process().memory_info().rss
            memory_delta = end_memory - start_memory
            
            # Enregistrer les métriques
            metric_tags = tags or {}
            metric_tags["operation"] = operation_name
            
            self.collector.record(f"{operation_name}_duration_ms", 
                                duration_ms, metric_tags)
            self.collector.record(f"{operation_name}_memory_delta_bytes", 
                                memory_delta, metric_tags)
            
            # Nettoyer
            self.active_profiles.pop(profile_id, None)


# === INSTANCES GLOBALES ===

# Collecteur principal de métriques
metrics_collector = MetricsCollector()

# Gestionnaire d'alertes
alert_manager = AlertManager(metrics_collector)

# Métriques spécialisées pour la recherche
search_metrics = SearchMetrics(metrics_collector)

# Profileur de performance
performance_profiler = PerformanceProfiler(metrics_collector)


# === FONCTIONS UTILITAIRES ===

def get_system_metrics() -> Dict[str, Any]:
    """Retourne un résumé des métriques système"""
    
    return {
        "uptime_seconds": (datetime.now() - metrics_collector._start_time).total_seconds(),
        "total_metrics": len(metrics_collector._definitions),
        "active_alerts": len(alert_manager.active_alerts),
        "memory_usage_mb": psutil.Process().memory_info().rss / (1024 * 1024),
        "cpu_usage_percent": psutil.Process().cpu_percent()
    }


def get_performance_summary(hours: int = 1) -> Dict[str, Any]:
    """Génère un résumé de performance sur une période"""
    
    since = datetime.now() - timedelta(hours=hours)
    
    # Métriques clés de performance
    key_metrics = [
        "lexical_search_duration_ms",
        "elasticsearch_search_duration_ms", 
        "api_request_duration_ms",
        "lexical_cache_hit_rate",
        "lexical_search_quality_score"
    ]
    
    summary = {
        "period_hours": hours,
        "generated_at": datetime.now().isoformat(),
        "metrics": {}
    }
    
    for metric_name in key_metrics:
        stats = metrics_collector.get_metric_stats(metric_name, since)
        summary["metrics"][metric_name] = stats
    
    return summary


def export_metrics_to_file(filepath: str, format: str = "json"):
    """Exporte les métriques vers un fichier"""
    
    try:
        data = metrics_collector.export_metrics(format)
        
        with open(filepath, 'w') as f:
            f.write(data)
        
        logger.info(f"Métriques exportées vers {filepath}")
        
    except Exception as e:
        logger.error(f"Erreur lors de l'export des métriques: {e}")
        raise


def cleanup_old_metrics(hours: int = 24):
    """Nettoie les anciennes métriques"""
    metrics_collector.cleanup_old_samples(hours)


def reset_all_counters():
    """Remet à zéro tous les compteurs"""
    
    counter_metrics = [
        name for name, definition in metrics_collector._definitions.items()
        if definition.type == MetricType.COUNTER
    ]
    
    for metric_name in counter_metrics:
        metrics_collector.record(metric_name, 0)
    
    logger.info(f"Remise à zéro de {len(counter_metrics)} compteurs")


def get_top_slow_operations(limit: int = 10, hours: int = 1) -> List[Dict[str, Any]]:
    """Retourne les opérations les plus lentes"""
    
    since = datetime.now() - timedelta(hours=hours)
    
    duration_metrics = [
        name for name, definition in metrics_collector._definitions.items()
        if "duration_ms" in name
    ]
    
    slow_operations = []
    
    for metric_name in duration_metrics:
        samples = metrics_collector.get_samples(metric_name, since)
        if not samples:
            continue
        
        # Trouver les échantillons les plus lents
        sorted_samples = sorted(samples, key=lambda x: x.value, reverse=True)
        
        for sample in sorted_samples[:limit]:
            slow_operations.append({
                "operation": metric_name.replace("_duration_ms", ""),
                "duration_ms": sample.value,
                "timestamp": sample.timestamp.isoformat(),
                "tags": sample.tags
            })
    
    # Trier par durée globale
    slow_operations.sort(key=lambda x: x["duration_ms"], reverse=True)
    
    return slow_operations[:limit]


def get_error_metrics_summary(hours: int = 24) -> Dict[str, Any]:
    """Résumé des métriques d'erreurs"""
    
    since = datetime.now() - timedelta(hours=hours)
    
    error_metrics = [
        name for name, definition in metrics_collector._definitions.items()
        if "error" in name.lower()
    ]
    
    summary = {
        "period_hours": hours,
        "total_errors": 0,
        "error_breakdown": {}
    }
    
    for metric_name in error_metrics:
        stats = metrics_collector.get_metric_stats(metric_name, since)
        error_count = stats.get("sum", 0)
        
        summary["total_errors"] += error_count
        summary["error_breakdown"][metric_name] = {
            "count": error_count,
            "rate_per_hour": error_count / hours if hours > 0 else 0
        }
    
    return summary


# === MÉTRIQUES SPÉCIALISÉES SEARCH SERVICE ===

class QueryMetrics:
    """Métriques pour le query_executor - Alias pour compatibilité"""
    
    def __init__(self, collector: MetricsCollector):
        self.collector = collector
        self.elasticsearch_metrics = ElasticsearchMetrics(collector)
    
    def record_execution_result(self, query_type: str, execution_time_ms: float,
                              success: bool, cache_hit: bool = False):
        """Enregistre le résultat d'une exécution de requête"""
        
        tags = {
            "query_type": query_type,
            "success": str(success),
            "cache_hit": str(cache_hit)
        }
        
        self.collector.record("query_execution_duration_ms", execution_time_ms, tags)
        
        if success:
            self.collector.increment("query_execution_success_count", tags=tags)
        else:
            self.collector.increment("query_execution_error_count", tags=tags)
    
    def get_summary(self) -> Dict[str, Any]:
        """Retourne un résumé des métriques d'exécution"""
        
        success_count = self.collector.get_current_value("query_execution_success_count", 0)
        error_count = self.collector.get_current_value("query_execution_error_count", 0)
        total_count = success_count + error_count
        
        success_rate = (success_count / total_count * 100) if total_count > 0 else 0
        
        return {
            "total_executions": total_count,
            "success_count": success_count,
            "error_count": error_count,
            "success_rate_percent": success_rate,
            "avg_duration_ms": self.collector.get_metric_stats("query_execution_duration_ms").get("avg", 0)
        }


class ResultMetrics:
    """Métriques pour le result_processor"""
    
    def __init__(self, collector: MetricsCollector):
        self.collector = collector
    
    def record_processing_result(self, processing_type: str, duration_ms: float,
                               input_count: int, output_count: int, success: bool):
        """Enregistre le résultat d'un traitement de résultats"""
        
        tags = {
            "processing_type": processing_type,
            "success": str(success)
        }
        
        self.collector.record("result_processing_duration_ms", duration_ms, tags)
        self.collector.record("result_input_count", input_count, tags)
        self.collector.record("result_output_count", output_count, tags)
        
        if success:
            self.collector.increment("result_processing_success_count", tags=tags)
        else:
            self.collector.increment("result_processing_error_count", tags=tags)
        
        # Calculer l'efficacité
        efficiency = (output_count / input_count * 100) if input_count > 0 else 0
        self.collector.record("result_processing_efficiency_percent", efficiency, tags)
    
    def record_enrichment_applied(self, enrichment_type: str, duration_ms: float):
        """Enregistre l'application d'un enrichissement"""
        
        tags = {"enrichment_type": enrichment_type}
        
        self.collector.record("result_enrichment_duration_ms", duration_ms, tags)
        self.collector.increment("result_enrichment_count", tags=tags)
    
    def get_summary(self) -> Dict[str, Any]:
        """Retourne un résumé des métriques de traitement"""
        
        success_count = self.collector.get_current_value("result_processing_success_count", 0)
        error_count = self.collector.get_current_value("result_processing_error_count", 0)
        total_count = success_count + error_count
        
        success_rate = (success_count / total_count * 100) if total_count > 0 else 0
        
        return {
            "total_processing": total_count,
            "success_count": success_count,
            "error_count": error_count,
            "success_rate_percent": success_rate,
            "avg_duration_ms": self.collector.get_metric_stats("result_processing_duration_ms").get("avg", 0),
            "avg_efficiency_percent": self.collector.get_metric_stats("result_processing_efficiency_percent").get("avg", 0)
        }


class SearchMetrics:
    """Métriques spécialisées pour la recherche"""
    
    def __init__(self, collector: MetricsCollector):
        self.collector = collector
    
    def record_search_request(self, user_id: int, intent_type: str, 
                            query_complexity: str):
        """Enregistre une demande de recherche"""
        
        tags = {
            "user_id": str(user_id),
            "intent_type": intent_type,
            "complexity": query_complexity
        }
        
        self.collector.increment("api_request_count", tags=tags)
        self.collector.increment("lexical_search_count", tags=tags)
    
    def record_search_execution(self, duration_ms: float, cache_hit: bool,
                              results_count: int, quality_score: float):
        """Enregistre l'exécution d'une recherche"""
        
        tags = {
            "cache_hit": str(cache_hit),
            "has_results": str(results_count > 0)
        }
        
        self.collector.record("lexical_search_duration_ms", duration_ms, tags)
        self.collector.record("lexical_search_quality_score", quality_score, tags)
        self.collector.record("results_processed_count", results_count, tags)
        
        # Mettre à jour le taux de cache hit
        if cache_hit:
            self.collector.increment("lexical_cache_hits")
        self.collector.increment("lexical_cache_requests")
        
        # Calculer le taux
        hits = self.collector.get_current_value("lexical_cache_hits", 0)
        requests = self.collector.get_current_value("lexical_cache_requests", 1)
        hit_rate = (hits / requests) * 100
        self.collector.set_gauge("lexical_cache_hit_rate", hit_rate)
    
    def record_elasticsearch_execution(self, duration_ms: float, 
                                     success: bool, error_type: str = None):
        """Enregistre l'exécution Elasticsearch"""
        
        tags = {
            "success": str(success),
            "error_type": error_type or "none"
        }
        
        self.collector.record("elasticsearch_search_duration_ms", duration_ms, tags)
        
        if success:
            self.collector.increment("elasticsearch_search_count", tags=tags)
        else:
            self.collector.increment("elasticsearch_error_count", tags=tags)
    
    def record_optimization_applied(self, optimization_type: str, 
                                  improvement_percent: float):
        """Enregistre une optimisation appliquée"""
        
        tags = {"optimization_type": optimization_type}
        
        self.collector.increment("query_optimization_count", tags=tags)
        self.collector.record("optimization_improvement_percent", 
                            improvement_percent, tags)


class ElasticsearchMetrics:
    """Métriques spécialisées Elasticsearch"""
    
    def __init__(self, collector: MetricsCollector):
        self.collector = collector
    
    def record_query_execution(self, query_type: str, duration_ms: float,
                              success: bool, cache_hit: bool = False,
                              shard_count: int = 1, result_count: int = 0):
        """Enregistre l'exécution d'une requête Elasticsearch"""
        
        tags = {
            "query_type": query_type,
            "success": str(success),
            "cache_hit": str(cache_hit),
            "shard_count": str(shard_count)
        }
        
        self.collector.record("elasticsearch_search_duration_ms", duration_ms, tags)
        
        if success:
            self.collector.increment("elasticsearch_search_count", tags=tags)
            self.collector.record("elasticsearch_result_count", result_count, tags)
        else:
            self.collector.increment("elasticsearch_error_count", tags=tags)
        
        # Métriques de cache
        if cache_hit:
            self.collector.increment("elasticsearch_cache_hits")
        self.collector.increment("elasticsearch_cache_requests")
        
        # Calculer le taux de cache hit
        hits = self.collector.get_current_value("elasticsearch_cache_hits", 0)
        requests = self.collector.get_current_value("elasticsearch_cache_requests", 1)
        hit_rate = (hits / requests) * 100
        self.collector.set_gauge("elasticsearch_cache_hit_rate", hit_rate)
    
    def record_aggregation_execution(self, agg_type: str, duration_ms: float,
                                   bucket_count: int):
        """Enregistre l'exécution d'une agrégation"""
        
        tags = {"aggregation_type": agg_type}
        
        self.collector.record("elasticsearch_aggregation_duration_ms", 
                            duration_ms, tags)
        self.collector.record("elasticsearch_aggregation_buckets", 
                            bucket_count, tags)
        self.collector.increment("elasticsearch_aggregation_count", tags=tags)
    
    def record_connection_event(self, event_type: str, node_count: int = 1):
        """Enregistre un événement de connexion"""
        
        tags = {"event_type": event_type}
        
        self.collector.increment("elasticsearch_connection_events", tags=tags)
        self.collector.set_gauge("elasticsearch_active_nodes", node_count)


class LexicalSearchMetrics:
    """Métriques spécialisées pour la recherche lexicale"""
    
    def __init__(self, collector: MetricsCollector):
        self.collector = collector
    
    def record_search_analysis(self, query_text: str, complexity: str,
                             field_count: int, filter_count: int):
        """Enregistre l'analyse d'une requête de recherche"""
        
        tags = {
            "complexity": complexity,
            "has_text": str(bool(query_text.strip())),
            "field_count": str(field_count)
        }
        
        self.collector.record("lexical_query_complexity_score", 
                            self._calculate_complexity_score(complexity), tags)
        self.collector.record("lexical_filter_count", filter_count, tags)
        self.collector.record("lexical_search_field_count", field_count, tags)
        self.collector.increment("lexical_query_analyzed", tags=tags)
    
    def record_result_processing(self, processing_duration_ms: float,
                               input_count: int, output_count: int,
                               enrichment_applied: List[str]):
        """Enregistre le traitement des résultats"""
        
        tags = {
            "enrichment_count": str(len(enrichment_applied)),
            "has_highlights": str("highlights" in enrichment_applied)
        }
        
        self.collector.record("result_processing_duration_ms", 
                            processing_duration_ms, tags)
        self.collector.record("result_input_count", input_count, tags)
        self.collector.record("result_output_count", output_count, tags)
        
        # Efficacité du traitement
        if input_count > 0:
            efficiency = output_count / input_count
            self.collector.record("result_processing_efficiency", efficiency, tags)
    
    def record_optimization_impact(self, optimization_type: str,
                                 before_duration_ms: float,
                                 after_duration_ms: float):
        """Enregistre l'impact d'une optimisation"""
        
        improvement_percent = 0
        if before_duration_ms > 0:
            improvement_percent = ((before_duration_ms - after_duration_ms) / 
                                 before_duration_ms) * 100
        
        tags = {"optimization_type": optimization_type}
        
        self.collector.record("optimization_improvement_percent", 
                            improvement_percent, tags)
        self.collector.record("optimization_before_duration_ms", 
                            before_duration_ms, tags)
        self.collector.record("optimization_after_duration_ms", 
                            after_duration_ms, tags)
        self.collector.increment("optimization_applied_count", tags=tags)
    
    def _calculate_complexity_score(self, complexity: str) -> int:
        """Convertit la complexité en score numérique"""
        mapping = {
            "simple": 1,
            "moderate": 2,
            "complex": 3,
            "very_complex": 4
        }
        return mapping.get(complexity, 2)


class BusinessMetrics:
    """Métriques métier pour le Search Service"""
    
    def __init__(self, collector: MetricsCollector):
        self.collector = collector
    
    def record_user_interaction(self, user_id: int, intent_type: str,
                              satisfaction_score: Optional[float] = None,
                              session_duration_ms: Optional[float] = None):
        """Enregistre une interaction utilisateur"""
        
        tags = {
            "user_id": str(user_id),
            "intent_type": intent_type
        }
        
        self.collector.increment("user_interaction_count", tags=tags)
        
        if satisfaction_score is not None:
            self.collector.record("user_satisfaction_score", 
                                satisfaction_score, tags)
        
        if session_duration_ms is not None:
            self.collector.record("user_session_duration_ms", 
                                session_duration_ms, tags)
    
    def record_search_success(self, intent_type: str, successful: bool,
                            results_found: int, user_clicked: bool = False):
        """Enregistre le succès d'une recherche"""
        
        tags = {
            "intent_type": intent_type,
            "successful": str(successful),
            "has_results": str(results_found > 0),
            "user_clicked": str(user_clicked)
        }
        
        self.collector.increment("search_attempt_count", tags=tags)
        
        if successful:
            self.collector.increment("search_success_count", tags=tags)
            self.collector.record("search_results_count", results_found, tags)
        
        # Calculer le taux de succès
        success_count = self.collector.get_current_value("search_success_count", 0)
        attempt_count = self.collector.get_current_value("search_attempt_count", 1)
        success_rate = (success_count / attempt_count) * 100
        self.collector.set_gauge("search_intent_success_rate", success_rate)
    
    def record_financial_query_patterns(self, category: str, amount_range: str,
                                      time_period: str, complexity: str):
        """Enregistre les patterns de requêtes financières"""
        
        tags = {
            "category": category,
            "amount_range": amount_range,
            "time_period": time_period,
            "complexity": complexity
        }
        
        self.collector.increment("financial_query_pattern_count", tags=tags)
        
        # Métriques spécifiques par catégorie
        category_tags = {"category": category}
        self.collector.increment(f"financial_category_{category.lower()}_count", 
                               tags=category_tags)

class ApiMetrics:
    """Métriques spécialisées pour l'API REST"""
    
    def __init__(self, collector: MetricsCollector):
        self.collector = collector
    
    def record_search_api_call(self, endpoint: str, duration_ms: float,
                              success: bool, user_id: Optional[int] = None,
                              results_count: int = 0, cache_hit: bool = False):
        """Enregistre un appel API de recherche"""
        
        tags = {
            "endpoint": endpoint,
            "success": str(success),
            "cache_hit": str(cache_hit),
            "has_results": str(results_count > 0)
        }
        
        if user_id:
            tags["user_id"] = str(user_id)
        
        self.collector.record("api_search_duration_ms", duration_ms, tags)
        self.collector.record("api_search_results_count", results_count, tags)
        self.collector.increment("api_search_count", tags=tags)
        
        if success:
            self.collector.increment("api_search_success_count", tags=tags)
        else:
            self.collector.increment("api_search_error_count", tags=tags)
    
    def record_validation_api_call(self, duration_ms: float, success: bool,
                                  user_id: Optional[int] = None,
                                  validation_errors: int = 0):
        """Enregistre un appel API de validation"""
        
        tags = {
            "success": str(success),
            "has_errors": str(validation_errors > 0)
        }
        
        if user_id:
            tags["user_id"] = str(user_id)
        
        self.collector.record("api_validation_duration_ms", duration_ms, tags)
        self.collector.record("api_validation_errors_count", validation_errors, tags)
        self.collector.increment("api_validation_count", tags=tags)
        
        if success:
            self.collector.increment("api_validation_success_count", tags=tags)
        else:
            self.collector.increment("api_validation_error_count", tags=tags)
    
    def record_health_check(self, duration_ms: float, overall_status: str,
                           components_healthy: int, components_total: int):
        """Enregistre un check de santé"""
        
        tags = {
            "status": overall_status,
            "all_healthy": str(components_healthy == components_total)
        }
        
        self.collector.record("api_health_check_duration_ms", duration_ms, tags)
        self.collector.record("api_health_components_healthy", components_healthy, tags)
        self.collector.record("api_health_components_total", components_total, tags)
        self.collector.increment("api_health_check_count", tags=tags)
        
        # Calculer le pourcentage de santé
        health_percentage = (components_healthy / components_total * 100) if components_total > 0 else 0
        self.collector.set_gauge("api_health_percentage", health_percentage)
    
    def record_admin_operation(self, operation: str, duration_ms: float,
                              success: bool, user_id: Optional[int] = None):
        """Enregistre une opération d'administration"""
        
        tags = {
            "operation": operation,
            "success": str(success)
        }
        
        if user_id:
            tags["user_id"] = str(user_id)
        
        self.collector.record("api_admin_duration_ms", duration_ms, tags)
        self.collector.increment("api_admin_count", tags=tags)
        
        if success:
            self.collector.increment("api_admin_success_count", tags=tags)
        else:
            self.collector.increment("api_admin_error_count", tags=tags)
    
    def record_rate_limit_event(self, endpoint: str, user_id: Optional[int],
                               limit_exceeded: bool, current_rate: float):
        """Enregistre un événement de limitation de taux"""
        
        tags = {
            "endpoint": endpoint,
            "limit_exceeded": str(limit_exceeded)
        }
        
        if user_id:
            tags["user_id"] = str(user_id)
        
        self.collector.record("api_rate_limit_current", current_rate, tags)
        self.collector.increment("api_rate_limit_checks", tags=tags)
        
        if limit_exceeded:
            self.collector.increment("api_rate_limit_exceeded", tags=tags)
    
    def record_authentication_event(self, auth_method: str, success: bool,
                                   duration_ms: float, user_id: Optional[int] = None):
        """Enregistre un événement d'authentification"""
        
        tags = {
            "auth_method": auth_method,
            "success": str(success)
        }
        
        if user_id:
            tags["user_id"] = str(user_id)
        
        self.collector.record("api_auth_duration_ms", duration_ms, tags)
        self.collector.increment("api_auth_attempts", tags=tags)
        
        if success:
            self.collector.increment("api_auth_success_count", tags=tags)
        else:
            self.collector.increment("api_auth_failure_count", tags=tags)
    
    def record_error_event(self, error_type: str, endpoint: str, 
                          status_code: int, user_id: Optional[int] = None):
        """Enregistre un événement d'erreur"""
        
        tags = {
            "error_type": error_type,
            "endpoint": endpoint,
            "status_code": str(status_code)
        }
        
        if user_id:
            tags["user_id"] = str(user_id)
        
        self.collector.increment("api_error_events", tags=tags)
        
        # Métriques spécifiques par type d'erreur
        if status_code >= 500:
            self.collector.increment("api_server_errors", tags=tags)
        elif status_code >= 400:
            self.collector.increment("api_client_errors", tags=tags)
    
    def get_api_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Retourne un résumé des métriques API"""
        
        since = datetime.now() - timedelta(hours=hours)
        
        # Métriques de base
        total_requests = self.collector.get_metric_stats("api_request_count", since).get("sum", 0)
        total_errors = self.collector.get_metric_stats("api_error_count", since).get("sum", 0)
        
        # Calcul du taux d'erreur
        error_rate = (total_errors / total_requests * 100) if total_requests > 0 else 0
        
        # Durées moyennes
        avg_duration = self.collector.get_metric_stats("api_request_duration_ms", since).get("avg", 0)
        
        # Métriques de recherche spécifiques
        search_requests = self.collector.get_metric_stats("api_search_count", since).get("sum", 0)
        search_success = self.collector.get_metric_stats("api_search_success_count", since).get("sum", 0)
        search_success_rate = (search_success / search_requests * 100) if search_requests > 0 else 0
        
        return {
            "period_hours": hours,
            "timestamp": datetime.now().isoformat(),
            "total_requests": total_requests,
            "total_errors": total_errors,
            "error_rate_percent": error_rate,
            "avg_duration_ms": avg_duration,
            "search": {
                "requests": search_requests,
                "success_rate_percent": search_success_rate
            },
            "health": {
                "current_percentage": self.collector.get_current_value("api_health_percentage", 100)
            }
        }
# === DASHBOARD ET REPORTING ===

class MetricsDashboard:
    """Générateur de dashboard pour les métriques"""
    
    def __init__(self, collector: MetricsCollector, alert_manager: AlertManager):
        self.collector = collector
        self.alert_manager = alert_manager
    
    def generate_health_report(self) -> Dict[str, Any]:
        """Génère un rapport de santé complet"""
        
        current_time = datetime.now()
        
        # Statut global
        active_alerts = self.alert_manager.get_active_alerts()
        critical_alerts = [a for a in active_alerts if a["level"] == "critical"]
        error_alerts = [a for a in active_alerts if a["level"] == "error"]
        
        if critical_alerts:
            overall_status = "critical"
        elif error_alerts:
            overall_status = "degraded"
        elif active_alerts:
            overall_status = "warning"
        else:
            overall_status = "healthy"
        
        # Métriques de performance clés
        key_performance = {}
        performance_metrics = [
            "lexical_search_duration_ms",
            "elasticsearch_search_duration_ms",
            "api_request_duration_ms"
        ]
        
        for metric in performance_metrics:
            stats = self.collector.get_metric_stats(metric, 
                                                   since=current_time - timedelta(hours=1))
            key_performance[metric] = {
                "avg_ms": stats.get("avg", 0),
                "p95_ms": stats.get("p95", 0),
                "count": stats.get("count", 0)
            }
        
        # Taux d'erreur
        error_count = self.collector.get_current_value("elasticsearch_error_count", 0)
        total_requests = self.collector.get_current_value("elasticsearch_search_count", 1)
        error_rate = (error_count / total_requests) * 100 if total_requests > 0 else 0
        
        return {
            "timestamp": current_time.isoformat(),
            "overall_status": overall_status,
            "alerts": {
                "active_count": len(active_alerts),
                "critical_count": len(critical_alerts),
                "error_count": len(error_alerts),
                "details": active_alerts
            },
            "performance": key_performance,
            "reliability": {
                "error_rate_percent": error_rate,
                "cache_hit_rate_percent": self.collector.get_current_value(
                    "elasticsearch_cache_hit_rate", 0
                ),
                "uptime_hours": (current_time - self.collector._start_time).total_seconds() / 3600
            },
            "system_resources": {
                "memory_usage_mb": psutil.Process().memory_info().rss / (1024 * 1024),
                "cpu_usage_percent": psutil.Process().cpu_percent()
            }
        }
    
    def generate_performance_trends(self, hours: int = 24) -> Dict[str, Any]:
        """Génère les tendances de performance"""
        
        since = datetime.now() - timedelta(hours=hours)
        
        trends = {
            "period_hours": hours,
            "generated_at": datetime.now().isoformat(),
            "metrics": {}
        }
        
        # Métriques à analyser pour les tendances
        trend_metrics = [
            "lexical_search_duration_ms",
            "elasticsearch_search_duration_ms",
            "lexical_cache_hit_rate",
            "search_intent_success_rate"
        ]
        
        for metric_name in trend_metrics:
            samples = self.collector.get_samples(metric_name, since)
            
            if len(samples) < 2:
                trends["metrics"][metric_name] = {"trend": "insufficient_data"}
                continue
            
            values = [s.value for s in samples]
            
            # Calculer la tendance (simple régression linéaire)
            n = len(values)
            x_values = list(range(n))
            
            # Calculs pour la régression linéaire
            x_mean = sum(x_values) / n
            y_mean = sum(values) / n
            
            numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, values))
            denominator = sum((x - x_mean) ** 2 for x in x_values)
            
            if denominator != 0:
                slope = numerator / denominator
                
                # Déterminer la direction de la tendance
                if abs(slope) < 0.1:
                    trend_direction = "stable"
                elif slope > 0:
                    trend_direction = "increasing"
                else:
                    trend_direction = "decreasing"
                
                trends["metrics"][metric_name] = {
                    "trend": trend_direction,
                    "slope": slope,
                    "current_value": values[-1],
                    "avg_value": y_mean,
                    "sample_count": n
                }
            else:
                trends["metrics"][metric_name] = {"trend": "stable"}
        
        return trends


# === INSTANCES SPÉCIALISÉES ===

# Métriques principales compatibles avec les imports existants
query_metrics = QueryMetrics(metrics_collector)
result_metrics = ResultMetrics(metrics_collector)

# Métriques spécialisées avec instances
search_metrics = SearchMetrics(metrics_collector)
elasticsearch_metrics = ElasticsearchMetrics(metrics_collector)
lexical_search_metrics = LexicalSearchMetrics(metrics_collector)
business_metrics = BusinessMetrics(metrics_collector)
api_metrics = ApiMetrics(metrics_collector)

# Dashboard
metrics_dashboard = MetricsDashboard(metrics_collector, alert_manager)


# === CALLBACKS D'ALERTE PAR DÉFAUT ===

def log_alert_callback(alert: MetricAlert):
    """Callback pour logger les alertes"""
    level_mapping = {
        AlertLevel.INFO: logger.info,
        AlertLevel.WARNING: logger.warning,
        AlertLevel.ERROR: logger.error,
        AlertLevel.CRITICAL: logger.critical
    }
    
    log_func = level_mapping.get(alert.level, logger.info)
    log_func(f"ALERTE MÉTRIQUE: {alert.message}")


def system_alert_callback(alert: MetricAlert):
    """Callback pour les alertes système critiques"""
    if alert.level == AlertLevel.CRITICAL:
        # Ici on pourrait intégrer avec des systèmes externes
        # comme PagerDuty, Slack, etc.
        logger.critical(f"🚨 ALERTE CRITIQUE SYSTÈME: {alert.message}")
        
        # Exemple d'action automatique
        if "memory" in alert.metric_name.lower():
            logger.info("🧹 Déclenchement du nettoyage automatique de cache")
            cleanup_old_metrics(hours=1)


# Enregistrer les callbacks par défaut
alert_manager.add_alert_callback(AlertLevel.WARNING, log_alert_callback)
alert_manager.add_alert_callback(AlertLevel.ERROR, log_alert_callback)
alert_manager.add_alert_callback(AlertLevel.CRITICAL, log_alert_callback)
alert_manager.add_alert_callback(AlertLevel.CRITICAL, system_alert_callback)


# === FONCTIONS D'INITIALISATION ===

def initialize_metrics_system():
    """Initialise le système de métriques"""
    
    logger.info("🚀 Initialisation du système de métriques Search Service")
    
    # Nettoyer les anciennes métriques au démarrage
    cleanup_old_metrics(hours=settings.metrics_retention_hours if hasattr(settings, 'metrics_retention_hours') else 24)
    
    # Enregistrer les métriques personnalisées si nécessaire
    # (Elles sont déjà enregistrées via _register_default_metrics)
    
    logger.info("✅ Système de métriques initialisé avec succès")


def shutdown_metrics_system():
    """Arrêt propre du système de métriques"""
    
    logger.info("🛑 Arrêt du système de métriques")
    
    # Exporter les métriques finales si configuré
    if hasattr(settings, 'export_metrics_on_shutdown') and settings.export_metrics_on_shutdown:
        try:
            export_metrics_to_file("/tmp/search_service_final_metrics.json")
        except Exception as e:
            logger.error(f"Erreur lors de l'export final des métriques: {e}")
    
    # Désactiver la collecte système
    metrics_collector._system_metrics_enabled = False
    
    logger.info("✅ Système de métriques arrêté")


# === EXPORTS PRINCIPAUX ===

__all__ = [
    # === CLASSES PRINCIPALES ===
    "MetricsCollector",
    "AlertManager", 
    "MetricsDashboard",
    
    # === MÉTRIQUES PRINCIPALES (compatibilité) ===
    "QueryMetrics",
    "ResultMetrics", 
    "query_metrics",
    "result_metrics",
    
    # === MÉTRIQUES SPÉCIALISÉES ===
    "SearchMetrics",
    "ElasticsearchMetrics",
    "LexicalSearchMetrics",
    "BusinessMetrics",
    "PerformanceProfiler",
    
    # === TYPES ET ENUMS ===
    "MetricType",
    "MetricCategory",
    "AlertLevel",
    "MetricDefinition",
    "MetricValue",
    "MetricSample",
    "MetricAlert",
    
    # === INSTANCES GLOBALES ===
    "metrics_collector",
    "alert_manager",
    "query_metrics",
    "result_metrics", 
    "search_metrics",
    "elasticsearch_metrics",
    "lexical_search_metrics",
    "business_metrics",
    "performance_profiler",
    "metrics_dashboard",
    "ApiMetrics",
    "api_metrics",
    
    # === FONCTIONS UTILITAIRES ===
    "get_system_metrics",
    "get_performance_summary",
    "export_metrics_to_file",
    "cleanup_old_metrics",
    "reset_all_counters",
    "get_top_slow_operations",
    "get_error_metrics_summary",
    "initialize_metrics_system",
    "shutdown_metrics_system",

    
    # === CALLBACKS ===
    "log_alert_callback",
    "system_alert_callback"
]


# === INFORMATIONS DU MODULE ===

__version__ = "1.0.0"
__author__ = "Search Service Team"
__description__ = "Système de métriques spécialisées pour le Search Service"

# Auto-initialisation au chargement du module
try:
    initialize_metrics_system()
except Exception as e:
    logger.error(f"Erreur lors de l'auto-initialisation des métriques: {e}")

logger.info(f"Module utils.metrics chargé - version {__version__}")