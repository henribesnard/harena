"""
Système de métriques et monitoring pour Conversation Service MVP.

Ce module fournit un système complet de collecte, agrégation et reporting
des métriques de performance pour tous les agents AutoGen, le client DeepSeek
et les opérations du service.

Features :
- Collecte temps réel des métriques agents
- Monitoring performance DeepSeek (tokens, coûts, latence)
- Agrégation intelligente avec histogrammes
- Alertes automatiques sur seuils
- Export métriques pour monitoring externe

Author: Conversation Service Team
Created: 2025-01-31
Version: 1.0.0 MVP
"""

import time
import logging
import threading
from typing import Dict, List, Optional, Any, Tuple, NamedTuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import os
import uuid
import statistics
try:  # pragma: no cover - psutil may not be available in all environments
    import psutil
except Exception:  # pragma: no cover
    psutil = None

logger = logging.getLogger(__name__)


class MetricType(str, Enum):
    """Types de métriques supportées."""
    
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class AlertLevel(str, Enum):
    """Niveaux d'alertes."""
    
    INFO = "info"
    WARNING = "warning"  
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class MetricPoint:
    """Point de métrique individuel avec timestamp."""
    
    name: str
    value: float
    timestamp: float
    labels: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire pour export."""
        return {
            "name": self.name,
            "value": self.value,
            "timestamp": self.timestamp,
            "labels": self.labels,
            "type": self.metric_type.value,
            "datetime": datetime.fromtimestamp(self.timestamp).isoformat()
        }


@dataclass
class Alert:
    """Alerte générée automatiquement."""
    
    id: str
    level: AlertLevel
    message: str
    metric_name: str
    threshold_value: float
    actual_value: float
    timestamp: float
    resolved: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire."""
        return {
            "id": self.id,
            "level": self.level.value,
            "message": self.message,
            "metric_name": self.metric_name,
            "threshold": self.threshold_value,
            "actual": self.actual_value,
            "timestamp": self.timestamp,
            "datetime": datetime.fromtimestamp(self.timestamp).isoformat(),
            "resolved": self.resolved
        }


class TimerContext:
    """Context manager pour mesurer le temps d'exécution."""
    
    def __init__(self, metrics_collector: 'MetricsCollector', operation: str, labels: Dict[str, str] = None):
        self.metrics_collector = metrics_collector
        self.operation = operation
        self.labels = labels or {}
        self.start_time: Optional[float] = None
        self.timer_id: Optional[str] = None
    
    def __enter__(self) -> str:
        """Démarre le timer."""
        self.timer_id = str(uuid.uuid4())
        self.start_time = time.time()
        return self.timer_id
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Termine le timer et enregistre la métrique."""
        if self.start_time is not None:
            duration = time.time() - self.start_time
            self.metrics_collector.record_timer(
                name=f"{self.operation}_duration_ms",
                duration_ms=duration * 1000,
                labels=self.labels
            )


class PerformanceMonitor:
    """
    Moniteur de performance avec timers et seuils.
    
    Features :
    - Timers haute précision
    - Détection automatique d'anomalies
    - Historique des performances
    - Calcul percentiles
    """
    
    def __init__(self, window_size: int = 1000):
        """
        Initialise le moniteur.
        
        Args:
            window_size: Taille de la fenêtre glissante pour les métriques
        """
        self.window_size = window_size
        self._timers: Dict[str, float] = {}
        self._performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self._lock = threading.RLock()
        
        # Seuils d'alerte depuis env vars
        self.performance_threshold_ms = float(os.getenv('PERFORMANCE_ALERT_THRESHOLD_MS', '2000'))
        self.error_rate_threshold = float(os.getenv('ERROR_RATE_ALERT_THRESHOLD', '0.05'))
        
        logger.debug(f"PerformanceMonitor initialized: window={window_size}, threshold={self.performance_threshold_ms}ms")
    
    def start_timer(self, operation: str) -> str:
        """
        Démarre un timer pour une opération.
        
        Args:
            operation: Nom de l'opération
            
        Returns:
            ID du timer
        """
        timer_id = f"{operation}_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
        
        with self._lock:
            self._timers[timer_id] = time.time()
        
        return timer_id
    
    def end_timer(self, timer_id: str) -> float:
        """
        Termine un timer et retourne la durée.
        
        Args:
            timer_id: ID du timer à terminer
            
        Returns:
            Durée en millisecondes
        """
        end_time = time.time()
        
        with self._lock:
            start_time = self._timers.pop(timer_id, None)
            
            if start_time is None:
                logger.warning(f"Timer {timer_id} not found")
                return 0.0
            
            duration_ms = (end_time - start_time) * 1000
            
            # Extraie le nom de l'opération depuis timer_id
            operation = timer_id.split('_')[0] if '_' in timer_id else "unknown"
            self._performance_history[operation].append(duration_ms)
            
            return duration_ms
    
    def get_performance_stats(self, operation: str) -> Dict[str, float]:
        """
        Retourne les statistiques de performance pour une opération.
        
        Args:
            operation: Nom de l'opération
            
        Returns:
            Dictionnaire avec les statistiques
        """
        with self._lock:
            history = list(self._performance_history[operation])
            
            if not history:
                return {
                    "count": 0,
                    "avg_ms": 0.0,
                    "min_ms": 0.0,
                    "max_ms": 0.0,
                    "p50_ms": 0.0,
                    "p95_ms": 0.0,
                    "p99_ms": 0.0
                }
            
            return {
                "count": len(history),
                "avg_ms": statistics.mean(history),
                "min_ms": min(history),
                "max_ms": max(history),
                "p50_ms": statistics.median(history),
                "p95_ms": self._percentile(history, 0.95),
                "p99_ms": self._percentile(history, 0.99)
            }
    
    def _percentile(self, data: List[float], p: float) -> float:
        """Calcule un percentile."""
        if not data:
            return 0.0
        
        sorted_data = sorted(data)
        index = int(len(sorted_data) * p)
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def check_performance_alerts(self, operation: str) -> List[Alert]:
        """
        Vérifie les seuils de performance et génère des alertes.
        
        Args:
            operation: Opération à vérifier
            
        Returns:
            Liste d'alertes générées
        """
        alerts = []
        stats = self.get_performance_stats(operation)
        
        if stats["count"] == 0:
            return alerts
        
        # Alerte latence moyenne
        if stats["avg_ms"] > self.performance_threshold_ms:
            alerts.append(Alert(
                id=str(uuid.uuid4()),
                level=AlertLevel.WARNING,
                message=f"High average latency for {operation}: {stats['avg_ms']:.1f}ms",
                metric_name=f"{operation}_avg_latency",
                threshold_value=self.performance_threshold_ms,
                actual_value=stats["avg_ms"],
                timestamp=time.time()
            ))
        
        # Alerte P99 critique
        if stats["p99_ms"] > self.performance_threshold_ms * 2:
            alerts.append(Alert(
                id=str(uuid.uuid4()),
                level=AlertLevel.ERROR,
                message=f"Critical P99 latency for {operation}: {stats['p99_ms']:.1f}ms",
                metric_name=f"{operation}_p99_latency",
                threshold_value=self.performance_threshold_ms * 2,
                actual_value=stats["p99_ms"],
                timestamp=time.time()
            ))
        
        return alerts


class MetricsCollector:
    """
    Collecteur principal de métriques pour tous les composants.
    
    Features :
    - Collecte multi-threading safe
    - Agrégation automatique
    - Export multiple formats
    - Intégration PerformanceMonitor
    - Génération d'alertes
    """
    
    def __init__(self, collection_interval: int = None):
        """
        Initialise le collecteur.
        
        Args:
            collection_interval: Intervalle de collecte en secondes
        """
        self.collection_interval = collection_interval or int(os.getenv('METRICS_COLLECTION_INTERVAL', '60'))
        self.enabled = os.getenv('ENABLE_METRICS', 'true').lower() == 'true'
        self.start_time = time.time()
        
        # Stockage des métriques
        self._metrics: List[MetricPoint] = []
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.RLock()
        
        # Composants intégrés
        self.performance_monitor = PerformanceMonitor()
        self._alerts: List[Alert] = []
        
        # Cache pour éviter spam d'alertes
        self._alert_cache: Dict[str, float] = {}
        self._alert_cooldown = 300  # 5 minutes
        
        logger.info(f"MetricsCollector initialized: enabled={self.enabled}, interval={self.collection_interval}s")
    
    def record_counter(self, name: str, value: float = 1.0, labels: Dict[str, str] = None) -> None:
        """
        Enregistre une métrique counter (cumulative).
        
        Args:
            name: Nom de la métrique
            value: Valeur à ajouter
            labels: Labels optionnels
        """
        if not self.enabled:
            return
        
        with self._lock:
            key = self._make_key(name, labels)
            self._counters[key] += value
            
            self._metrics.append(MetricPoint(
                name=name,
                value=value,
                timestamp=time.time(),
                labels=labels or {},
                metric_type=MetricType.COUNTER
            ))
    
    def record_gauge(self, name: str, value: float, labels: Dict[str, str] = None) -> None:
        """
        Enregistre une métrique gauge (valeur instantanée).
        
        Args:
            name: Nom de la métrique
            value: Valeur actuelle
            labels: Labels optionnels
        """
        if not self.enabled:
            return
        
        with self._lock:
            key = self._make_key(name, labels)
            self._gauges[key] = value
            
            self._metrics.append(MetricPoint(
                name=name,
                value=value,
                timestamp=time.time(),
                labels=labels or {},
                metric_type=MetricType.GAUGE
            ))
    
    def record_histogram(self, name: str, value: float, labels: Dict[str, str] = None) -> None:
        """
        Enregistre une valeur dans un histogramme.
        
        Args:
            name: Nom de la métrique
            value: Valeur à enregistrer
            labels: Labels optionnels
        """
        if not self.enabled:
            return
        
        with self._lock:
            key = self._make_key(name, labels)
            self._histograms[key].append(value)
            
            self._metrics.append(MetricPoint(
                name=name,
                value=value,
                timestamp=time.time(),
                labels=labels or {},
                metric_type=MetricType.HISTOGRAM
            ))
    
    def record_timer(self, name: str, duration_ms: float, labels: Dict[str, str] = None) -> None:
        """
        Enregistre une durée d'exécution.
        
        Args:
            name: Nom de l'opération
            duration_ms: Durée en millisecondes
            labels: Labels optionnels
        """
        if not self.enabled:
            return
        
        self.record_histogram(name, duration_ms, labels)
        
        # Vérifie les seuils de performance
        if duration_ms > self.performance_monitor.performance_threshold_ms:
            self._maybe_generate_alert(
                name=f"{name}_slow",
                message=f"Slow operation {name}: {duration_ms:.1f}ms",
                threshold=self.performance_monitor.performance_threshold_ms,
                actual=duration_ms,
                level=AlertLevel.WARNING
            )

    def record_request(self, endpoint: str, user_id: int) -> None:
        """Enregistre une requête reçue pour un endpoint."""
        labels = {"endpoint": endpoint, "user_id": str(user_id)}
        self.record_counter("requests_total", 1.0, labels)

    def record_response_time(self, endpoint: str, duration_ms: float) -> None:
        """Enregistre le temps de réponse d'un endpoint."""
        labels = {"endpoint": endpoint}
        self.record_timer("response_time_ms", duration_ms, labels)

    def record_success(self, endpoint: str) -> None:
        """Enregistre un succès pour un endpoint."""
        labels = {"endpoint": endpoint}
        self.record_counter("success_total", 1.0, labels)

    def record_error(self, endpoint: str, error_message: str) -> None:
        """Enregistre une erreur survenue pour un endpoint et loggue le message."""
        labels = {"endpoint": endpoint}
        self.record_counter("errors_total", 1.0, labels)
        logger.error(f"[{endpoint}] {error_message}")

    def record_agent_execution(self, agent_name: str, duration_ms: float, success: bool = True, **labels) -> None:
        """
        Enregistre l'exécution d'un agent AutoGen.
        
        Args:
            agent_name: Nom de l'agent
            duration_ms: Temps d'exécution en ms
            success: Succès de l'opération
            **labels: Labels additionnels
        """
        base_labels = {"agent": agent_name, "success": str(success)}
        base_labels.update(labels)
        
        self.record_counter("agent_executions_total", 1.0, base_labels)
        self.record_timer("agent_execution_duration_ms", duration_ms, base_labels)
        
        if not success:
            self.record_counter("agent_errors_total", 1.0, {"agent": agent_name})
    
    def record_intent_detection(self, method: str, confidence: float, success: bool = True, **labels) -> None:
        """
        Enregistre une détection d'intention.
        
        Args:
            method: Méthode utilisée (rule_based, llm_based, hybrid)
            confidence: Score de confiance
            success: Succès de la détection
            **labels: Labels additionnels
        """
        base_labels = {"method": method, "success": str(success)}
        base_labels.update(labels)
        
        self.record_counter("intent_detections_total", 1.0, base_labels)
        self.record_histogram("intent_confidence", confidence, base_labels)
        
        # Alerte si confiance faible
        if confidence < 0.7:
            self._maybe_generate_alert(
                name="low_intent_confidence",
                message=f"Low intent confidence: {confidence:.2f} with method {method}",
                threshold=0.7,
                actual=confidence,
                level=AlertLevel.WARNING
            )
    
    def record_deepseek_usage(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        duration_ms: float,
        cost_usd: float,
        success: bool = True
    ) -> None:
        """
        Enregistre l'utilisation de DeepSeek.
        
        Args:
            model: Modèle utilisé
            input_tokens: Tokens d'entrée
            output_tokens: Tokens de sortie
            duration_ms: Durée de l'appel
            cost_usd: Coût en USD
            success: Succès de l'appel
        """
        labels = {"model": model, "success": str(success)}
        
        self.record_counter("deepseek_requests_total", 1.0, labels)
        self.record_counter("deepseek_input_tokens_total", input_tokens, labels)
        self.record_counter("deepseek_output_tokens_total", output_tokens, labels)
        self.record_counter("deepseek_cost_usd_total", cost_usd, labels)
        self.record_timer("deepseek_request_duration_ms", duration_ms, labels)
        
        if not success:
            self.record_counter("deepseek_errors_total", 1.0, {"model": model})
    
    def _make_key(self, name: str, labels: Optional[Dict[str, str]]) -> str:
        """Génère une clé unique pour une métrique avec labels."""
        if not labels:
            return name
        
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"
    
    def _maybe_generate_alert(
        self,
        name: str,
        message: str,
        threshold: float,
        actual: float,
        level: AlertLevel
    ) -> None:
        """Génère une alerte si elle n'est pas en cooldown."""
        now = time.time()
        
        # Vérifie le cooldown
        if name in self._alert_cache:
            if now - self._alert_cache[name] < self._alert_cooldown:
                return
        
        # Génère l'alerte
        alert = Alert(
            id=str(uuid.uuid4()),
            level=level,
            message=message,
            metric_name=name,
            threshold_value=threshold,
            actual_value=actual,
            timestamp=now
        )
        
        with self._lock:
            self._alerts.append(alert)
        
        self._alert_cache[name] = now
        logger.warning(f"Alert generated: {message}")
    
    def timer(self, operation: str, labels: Dict[str, str] = None) -> TimerContext:
        """
        Retourne un context manager pour mesurer le temps.
        
        Args:
            operation: Nom de l'opération
            labels: Labels optionnels
            
        Returns:
            TimerContext
        """
        return TimerContext(self, operation, labels)
    
    def get_alerts(self, unresolved_only: bool = True) -> List[Alert]:
        """
        Retourne les alertes.
        
        Args:
            unresolved_only: Ne retourne que les alertes non résolues
            
        Returns:
            Liste d'alertes
        """
        with self._lock:
            if unresolved_only:
                return [alert for alert in self._alerts if not alert.resolved]
            return self._alerts.copy()
    
    def resolve_alert(self, alert_id: str) -> bool:
        """
        Résout une alerte.
        
        Args:
            alert_id: ID de l'alerte
            
        Returns:
            True si trouvée et résolue
        """
        with self._lock:
            for alert in self._alerts:
                if alert.id == alert_id:
                    alert.resolved = True
                    return True
        return False

    def get_summary(self) -> Dict[str, Any]:
        """Retourne un résumé des compteurs, jauges et histogrammes."""
        with self._lock:
            counters_summary = dict(self._counters)
            gauges_summary = dict(self._gauges)
            histograms_summary: Dict[str, Dict[str, float]] = {}

            total_requests = 0.0
            response_times: List[float] = []

            for key, values in self._histograms.items():
                if values:
                    histograms_summary[key] = {
                        "count": len(values),
                        "min": min(values),
                        "max": max(values),
                        "avg": statistics.mean(values)
                    }

                    if key.startswith("response_time_ms"):
                        response_times.extend(values)

            for key, value in self._counters.items():
                if key.startswith("requests_total"):
                    total_requests += value

        avg_response_time = statistics.mean(response_times) if response_times else 0

        return {
            "counters": counters_summary,
            "gauges": gauges_summary,
            "histograms": histograms_summary,
            "total_requests": total_requests,
            "avg_response_time": avg_response_time
        }

    def get_memory_usage(self) -> Dict[str, float]:
        """Retourne l'utilisation mémoire du processus."""
        if psutil is None:  # pragma: no cover
            return {}

        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        return {
            "rss": mem_info.rss,
            "vms": mem_info.vms,
            "percent": psutil.virtual_memory().percent
        }

    def get_cpu_usage(self) -> float:
        """Retourne l'utilisation CPU en pourcentage."""
        if psutil is None:  # pragma: no cover
            return 0.0
        return psutil.cpu_percent(interval=None)
    
    def get_performance_report(self) -> Dict[str, Any]:
        """
        Génère un rapport de performance complet.
        
        Returns:
            Rapport détaillé avec toutes les métriques
        """
        with self._lock:
            # Statistiques de base
            total_metrics = len(self._metrics)
            active_alerts = len([a for a in self._alerts if not a.resolved])
            
            # Métriques par type
            counters_summary = {k: v for k, v in self._counters.items()}
            gauges_summary = {k: v for k, v in self._gauges.items()}
            
            # Histogrammes avec statistiques
            histograms_summary = {}
            for key, values in self._histograms.items():
                if values:
                    histograms_summary[key] = {
                        "count": len(values),
                        "min": min(values),
                        "max": max(values),
                        "avg": statistics.mean(values),
                        "p50": statistics.median(values),
                        "p95": self.performance_monitor._percentile(values, 0.95),
                        "p99": self.performance_monitor._percentile(values, 0.99)
                    }
            
            return {
                "summary": {
                    "total_metrics": total_metrics,
                    "active_alerts": active_alerts,
                    "collection_enabled": self.enabled,
                    "collection_interval": self.collection_interval,
                    "report_generated_at": datetime.now().isoformat()
                },
                "counters": counters_summary,
                "gauges": gauges_summary,
                "histograms": histograms_summary,
                "alerts": [alert.to_dict() for alert in self._alerts[-10:]],  # 10 dernières alertes
                "performance": {
                    name: self.performance_monitor.get_performance_stats(name)
                    for name in ["agent_execution", "intent_detection", "deepseek_request"]
                }
            }
    
    def export_metrics(self, format: str = "json") -> str:
        """
        Exporte les métriques dans un format spécifique.
        
        Args:
            format: Format d'export ("json", "prometheus")
            
        Returns:
            String formatée
        """
        if format == "json":
            import json
            report = self.get_performance_report()
            return json.dumps(report, indent=2, ensure_ascii=False)
        
        elif format == "prometheus":
            # Format Prometheus simple
            lines = []
            
            with self._lock:
                # Counters
                for key, value in self._counters.items():
                    lines.append(f"{key} {value}")
                
                # Gauges  
                for key, value in self._gauges.items():
                    lines.append(f"{key} {value}")
            
            return "\n".join(lines)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def clear_metrics(self) -> None:
        """Vide toutes les métriques (utile pour tests)."""
        with self._lock:
            self._metrics.clear()
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()
            self._alerts.clear()
            self._alert_cache.clear()
        
        logger.info("All metrics cleared")


class MetricsAggregator:
    """
    Agrégateur de métriques avec fenêtres temporelles.
    
    Utile pour créer des dashboards et rapports périodiques.
    """
    
    def __init__(self, metrics_collector: MetricsCollector):
        """
        Initialise l'agrégateur.
        
        Args:
            metrics_collector: Collecteur source
        """
        self.metrics_collector = metrics_collector
        self._aggregation_cache = {}
        self._cache_ttl = 60  # 1 minute
    
    def aggregate_by_time_window(
        self,
        window_minutes: int = 5,
        metric_names: List[str] = None
    ) -> Dict[str, Any]:
        """
        Agrège les métriques par fenêtre temporelle.
        
        Args:
            window_minutes: Taille de la fenêtre en minutes
            metric_names: Noms des métriques à inclure (None = toutes)
            
        Returns:
            Métriques agrégées par fenêtre
        """
        cache_key = f"time_window_{window_minutes}_{hash(tuple(metric_names or []))}"
        now = time.time()
        
        # Vérifie le cache
        if cache_key in self._aggregation_cache:
            cached_data, cache_time = self._aggregation_cache[cache_key]
            if now - cache_time < self._cache_ttl:
                return cached_data
        
        # Calcule l'agrégation
        window_start = now - (window_minutes * 60)
        
        with self.metrics_collector._lock:
            relevant_metrics = [
                m for m in self.metrics_collector._metrics
                if m.timestamp >= window_start and (
                    metric_names is None or m.name in metric_names
                )
            ]
        
        # Groupe par nom de métrique
        grouped = defaultdict(list)
        for metric in relevant_metrics:
            grouped[metric.name].append(metric)
        
        # Calcule les statistiques par groupe
        result = {}
        for name, metrics in grouped.items():
            values = [m.value for m in metrics]
            if values:
                result[name] = {
                    "count": len(values),
                    "sum": sum(values),
                    "avg": statistics.mean(values),
                    "min": min(values),
                    "max": max(values),
                    "first_timestamp": min(m.timestamp for m in metrics),
                    "last_timestamp": max(m.timestamp for m in metrics)
                }
        
        # Met en cache
        self._aggregation_cache[cache_key] = (result, now)
        
        return result
    
    def get_top_metrics(self, metric_type: str = "counter", limit: int = 10) -> List[Tuple[str, float]]:
        """
        Retourne les top métriques par valeur.
        
        Args:
            metric_type: Type de métrique à considérer
            limit: Nombre max de résultats
            
        Returns:
            Liste de tuples (nom, valeur) triée
        """
        with self.metrics_collector._lock:
            if metric_type == "counter":
                items = list(self.metrics_collector._counters.items())
            elif metric_type == "gauge":
                items = list(self.metrics_collector._gauges.items())
            else:
                return []
        
        # Trie par valeur décroissante
        sorted_items = sorted(items, key=lambda x: x[1], reverse=True)
        return sorted_items[:limit]


# Instance globale par défaut
_default_metrics_collector: Optional[MetricsCollector] = None

def get_default_metrics_collector() -> MetricsCollector:
    """Retourne l'instance de collecteur par défaut."""
    global _default_metrics_collector
    
    if _default_metrics_collector is None:
        _default_metrics_collector = MetricsCollector()
    return _default_metrics_collector
