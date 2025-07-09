"""
Système de métriques pour le Search Service.

Collecte, agrège et analyse les métriques de performance
pour le monitoring et l'optimisation du système.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
import threading

from ..config.settings import SearchServiceSettings, get_settings


logger = logging.getLogger(__name__)


class MetricType(str, Enum):
    """Types de métriques."""
    COUNTER = "counter"
    GAUGE = "gauge"
"""
Système de métriques pour le Search Service.

Collecte, agrège et analyse les métriques de performance
pour le monitoring et l'optimisation du système.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
import threading
import statistics

from ..config.settings import SearchServiceSettings, get_settings


logger = logging.getLogger(__name__)


class MetricType(str, Enum):
    """Types de métriques."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class MetricValue:
    """Valeur de métrique avec timestamp."""
    value: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class MetricSummary:
    """Résumé statistique d'une métrique."""
    count: int
    sum: float
    min: float
    max: float
    avg: float
    p50: float
    p95: float
    p99: float


class SearchMetrics:
    """
    Système de métriques pour les opérations de recherche.
    
    Collecte et analyse les métriques de performance des recherches:
    - Temps d'exécution
    - Qualité des résultats
    - Cache hit rates
    - Taux d'erreur
    """
    
    def __init__(self, settings: Optional[SearchServiceSettings] = None):
        self.settings = settings or get_settings()
        
        # Stockage des métriques avec retention
        self.retention_hours = 24
        self.max_values_per_metric = 10000
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Métriques de recherche
        self.search_durations: deque = deque(maxlen=self.max_values_per_metric)
        self.search_counts = defaultdict(int)
        self.quality_scores: deque = deque(maxlen=self.max_values_per_metric)
        self.error_counts = defaultdict(int)
        
        # Métriques par intention
        self.metrics_by_intent = defaultdict(lambda: {
            "durations": deque(maxlen=1000),
            "count": 0,
            "errors": 0
        })
        
        # Métriques par complexité
        self.metrics_by_complexity = defaultdict(lambda: {
            "durations": deque(maxlen=1000),
            "count": 0
        })
        
        logger.info("Search metrics initialized")
    
    def record_search(
        self,
        execution_time_ms: float,
        quality: str,
        complexity: str,
        intent: Optional[str] = None,
        success: bool = True,
        cache_hit: bool = False,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Enregistre une métrique de recherche.
        
        Args:
            execution_time_ms: Temps d'exécution en ms
            quality: Niveau de qualité (excellent, good, fair, poor)
            complexity: Complexité de la requête
            intent: Type d'intention (optionnel)
            success: Succès de la recherche
            cache_hit: Hit de cache
            labels: Labels additionnels
        """
        with self._lock:
            timestamp = datetime.utcnow()
            
            # Métriques globales
            self.search_durations.append(MetricValue(execution_time_ms, timestamp, labels or {}))
            self.search_counts["total"] += 1
            
            if success:
                self.search_counts["success"] += 1
            else:
                self.search_counts["error"] += 1
                self.error_counts["search_error"] += 1
            
            if cache_hit:
                self.search_counts["cache_hit"] += 1
            else:
                self.search_counts["cache_miss"] += 1
            
            # Métriques de qualité
            quality_score = self._quality_to_score(quality)
            self.quality_scores.append(MetricValue(quality_score, timestamp))
            
            # Métriques par intention
            if intent:
                intent_metrics = self.metrics_by_intent[intent]
                intent_metrics["durations"].append(MetricValue(execution_time_ms, timestamp))
                intent_metrics["count"] += 1
                if not success:
                    intent_metrics["errors"] += 1
            
            # Métriques par complexité
            complexity_metrics = self.metrics_by_complexity[complexity]
            complexity_metrics["durations"].append(MetricValue(execution_time_ms, timestamp))
            complexity_metrics["count"] += 1
            
            # Nettoyage périodique
            if self.search_counts["total"] % 1000 == 0:
                self._cleanup_old_metrics()
    
    def get_search_summary(self, hours: int = 1) -> Dict[str, Any]:
        """
        Récupère un résumé des métriques de recherche.
        
        Args:
            hours: Période en heures
            
        Returns:
            Dict: Résumé des métriques
        """
        with self._lock:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            
            # Filtrage par période
            recent_durations = [
                mv.value for mv in self.search_durations
                if mv.timestamp >= cutoff_time
            ]
            
            recent_quality = [
                mv.value for mv in self.quality_scores
                if mv.timestamp >= cutoff_time
            ]
            
            if not recent_durations:
                return {"period_hours": hours, "searches": 0}
            
            # Calcul des statistiques
            duration_summary = self._calculate_summary(recent_durations)
            quality_summary = self._calculate_summary(recent_quality)
            
            # Taux de succès et cache
            total_searches = len(recent_durations)
            cache_hit_rate = self.search_counts.get("cache_hit", 0) / max(total_searches, 1)
            success_rate = self.search_counts.get("success", 0) / max(total_searches, 1)
            
            return {
                "period_hours": hours,
                "searches": total_searches,
                "duration_ms": duration_summary.__dict__,
                "quality_score": quality_summary.__dict__,
                "cache_hit_rate": cache_hit_rate,
                "success_rate": success_rate,
                "error_count": self.search_counts.get("error", 0)
            }
    
    def get_intent_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Récupère les métriques par intention."""
        with self._lock:
            intent_metrics = {}
            
            for intent, metrics in self.metrics_by_intent.items():
                durations = [mv.value for mv in metrics["durations"]]
                
                if durations:
                    duration_summary = self._calculate_summary(durations)
                    intent_metrics[intent] = {
                        "count": metrics["count"],
                        "errors": metrics["errors"],
                        "error_rate": metrics["errors"] / max(metrics["count"], 1),
                        "duration_ms": duration_summary.__dict__
                    }
            
            return intent_metrics
    
    def get_complexity_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Récupère les métriques par complexité."""
        with self._lock:
            complexity_metrics = {}
            
            for complexity, metrics in self.metrics_by_complexity.items():
                durations = [mv.value for mv in metrics["durations"]]
                
                if durations:
                    duration_summary = self._calculate_summary(durations)
                    complexity_metrics[complexity] = {
                        "count": metrics["count"],
                        "duration_ms": duration_summary.__dict__
                    }
            
            return complexity_metrics
    
    def _quality_to_score(self, quality: str) -> float:
        """Convertit un niveau de qualité en score numérique."""
        quality_mapping = {
            "excellent": 1.0,
            "good": 0.8,
            "fair": 0.6,
            "poor": 0.4
        }
        return quality_mapping.get(quality.lower(), 0.5)
    
    def _calculate_summary(self, values: List[float]) -> MetricSummary:
        """Calcule un résumé statistique."""
        if not values:
            return MetricSummary(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        
        sorted_values = sorted(values)
        count = len(values)
        
        return MetricSummary(
            count=count,
            sum=sum(values),
            min=min(values),
            max=max(values),
            avg=statistics.mean(values),
            p50=statistics.median(values),
            p95=sorted_values[int(count * 0.95)] if count > 0 else 0.0,
            p99=sorted_values[int(count * 0.99)] if count > 0 else 0.0
        )
    
    def _cleanup_old_metrics(self) -> None:
        """Nettoie les anciennes métriques."""
        cutoff_time = datetime.utcnow() - timedelta(hours=self.retention_hours)
        
        # Nettoyage des métriques principales
        self.search_durations = deque(
            (mv for mv in self.search_durations if mv.timestamp >= cutoff_time),
            maxlen=self.max_values_per_metric
        )
        
        self.quality_scores = deque(
            (mv for mv in self.quality_scores if mv.timestamp >= cutoff_time),
            maxlen=self.max_values_per_metric
        )
    
    def reset(self) -> None:
        """Remet à zéro toutes les métriques."""
        with self._lock:
            self.search_durations.clear()
            self.search_counts.clear()
            self.quality_scores.clear()
            self.error_counts.clear()
            self.metrics_by_intent.clear()
            self.metrics_by_complexity.clear()
        
        logger.info("Search metrics reset")


class QueryExecutionMetrics:
    """Métriques spécialisées pour l'exécution de requêtes."""
    
    def __init__(self):
        self._lock = threading.RLock()
        self.execution_times: deque = deque(maxlen=5000)
        self.priority_metrics = defaultdict(lambda: {
            "count": 0,
            "total_time": 0.0,
            "errors": 0
        })
        self.parallel_execution_count = 0
        self.retry_count = 0
    
    def record_execution(
        self,
        execution_time_ms: float,
        priority: str,
        success: bool
    ) -> None:
        """Enregistre une exécution de requête."""
        with self._lock:
            timestamp = datetime.utcnow()
            self.execution_times.append(MetricValue(execution_time_ms, timestamp))
            
            priority_metric = self.priority_metrics[priority]
            priority_metric["count"] += 1
            priority_metric["total_time"] += execution_time_ms
            
            if not success:
                priority_metric["errors"] += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Récupère les métriques d'exécution."""
        with self._lock:
            if not self.execution_times:
                return {}
            
            recent_times = [mv.value for mv in self.execution_times]
            execution_summary = self._calculate_summary(recent_times)
            
            priority_stats = {}
            for priority, metrics in self.priority_metrics.items():
                if metrics["count"] > 0:
                    priority_stats[priority] = {
                        "count": metrics["count"],
                        "avg_time_ms": metrics["total_time"] / metrics["count"],
                        "error_rate": metrics["errors"] / metrics["count"]
                    }
            
            return {
                "execution_summary": execution_summary.__dict__,
                "priority_stats": priority_stats,
                "parallel_executions": self.parallel_execution_count,
                "retry_count": self.retry_count
            }
    
    def _calculate_summary(self, values: List[float]) -> MetricSummary:
        """Calcule un résumé statistique."""
        if not values:
            return MetricSummary(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        
        sorted_values = sorted(values)
        count = len(values)
        
        return MetricSummary(
            count=count,
            sum=sum(values),
            min=min(values),
            max=max(values),
            avg=statistics.mean(values),
            p50=statistics.median(values),
            p95=sorted_values[int(count * 0.95)] if count > 0 else 0.0,
            p99=sorted_values[int(count * 0.99)] if count > 0 else 0.0
        )
    
    def reset(self) -> None:
        """Remet à zéro les métriques."""
        with self._lock:
            self.execution_times.clear()
            self.priority_metrics.clear()
            self.parallel_execution_count = 0
            self.retry_count = 0


class ResultProcessingMetrics:
    """Métriques pour le traitement des résultats."""
    
    def __init__(self):
        self._lock = threading.RLock()
        self.processing_times: deque = deque(maxlen=5000)
        self.results_processed = 0
        self.errors_count = 0
        self.deduplication_stats = {
            "duplicates_removed": 0,
            "total_processed": 0
        }
    
    def record_processing(
        self,
        results_count: int,
        processing_time_ms: float,
        error: bool = False
    ) -> None:
        """Enregistre un traitement de résultats."""
        with self._lock:
            timestamp = datetime.utcnow()
            self.processing_times.append(MetricValue(processing_time_ms, timestamp))
            self.results_processed += results_count
            
            if error:
                self.errors_count += 1
    
    def record_deduplication(self, duplicates_removed: int, total_processed: int) -> None:
        """Enregistre des statistiques de déduplication."""
        with self._lock:
            self.deduplication_stats["duplicates_removed"] += duplicates_removed
            self.deduplication_stats["total_processed"] += total_processed
    
    def get_metrics(self) -> Dict[str, Any]:
        """Récupère les métriques de traitement."""
        with self._lock:
            if not self.processing_times:
                return {}
            
            processing_times = [mv.value for mv in self.processing_times]
            processing_summary = self._calculate_summary(processing_times)
            
            dedup_rate = 0.0
            if self.deduplication_stats["total_processed"] > 0:
                dedup_rate = (
                    self.deduplication_stats["duplicates_removed"] / 
                    self.deduplication_stats["total_processed"]
                )
            
            return {
                "processing_summary_ms": processing_summary.__dict__,
                "results_processed": self.results_processed,
                "errors_count": self.errors_count,
                "error_rate": self.errors_count / max(len(self.processing_times), 1),
                "deduplication_rate": dedup_rate,
                "duplicates_removed": self.deduplication_stats["duplicates_removed"]
            }
    
    def _calculate_summary(self, values: List[float]) -> MetricSummary:
        """Calcule un résumé statistique."""
        if not values:
            return MetricSummary(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        
        sorted_values = sorted(values)
        count = len(values)
        
        return MetricSummary(
            count=count,
            sum=sum(values),
            min=min(values),
            max=max(values),
            avg=statistics.mean(values),
            p50=statistics.median(values),
            p95=sorted_values[int(count * 0.95)] if count > 0 else 0.0,
            p99=sorted_values[int(count * 0.99)] if count > 0 else 0.0
        )
    
    def reset(self) -> None:
        """Remet à zéro les métriques."""
        with self._lock:
            self.processing_times.clear()
            self.results_processed = 0
            self.errors_count = 0
            self.deduplication_stats = {
                "duplicates_removed": 0,
                "total_processed": 0
            }


class PerformanceMetrics:
    """Métriques pour l'optimisation des performances."""
    
    def __init__(self):
        self._lock = threading.RLock()
        self.optimization_attempts = 0
        self.successful_optimizations = 0
        self.performance_gains: List[float] = []
        self.optimization_types = defaultdict(int)
    
    def record_optimization(
        self,
        optimization_type: str,
        performance_gain_percent: float,
        success: bool
    ) -> None:
        """Enregistre une tentative d'optimisation."""
        with self._lock:
            self.optimization_attempts += 1
            self.optimization_types[optimization_type] += 1
            
            if success:
                self.successful_optimizations += 1
                self.performance_gains.append(performance_gain_percent)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Récupère les métriques d'optimisation."""
        with self._lock:
            success_rate = 0.0
            if self.optimization_attempts > 0:
                success_rate = self.successful_optimizations / self.optimization_attempts
            
            avg_gain = 0.0
            if self.performance_gains:
                avg_gain = statistics.mean(self.performance_gains)
            
            return {
                "optimization_attempts": self.optimization_attempts,
                "successful_optimizations": self.successful_optimizations,
                "success_rate": success_rate,
                "average_performance_gain_percent": avg_gain,
                "total_performance_gain_percent": sum(self.performance_gains),
                "optimization_types": dict(self.optimization_types)
            }
    
    def reset(self) -> None:
        """Remet à zéro les métriques."""
        with self._lock:
            self.optimization_attempts = 0
            self.successful_optimizations = 0
            self.performance_gains.clear()
            self.optimization_types.clear()


class MetricsCollector:
    """
    Collecteur central de métriques pour le Search Service.
    
    Agrège toutes les métriques des différents composants
    et fournit une interface unifiée pour le monitoring.
    """
    
    def __init__(self, settings: Optional[SearchServiceSettings] = None):
        self.settings = settings or get_settings()
        
        # Composants de métriques
        self.search_metrics = SearchMetrics(settings)
        self.execution_metrics = QueryExecutionMetrics()
        self.processing_metrics = ResultProcessingMetrics()
        self.performance_metrics = PerformanceMetrics()
        
        # Métriques système
        self.start_time = datetime.utcnow()
        self.health_checks = 0
        self.health_failures = 0
        
        logger.info("Metrics collector initialized")
    
    def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """
        Récupère toutes les métriques du système.
        
        Returns:
            Dict: Métriques complètes
        """
        uptime_seconds = (datetime.utcnow() - self.start_time).total_seconds()
        
        return {
            "system": {
                "uptime_seconds": uptime_seconds,
                "start_time": self.start_time.isoformat(),
                "health_checks": self.health_checks,
                "health_failures": self.health_failures,
                "health_rate": (
                    (self.health_checks - self.health_failures) / max(self.health_checks, 1)
                )
            },
            "search": self.search_metrics.get_search_summary(),
            "search_by_intent": self.search_metrics.get_intent_metrics(),
            "search_by_complexity": self.search_metrics.get_complexity_metrics(),
            "execution": self.execution_metrics.get_metrics(),
            "processing": self.processing_metrics.get_metrics(),
            "performance": self.performance_metrics.get_metrics()
        }
    
    def get_health_metrics(self) -> Dict[str, Any]:
        """Récupère les métriques de santé du système."""
        search_summary = self.search_metrics.get_search_summary(hours=1)
        
        # Seuils de santé
        healthy_success_rate = 0.95
        healthy_avg_time_ms = 500
        healthy_cache_hit_rate = 0.5
        
        # Évaluation de la santé
        is_healthy = True
        health_issues = []
        
        if search_summary.get("success_rate", 1.0) < healthy_success_rate:
            is_healthy = False
            health_issues.append("Low success rate")
        
        duration_stats = search_summary.get("duration_ms", {})
        if duration_stats.get("avg", 0) > healthy_avg_time_ms:
            is_healthy = False
            health_issues.append("High average response time")
        
        if search_summary.get("cache_hit_rate", 1.0) < healthy_cache_hit_rate:
            health_issues.append("Low cache hit rate")  # Warning, pas critique
        
        return {
            "healthy": is_healthy,
            "issues": health_issues,
            "metrics": {
                "success_rate": search_summary.get("success_rate", 1.0),
                "avg_response_time_ms": duration_stats.get("avg", 0),
                "cache_hit_rate": search_summary.get("cache_hit_rate", 1.0),
                "recent_searches": search_summary.get("searches", 0)
            }
        }
    
    def record_health_check(self, success: bool) -> None:
        """Enregistre un health check."""
        self.health_checks += 1
        if not success:
            self.health_failures += 1
    
    def export_prometheus_metrics(self) -> str:
        """
        Exporte les métriques au format Prometheus.
        
        Returns:
            str: Métriques formatées Prometheus
        """
        metrics = self.get_comprehensive_metrics()
        lines = []
        
        # Métriques système
        lines.append(f"search_service_uptime_seconds {metrics['system']['uptime_seconds']}")
        lines.append(f"search_service_health_checks_total {metrics['system']['health_checks']}")
        lines.append(f"search_service_health_failures_total {metrics['system']['health_failures']}")
        
        # Métriques de recherche
        search_metrics = metrics.get('search', {})
        if search_metrics:
            lines.append(f"search_service_searches_total {search_metrics.get('searches', 0)}")
            lines.append(f"search_service_success_rate {search_metrics.get('success_rate', 0)}")
            lines.append(f"search_service_cache_hit_rate {search_metrics.get('cache_hit_rate', 0)}")
            
            duration_stats = search_metrics.get('duration_ms', {})
            if duration_stats:
                lines.append(f"search_service_duration_ms_avg {duration_stats.get('avg', 0)}")
                lines.append(f"search_service_duration_ms_p95 {duration_stats.get('p95', 0)}")
                lines.append(f"search_service_duration_ms_p99 {duration_stats.get('p99', 0)}")
        
        return '\n'.join(lines)
    
    def reset_all_metrics(self) -> None:
        """Remet à zéro toutes les métriques."""
        self.search_metrics.reset()
        self.execution_metrics.reset()
        self.processing_metrics.reset()
        self.performance_metrics.reset()
        
        self.start_time = datetime.utcnow()
        self.health_checks = 0
        self.health_failures = 0
        
        logger.info("All metrics reset")


# === HELPER FUNCTIONS ===

def create_metrics_collector(
    settings: Optional[SearchServiceSettings] = None
) -> MetricsCollector:
    """
    Factory pour créer un collecteur de métriques.
    
    Args:
        settings: Configuration
        
    Returns:
        MetricsCollector: Collecteur configuré
    """
    return MetricsCollector(settings=settings or get_settings())


def format_metrics_for_logging(metrics: Dict[str, Any]) -> str:
    """
    Formate les métriques pour les logs.
    
    Args:
        metrics: Métriques à formater
        
    Returns:
        str: Métriques formatées
    """
    formatted_lines = []
    
    def format_section(name: str, data: Any, indent: int = 0) -> None:
        prefix = "  " * indent
        if isinstance(data, dict):
            formatted_lines.append(f"{prefix}{name}:")
            for key, value in data.items():
                format_section(key, value, indent + 1)
        elif isinstance(data, (int, float)):
            formatted_lines.append(f"{prefix}{name}: {value:.2f}" if isinstance(data, float) else f"{prefix}{name}: {value}")
        else:
            formatted_lines.append(f"{prefix}{name}: {data}")
    
    format_section("Metrics", metrics)
    return "\n".join(formatted_lines)