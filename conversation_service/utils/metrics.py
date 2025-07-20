"""
Système de métriques temps réel pour monitoring et observabilité
Intégration Prometheus/Datadog compatible Heroku
"""

import asyncio
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime, timedelta
import statistics

@dataclass
class MetricSnapshot:
    """Snapshot métrique avec timestamp"""
    timestamp: datetime
    value: float
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass 
class TimeSeriesMetric:
    """Métrique série temporelle avec window glissante"""
    name: str
    values: deque = field(default_factory=lambda: deque(maxlen=1000))
    window_minutes: int = 15
    
    def add_value(self, value: float, labels: Optional[Dict[str, str]] = None):
        """Ajout valeur avec nettoyage automatique"""
        snapshot = MetricSnapshot(
            timestamp=datetime.utcnow(),
            value=value,
            labels=labels or {}
        )
        self.values.append(snapshot)
        self._cleanup_old_values()
    
    def _cleanup_old_values(self):
        """Nettoyage valeurs anciennes"""
        cutoff = datetime.utcnow() - timedelta(minutes=self.window_minutes)
        while self.values and self.values[0].timestamp < cutoff:
            self.values.popleft()
    
    def get_stats(self) -> Dict[str, float]:
        """Statistiques window courante"""
        if not self.values:
            return {"count": 0, "avg": 0, "min": 0, "max": 0, "p95": 0}
        
        values = [v.value for v in self.values]
        return {
            "count": len(values),
            "avg": statistics.mean(values),
            "min": min(values),
            "max": max(values),
            "p95": statistics.quantiles(values, n=20)[18] if len(values) >= 20 else max(values)
        }


class CacheMetrics:
    """
    Métriques cache Redis avec optimisation performance
    """
    
    def __init__(self):
        self.hits = TimeSeriesMetric("cache_hits")
        self.misses = TimeSeriesMetric("cache_misses") 
        self.errors = TimeSeriesMetric("cache_errors")
        self.latencies = TimeSeriesMetric("cache_latencies")
        self.compression_savings = TimeSeriesMetric("compression_savings")
        self.circuit_breaker_hits = TimeSeriesMetric("circuit_breaker_hits")
        
        # Compteurs cumulatifs
        self.total_operations = 0
        self.total_bytes_saved = 0
        
    def record_cache_hit(self, key: str, latency_ms: int):
        """Enregistrement cache hit"""
        self.hits.add_value(1.0, {"key_prefix": key.split(":")[0]})
        self.latencies.add_value(latency_ms)
        self.total_operations += 1
    
    def record_cache_miss(self, key: str):
        """Enregistrement cache miss"""
        self.misses.add_value(1.0, {"key_prefix": key.split(":")[0]})
        self.total_operations += 1
    
    def record_cache_error(self, key: str, error: str):
        """Enregistrement erreur cache"""
        self.errors.add_value(1.0, {
            "key_prefix": key.split(":")[0],
            "error_type": type(Exception(error)).__name__
        })
    
    def record_cache_set(self, key: str, size_bytes: int):
        """Enregistrement stockage cache"""
        self.total_operations += 1
    
    def record_cache_delete(self, key: str):
        """Enregistrement suppression cache"""
        self.total_operations += 1
    
    def record_compression_saving(self, original_size: int, compressed_size: int):
        """Enregistrement économie compression"""
        savings = original_size - compressed_size
        self.compression_savings.add_value(savings)
        self.total_bytes_saved += savings
    
    def record_circuit_breaker_hit(self):
        """Enregistrement circuit breaker activation"""
        self.circuit_breaker_hits.add_value(1.0)
    
    def record_pipeline_operation(self, operation_count: int):
        """Enregistrement opération pipeline"""
        self.total_operations += operation_count
    
    async def get_current_metrics(self) -> Dict[str, Any]:
        """Métriques actuelles formatées"""
        hits_stats = self.hits.get_stats()
        misses_stats = self.misses.get_stats()
        
        total_requests = hits_stats["count"] + misses_stats["count"]
        hit_rate = (hits_stats["count"] / total_requests) if total_requests > 0 else 0
        
        return {
            "cache_performance": {
                "hit_rate": hit_rate,
                "total_requests": total_requests,
                "hits": hits_stats["count"],
                "misses": misses_stats["count"],
                "errors": self.errors.get_stats()["count"]
            },
            "latency": self.latencies.get_stats(),
            "compression": {
                "total_bytes_saved": self.total_bytes_saved,
                "recent_savings": self.compression_savings.get_stats()
            },
            "circuit_breaker": {
                "hits": self.circuit_breaker_hits.get_stats()["count"]
            }
        }


class IntentMetrics:
    """
    Métriques détection d'intention par niveau
    """
    
    def __init__(self):
        self.l0_pattern_metrics = TimeSeriesMetric("l0_pattern")
        self.l1_lightweight_metrics = TimeSeriesMetric("l1_lightweight") 
        self.l2_llm_metrics = TimeSeriesMetric("l2_llm")
        self.error_metrics = TimeSeriesMetric("intent_errors")
        
        # Distribution par intention
        self.intent_distribution = defaultdict(int)
        self.confidence_scores = TimeSeriesMetric("confidence_scores")
        
    def record_intent_detection(
        self, 
        level: str,
        latency_ms: int,
        cache_hit: bool = False,
        intent_type: str = "unknown",
        confidence: float = 0.0
    ):
        """Enregistrement détection intention"""
        
        # Métriques par niveau
        if level == "L0_PATTERN":
            self.l0_pattern_metrics.add_value(latency_ms, {"cache_hit": str(cache_hit)})
        elif level == "L1_LIGHTWEIGHT":
            self.l1_lightweight_metrics.add_value(latency_ms, {"cache_hit": str(cache_hit)})
        elif level == "L2_LLM":
            self.l2_llm_metrics.add_value(latency_ms, {"cache_hit": str(cache_hit)})
        
        # Distribution intentions
        self.intent_distribution[intent_type] += 1
        
        # Scores confiance
        self.confidence_scores.add_value(confidence, {"intent_type": intent_type})
    
    def record_intent_error(self, error: str):
        """Enregistrement erreur intention"""
        self.error_metrics.add_value(1.0, {"error_type": error})
    
    def record_l0_error(self, error: str):
        """Erreur niveau L0"""
        self.error_metrics.add_value(1.0, {"level": "L0", "error": error})
    
    def record_l1_error(self, error: str):
        """Erreur niveau L1"""
        self.error_metrics.add_value(1.0, {"level": "L1", "error": error})
    
    def record_l2_error(self, error: str):
        """Erreur niveau L2"""
        self.error_metrics.add_value(1.0, {"level": "L2", "error": error})
    
    def record_warmup_error(self, error: str):
        """Erreur warmup cache"""
        self.error_metrics.add_value(1.0, {"type": "warmup", "error": error})
    
    async def get_current_metrics(self) -> Dict[str, Any]:
        """Métriques intention actuelles"""
        
        l0_stats = self.l0_pattern_metrics.get_stats()
        l1_stats = self.l1_lightweight_metrics.get_stats()
        l2_stats = self.l2_llm_metrics.get_stats()
        
        total_detections = l0_stats["count"] + l1_stats["count"] + l2_stats["count"]
        
        return {
            "detection_performance": {
                "total_detections": total_detections,
                "l0_percentage": (l0_stats["count"] / total_detections) if total_detections > 0 else 0,
                "l1_percentage": (l1_stats["count"] / total_detections) if total_detections > 0 else 0,
                "l2_percentage": (l2_stats["count"] / total_detections) if total_detections > 0 else 0
            },
            "latency_by_level": {
                "l0_pattern": l0_stats,
                "l1_lightweight": l1_stats, 
                "l2_llm": l2_stats
            },
            "intent_distribution": dict(self.intent_distribution),
            "confidence_scores": self.confidence_scores.get_stats(),
            "errors": self.error_metrics.get_stats()
        }


class RequestMetrics:
    """
    Métriques requêtes conversation et performance globale
    """
    
    def __init__(self):
        self.conversation_latencies = TimeSeriesMetric("conversation_latencies")
        self.endpoint_calls = TimeSeriesMetric("endpoint_calls")
        self.error_rates = TimeSeriesMetric("error_rates")
        
        # Métriques business
        self.daily_conversations = 0
        self.unique_users = set()
        
    async def record_conversation(
        self,
        conversation_id: str,
        intent_type: str,
        processing_time_ms: int,
        confidence_score: float,
        detection_level: str,
        user_id: Optional[int] = None
    ):
        """Enregistrement conversation complète"""
        
        self.conversation_latencies.add_value(
            processing_time_ms,
            {
                "intent_type": intent_type,
                "detection_level": detection_level,
                "confidence_range": self._get_confidence_range(confidence_score)
            }
        )
        
        self.daily_conversations += 1
        
        if user_id:
            self.unique_users.add(user_id)
    
    def record_endpoint_call(self, endpoint: str, status_code: int, duration_ms: int):
        """Enregistrement appel endpoint"""
        self.endpoint_calls.add_value(
            duration_ms,
            {
                "endpoint": endpoint,
                "status_code": str(status_code),
                "status_class": f"{status_code // 100}xx"
            }
        )
        
        if status_code >= 400:
            self.error_rates.add_value(1.0, {"endpoint": endpoint, "status_code": str(status_code)})
    
    def _get_confidence_range(self, confidence: float) -> str:
        """Catégorisation score confiance"""
        if confidence >= 0.9:
            return "very_high"
        elif confidence >= 0.8:
            return "high"
        elif confidence >= 0.7:
            return "medium"
        else:
            return "low"
    
    async def get_current_metrics(self) -> Dict[str, Any]:
        """Métriques requêtes actuelles"""
        
        return {
            "conversation_performance": self.conversation_latencies.get_stats(),
            "endpoint_performance": self.endpoint_calls.get_stats(),
            "error_metrics": self.error_rates.get_stats(),
            "business_metrics": {
                "daily_conversations": self.daily_conversations,
                "unique_users": len(self.unique_users)
            }
        }


class ServiceMetrics:
    """
    Métriques service globales avec agrégation
    """
    
    def __init__(self, redis_manager=None):
        self.redis_manager = redis_manager
        self.startup_times = TimeSeriesMetric("startup_times")
        self.memory_usage = TimeSeriesMetric("memory_usage")
        self.active_connections = TimeSeriesMetric("active_connections")
        
        # Agrégateurs métriques
        self.cache_metrics = CacheMetrics()
        self.intent_metrics = IntentMetrics()
        self.request_metrics = RequestMetrics()
        
    async def record_startup_time(self, duration_ms: int):
        """Enregistrement temps démarrage"""
        self.startup_times.add_value(duration_ms)
    
    async def record_memory_usage(self, memory_mb: float):
        """Enregistrement usage mémoire"""
        self.memory_usage.add_value(memory_mb)
    
    async def record_active_connections(self, count: int):
        """Enregistrement connexions actives"""
        self.active_connections.add_value(count)
    
    async def get_prometheus_metrics(self) -> str:
        """Export métriques format Prometheus"""
        
        metrics_lines = []
        
        # Métriques cache
        cache_metrics = await self.cache_metrics.get_current_metrics()
        metrics_lines.extend([
            f'cache_hit_rate {cache_metrics["cache_performance"]["hit_rate"]}',
            f'cache_latency_p95 {cache_metrics["latency"]["p95"]}',
            f'cache_errors_total {cache_metrics["cache_performance"]["errors"]}'
        ])
        
        # Métriques intention
        intent_metrics = await self.intent_metrics.get_current_metrics()
        metrics_lines.extend([
            f'intent_detection_l0_percentage {intent_metrics["detection_performance"]["l0_percentage"]}',
            f'intent_detection_l1_percentage {intent_metrics["detection_performance"]["l1_percentage"]}',
            f'intent_detection_l2_percentage {intent_metrics["detection_performance"]["l2_percentage"]}',
            f'intent_confidence_p95 {intent_metrics["confidence_scores"]["p95"]}'
        ])
        
        # Métriques requêtes
        request_metrics = await self.request_metrics.get_current_metrics()
        metrics_lines.extend([
            f'conversation_latency_p95 {request_metrics["conversation_performance"]["p95"]}',
            f'daily_conversations_total {request_metrics["business_metrics"]["daily_conversations"]}',
            f'unique_users_total {request_metrics["business_metrics"]["unique_users"]}'
        ])
        
        return '\n'.join(metrics_lines)
    
    async def get_health_summary(self) -> Dict[str, Any]:
        """Résumé santé service"""
        
        cache_metrics = await self.cache_metrics.get_current_metrics()
        intent_metrics = await self.intent_metrics.get_current_metrics()
        request_metrics = await self.request_metrics.get_current_metrics()
        
        # Calcul santé globale
        health_score = self._calculate_health_score(cache_metrics, intent_metrics, request_metrics)
        
        return {
            "overall_health": "healthy" if health_score > 0.8 else "degraded" if health_score > 0.5 else "unhealthy",
            "health_score": health_score,
            "components": {
                "cache": "healthy" if cache_metrics["cache_performance"]["hit_rate"] > 0.7 else "degraded",
                "intent_detection": "healthy" if intent_metrics["detection_performance"]["total_detections"] > 0 else "degraded",
                "api_endpoints": "healthy" if request_metrics["error_metrics"]["count"] < 10 else "degraded"
            },
            "key_metrics": {
                "cache_hit_rate": cache_metrics["cache_performance"]["hit_rate"],
                "avg_response_time": request_metrics["conversation_performance"]["avg"],
                "error_rate": request_metrics["error_metrics"]["count"]
            }
        }
    
    def _calculate_health_score(self, cache_metrics, intent_metrics, request_metrics) -> float:
        """Calcul score santé composite"""
        
        # Pondération composants
        cache_score = min(cache_metrics["cache_performance"]["hit_rate"] * 1.2, 1.0)
        intent_score = 1.0 if intent_metrics["detection_performance"]["total_detections"] > 0 else 0.0
        request_score = max(0.0, 1.0 - (request_metrics["error_metrics"]["count"] / 100))
        
        # Score pondéré
        return (cache_score * 0.3 + intent_score * 0.4 + request_score * 0.3)
