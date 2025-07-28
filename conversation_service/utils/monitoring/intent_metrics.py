#!/usr/bin/env python3
"""
📊 Métriques Détaillées - Monitoring Performance

Système de métriques complet pour monitoring de la détection d'intention
avec analytics temps réel et alertes de performance.
"""

import time
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import deque, defaultdict
import statistics
import logging
from conversation_service.models.enums import IntentType, DetectionMethod, MetricType
from conversation_service.config import config

logger = logging.getLogger(__name__)


@dataclass
class RequestMetric:
    """Métrique d'une requête individuelle"""
    timestamp: float
    query: str
    intent: str
    confidence: float
    method: str
    processing_time_ms: float
    cost: float = 0.0
    user_id: Optional[str] = None
    entities_count: int = 0
    cached: bool = False
    error: Optional[str] = None


@dataclass
class PerformanceWindow:
    """Fenêtre de performance pour calculs temps réel"""
    window_size: int = 100
    requests: deque = field(default_factory=deque)
    start_time: float = field(default_factory=time.time)
    
    def add_request(self, metric: RequestMetric):
        """Ajoute une requête à la fenêtre"""
        self.requests.append(metric)
        if len(self.requests) > self.window_size:
            self.requests.popleft()
    
    def get_avg_latency(self) -> float:
        """Latence moyenne de la fenêtre"""
        if not self.requests:
            return 0.0
        return statistics.mean([r.processing_time_ms for r in self.requests])
    
    def get_success_rate(self) -> float:
        """Taux de succès de la fenêtre"""
        if not self.requests:
            return 0.0
        successful = sum(1 for r in self.requests if not r.error)
        return successful / len(self.requests)


class IntentMetricsCollector:
    """
    Collecteur de métriques intelligent pour détection d'intention
    
    Fonctionnalités:
    - Métriques temps réel et historiques
    - Fenêtres glissantes pour tendances
    - Alertes automatiques sur dégradation
    - Analytics par intention, méthode, utilisateur
    - Export pour dashboards externes
    """
    
    def __init__(self, history_size: int = 10000):
        self.history_size = history_size
        
        # Historique complet des requêtes
        self._request_history: deque[RequestMetric] = deque(maxlen=history_size)
        
        # Fenêtres glissantes pour différentes analyses
        self._windows = {
            "real_time": PerformanceWindow(100),      # 100 dernières requêtes
            "short_term": PerformanceWindow(1000),    # 1000 dernières requêtes  
            "medium_term": PerformanceWindow(5000)    # 5000 dernières requêtes
        }
        
        # Métriques agrégées par dimension
        self._intent_metrics: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "count": 0,
            "total_latency": 0.0,
            "total_confidence": 0.0,
            "success_count": 0,
            "error_count": 0
        })
        
        self._method_metrics: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "count": 0,
            "total_latency": 0.0,
            "total_cost": 0.0,
            "success_count": 0
        })
        
        self._user_metrics: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "request_count": 0,
            "unique_intents": set(),
            "avg_confidence": 0.0,
            "last_seen": 0.0
        })
        
        # Métriques globales
        self._global_stats = {
            "total_requests": 0,
            "total_errors": 0,
            "total_cost": 0.0,
            "service_start_time": time.time(),
            "peak_requests_per_hour": 0,
            "peak_avg_latency": 0.0
        }
        
        # Seuils d'alerte
        self._alert_thresholds = {
            "max_latency_ms": config.performance.target_latency_ms * 2,
            "min_success_rate": 0.95,
            "max_error_rate": 0.05,
            "max_cost_per_hour": 10.0
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Collecteur de métriques initialisé")
    
    def record_request(
        self,
        query,  # type: str
        intent,  # type: str
        confidence,  # type: float
        method,  # type: str
        processing_time_ms,  # type: float
        cost=0.0,  # type: float
        user_id=None,  # type: Optional[str]
        entities_count=0,  # type: int
        cached=False,  # type: bool
        error=None  # type: Optional[str]
    ):
        """
        Enregistre une nouvelle requête avec toutes ses métriques
        
        Args:
            query: Requête utilisateur
            intent: Intention détectée (IntentType.value)
            confidence: Confiance de détection
            method: Méthode utilisée (DetectionMethod.value)
            processing_time_ms: Temps de traitement
            cost: Coût de la requête
            user_id: ID utilisateur optionnel
            entities_count: Nombre d'entités extraites
            cached: Si résultat du cache
            error: Message d'erreur éventuel
        """
        with self._lock:
            # Validation des types énumérés
            if intent not in [e.value for e in IntentType]:
                logger.warning(f"Intent non reconnu: {intent}")
                intent = IntentType.UNKNOWN.value
            
            if method not in [e.value for e in DetectionMethod]:
                logger.warning(f"Méthode non reconnue: {method}")
                method = DetectionMethod.RULES.value
            # Création métrique
            metric = RequestMetric(
                timestamp=time.time(),
                query=query[:100],  # Tronquer pour confidentialité
                intent=intent,
                confidence=confidence,
                method=method,
                processing_time_ms=processing_time_ms,
                cost=cost,
                user_id=user_id,
                entities_count=entities_count,
                cached=cached,
                error=error
            )
            
            # Ajout à l'historique
            self._request_history.append(metric)
            
            # Mise à jour fenêtres
            for window in self._windows.values():
                window.add_request(metric)
            
            # Mise à jour métriques agrégées
            self._update_intent_metrics(metric)
            self._update_method_metrics(metric)
            if user_id:
                self._update_user_metrics(metric)
            self._update_global_stats(metric)
            
            # Vérification alertes
            self._check_alerts(metric)
    
    def _update_intent_metrics(self, metric: RequestMetric):
        """Met à jour métriques par intention"""
        intent = metric.intent
        metrics = self._intent_metrics[intent]
        
        metrics["count"] += 1
        metrics["total_latency"] += metric.processing_time_ms
        metrics["total_confidence"] += metric.confidence
        
        if metric.error:
            metrics["error_count"] += 1
        else:
            metrics["success_count"] += 1
    
    def _update_method_metrics(self, metric: RequestMetric):
        """Met à jour métriques par méthode"""
        method = metric.method
        metrics = self._method_metrics[method]
        
        metrics["count"] += 1
        metrics["total_latency"] += metric.processing_time_ms
        metrics["total_cost"] += metric.cost
        
        if not metric.error:
            metrics["success_count"] += 1
    
    def _update_user_metrics(self, metric: RequestMetric):
        """Met à jour métriques par utilisateur"""
        if not metric.user_id:
            return
        
        user_id = str(metric.user_id)
        metrics = self._user_metrics[user_id]
        
        metrics["request_count"] += 1
        metrics["unique_intents"].add(metric.intent)
        metrics["last_seen"] = metric.timestamp
        
        # Moyenne mobile de confiance
        if metrics["request_count"] == 1:
            metrics["avg_confidence"] = metric.confidence
        else:
            alpha = 0.1  # Facteur de lissage
            metrics["avg_confidence"] = (
                alpha * metric.confidence + 
                (1 - alpha) * metrics["avg_confidence"]
            )
    
    def _update_global_stats(self, metric: RequestMetric):
        """Met à jour statistiques globales"""
        self._global_stats["total_requests"] += 1
        self._global_stats["total_cost"] += metric.cost
        
        if metric.error:
            self._global_stats["total_errors"] += 1
        
        # Calcul pic de latence
        if metric.processing_time_ms > self._global_stats["peak_avg_latency"]:
            self._global_stats["peak_avg_latency"] = metric.processing_time_ms
    
    def _check_alerts(self, metric: RequestMetric):
        """Vérifie seuils d'alerte et log warnings"""
        # Alerte latence
        if metric.processing_time_ms > self._alert_thresholds["max_latency_ms"]:
            self.logger.warning(
                f"🚨 ALERTE LATENCE: {metric.processing_time_ms:.1f}ms "
                f"(seuil: {self._alert_thresholds['max_latency_ms']}ms) "
                f"pour '{metric.query[:50]}...'"
            )
        
        # Alerte taux de succès (sur fenêtre courte)
        success_rate = self._windows["real_time"].get_success_rate()
        if success_rate < self._alert_thresholds["min_success_rate"]:
            self.logger.warning(
                f"🚨 ALERTE TAUX SUCCÈS: {success_rate:.3f} "
                f"(seuil: {self._alert_thresholds['min_success_rate']}) "
                f"sur les 100 dernières requêtes"
            )
    
    def get_real_time_metrics(self) -> Dict[str, Any]:
        """Métriques temps réel (100 dernières requêtes)"""
        with self._lock:
            window = self._windows["real_time"]
            
            if not window.requests:
                return {"status": "no_data", "requests_count": 0}
            
            requests = list(window.requests)
            
            # Calculs temps réel
            avg_latency = statistics.mean([r.processing_time_ms for r in requests])
            success_rate = sum(1 for r in requests if not r.error) / len(requests)
            avg_confidence = statistics.mean([r.confidence for r in requests])
            
            # Distribution méthodes
            method_dist = defaultdict(int)
            for r in requests:
                method_dist[r.method] += 1
            
            # Distribution intentions
            intent_dist = defaultdict(int)
            for r in requests:
                intent_dist[r.intent] += 1
            
            return {
                "status": "active",
                "window_size": len(requests),
                "time_range_minutes": (time.time() - requests[0].timestamp) / 60,
                "performance": {
                    "avg_latency_ms": round(avg_latency, 2),
                    "success_rate": round(success_rate, 3),
                    "avg_confidence": round(avg_confidence, 3),
                    "requests_per_minute": round(len(requests) / max(1, (time.time() - requests[0].timestamp) / 60), 1)
                },
                "distribution": {
                    "by_method": dict(method_dist),
                    "by_intent": dict(intent_dist)
                },
                "quality_indicators": {
                    "high_confidence_rate": round(
                        sum(1 for r in requests if r.confidence >= 0.8) / len(requests), 3
                    ),
                    "cache_hit_rate": round(
                        sum(1 for r in requests if r.cached) / len(requests), 3
                    ),
                    "avg_entities_per_request": round(
                        statistics.mean([r.entities_count for r in requests]), 1
                    )
                }
            }
    
    def get_historical_metrics(self, hours: int = 24) -> Dict[str, Any]:
        """Métriques historiques sur période donnée"""
        with self._lock:
            cutoff_time = time.time() - (hours * 3600)
            recent_requests = [
                r for r in self._request_history 
                if r.timestamp >= cutoff_time
            ]
            
            if not recent_requests:
                return {"status": "no_data", "period_hours": hours}
            
            # Calculs historiques
            total_requests = len(recent_requests)
            successful_requests = sum(1 for r in recent_requests if not r.error)
            total_cost = sum(r.cost for r in recent_requests)
            
            latencies = [r.processing_time_ms for r in recent_requests]
            confidences = [r.confidence for r in recent_requests]
            
            return {
                "status": "available",
                "period_hours": hours,
                "total_requests": total_requests,
                "request_volume": {
                    "requests_per_hour": round(total_requests / hours, 1),
                    "peak_hour_requests": self._calculate_peak_hour_requests(recent_requests),
                    "requests_trend": self._calculate_trend(recent_requests)
                },
                "performance_summary": {
                    "avg_latency_ms": round(statistics.mean(latencies), 2),
                    "p50_latency_ms": round(statistics.median(latencies), 2),
                    "p95_latency_ms": round(statistics.quantiles(latencies, n=20)[18], 2) if len(latencies) > 20 else round(max(latencies), 2),
                    "p99_latency_ms": round(statistics.quantiles(latencies, n=100)[98], 2) if len(latencies) > 100 else round(max(latencies), 2),
                    "success_rate": round(successful_requests / total_requests, 3),
                    "avg_confidence": round(statistics.mean(confidences), 3)
                },
                "cost_analysis": {
                    "total_cost": round(total_cost, 4),
                    "cost_per_request": round(total_cost / total_requests, 6),
                    "cost_per_hour": round(total_cost / hours, 4),
                    "projected_monthly_cost": round(total_cost * 24 * 30 / hours, 2)
                },
                "quality_metrics": {
                    "high_confidence_rate": round(
                        sum(1 for r in recent_requests if r.confidence >= 0.8) / total_requests, 3
                    ),
                    "low_confidence_rate": round(
                        sum(1 for r in recent_requests if r.confidence < 0.5) / total_requests, 3
                    ),
                    "error_rate": round(
                        sum(1 for r in recent_requests if r.error) / total_requests, 3
                    )
                }
            }
    
    def get_intent_analytics(self) -> Dict[str, Any]:
        """Analytics détaillées par intention"""
        with self._lock:
            analytics = {}
            
            for intent, metrics in self._intent_metrics.items():
                if metrics["count"] == 0:
                    continue
                
                avg_latency = metrics["total_latency"] / metrics["count"]
                avg_confidence = metrics["total_confidence"] / metrics["count"]
                success_rate = metrics["success_count"] / metrics["count"]
                
                analytics[intent] = {
                    "request_count": metrics["count"],
                    "avg_latency_ms": round(avg_latency, 2),
                    "avg_confidence": round(avg_confidence, 3),
                    "success_rate": round(success_rate, 3),
                    "error_count": metrics["error_count"],
                    "performance_grade": self._calculate_performance_grade(
                        avg_latency, avg_confidence, success_rate
                    )
                }
            
            # Classement par performance
            sorted_intents = sorted(
                analytics.items(),
                key=lambda x: x[1]["performance_grade"],
                reverse=True
            )
            
            return {
                "intent_performance": analytics,
                "rankings": {
                    "best_performing": sorted_intents[:3],
                    "needs_attention": [item for item in sorted_intents if item[1]["performance_grade"] < 0.7]
                },
                "distribution": {
                    intent: metrics["count"] 
                    for intent, metrics in self._intent_metrics.items()
                }
            }
    
    def get_method_analytics(self) -> Dict[str, Any]:
        """Analytics détaillées par méthode de détection"""
        with self._lock:
            analytics = {}
            
            for method, metrics in self._method_metrics.items():
                if metrics["count"] == 0:
                    continue
                
                avg_latency = metrics["total_latency"] / metrics["count"]
                avg_cost = metrics["total_cost"] / metrics["count"]
                success_rate = metrics["success_count"] / metrics["count"]
                
                analytics[method] = {
                    "request_count": metrics["count"],
                    "usage_percentage": round(
                        metrics["count"] / self._global_stats["total_requests"] * 100, 1
                    ),
                    "avg_latency_ms": round(avg_latency, 2),
                    "avg_cost": round(avg_cost, 6),
                    "total_cost": round(metrics["total_cost"], 4),
                    "success_rate": round(success_rate, 3),
                    "efficiency_score": self._calculate_method_efficiency(
                        avg_latency, avg_cost, success_rate
                    )
                }
            
            return {
                "method_performance": analytics,
                "efficiency_ranking": sorted(
                    analytics.items(),
                    key=lambda x: x[1]["efficiency_score"],
                    reverse=True
                ),
                "cost_breakdown": {
                    method: analytics[method]["total_cost"]
                    for method in analytics
                },
                "latency_comparison": {
                    method: analytics[method]["avg_latency_ms"]
                    for method in analytics
                }
            }
    
    def get_user_analytics(self, limit: int = 50) -> Dict[str, Any]:
        """Analytics par utilisateur (top users)"""
        with self._lock:
            # Tri par nombre de requêtes
            sorted_users = sorted(
                self._user_metrics.items(),
                key=lambda x: x[1]["request_count"],
                reverse=True
            )[:limit]
            
            user_analytics = {}
            for user_id, metrics in sorted_users:
                user_analytics[user_id] = {
                    "request_count": metrics["request_count"],
                    "unique_intents": len(metrics["unique_intents"]),
                    "avg_confidence": round(metrics["avg_confidence"], 3),
                    "last_seen": metrics["last_seen"],
                    "days_since_last_seen": round(
                        (time.time() - metrics["last_seen"]) / 86400, 1
                    )
                }
            
            return {
                "total_users": len(self._user_metrics),
                "active_users_24h": len([
                    u for u in self._user_metrics.values()
                    if time.time() - u["last_seen"] < 86400
                ]),
                "top_users": user_analytics,
                "user_behavior": {
                    "avg_requests_per_user": round(
                        statistics.mean([
                            u["request_count"] for u in self._user_metrics.values()
                        ]), 1
                    ) if self._user_metrics else 0,
                    "avg_intents_per_user": round(
                        statistics.mean([
                            len(u["unique_intents"]) for u in self._user_metrics.values()
                        ]), 1
                    ) if self._user_metrics else 0
                }
            }
    
    def _calculate_peak_hour_requests(self, requests: List[RequestMetric]) -> int:
        """Calcule le pic de requêtes par heure"""
        if not requests:
            return 0
        
        # Grouper par heure
        hourly_counts = defaultdict(int)
        for request in requests:
            hour = int(request.timestamp // 3600)
            hourly_counts[hour] += 1
        
        return max(hourly_counts.values()) if hourly_counts else 0
    
    def _calculate_trend(self, requests: List[RequestMetric]) -> str:
        """Calcule tendance du volume de requêtes"""
        if len(requests) < 10:
            return "insufficient_data"
        
        # Diviser en deux moitiés temporelles
        mid_point = len(requests) // 2
        first_half = requests[:mid_point]
        second_half = requests[mid_point:]
        
        first_half_rate = len(first_half) / (first_half[-1].timestamp - first_half[0].timestamp)
        second_half_rate = len(second_half) / (second_half[-1].timestamp - second_half[0].timestamp)
        
        change_ratio = second_half_rate / first_half_rate if first_half_rate > 0 else 1
        
        if change_ratio > 1.2:
            return "increasing"
        elif change_ratio < 0.8:
            return "decreasing"
        else:
            return "stable"
    
    def _calculate_performance_grade(
        self, 
        avg_latency: float, 
        avg_confidence: float, 
        success_rate: float
    ) -> float:
        """Calcule note de performance pour une intention"""
        # Pondération: latence 30%, confiance 40%, succès 30%
        latency_score = max(0, 1 - (avg_latency - 50) / 200)  # Optimal à 50ms
        confidence_score = avg_confidence
        success_score = success_rate
        
        return round(
            0.3 * latency_score + 0.4 * confidence_score + 0.3 * success_score, 3
        )
    
    def _calculate_method_efficiency(
        self, 
        avg_latency: float, 
        avg_cost: float, 
        success_rate: float
    ) -> float:
        """Calcule score d'efficacité pour une méthode"""
        # Pondération: vitesse 40%, coût 20%, succès 40%
        speed_score = max(0, 1 - avg_latency / 1000)  # Optimal < 1s
        cost_score = max(0, 1 - avg_cost / 0.01)  # Optimal < 0.01$
        success_score = success_rate
        
        return round(
            0.4 * speed_score + 0.2 * cost_score + 0.4 * success_score, 3
        )
    
    def get_comprehensive_report(self) -> Dict[str, Any]:
        """Rapport complet de tous les métriques"""
        with self._lock:
            service_uptime = time.time() - self._global_stats["service_start_time"]
            
            return {
                "report_timestamp": time.time(),
                "service_info": {
                    "uptime_hours": round(service_uptime / 3600, 2),
                    "total_requests": self._global_stats["total_requests"],
                    "requests_per_hour": round(
                        self._global_stats["total_requests"] / (service_uptime / 3600), 1
                    ) if service_uptime > 0 else 0,
                    "total_cost": round(self._global_stats["total_cost"], 4),
                    "error_count": self._global_stats["total_errors"]
                },
                "real_time_performance": self.get_real_time_metrics(),
                "historical_analysis": self.get_historical_metrics(24),
                "intent_analytics": self.get_intent_analytics(),
                "method_analytics": self.get_method_analytics(),
                "user_analytics": self.get_user_analytics(20),
                "alerts_status": {
                    "latency_threshold": self._alert_thresholds["max_latency_ms"],
                    "success_rate_threshold": self._alert_thresholds["min_success_rate"],
                    "cost_threshold_per_hour": self._alert_thresholds["max_cost_per_hour"]
                }
            }
    
    def get_metrics_by_type(self, metric_type):
        # type: (MetricType) -> Dict[str, Any]
        """
        Retourne métriques spécifiques par type
        
        Args:
            metric_type: Type de métrique à récupérer
            
        Returns:
            Dict avec métriques du type demandé
        """
        with self._lock:
            if metric_type == MetricType.LATENCY:
                return self._get_latency_metrics()
            elif metric_type == MetricType.ACCURACY:
                return self._get_accuracy_metrics()
            elif metric_type == MetricType.API_COST:
                return self._get_cost_metrics()
            elif metric_type == MetricType.INTENT_DISTRIBUTION:
                return {"intent_distribution": self._intent_metrics}
            elif metric_type == MetricType.METHOD_DISTRIBUTION:
                return {"method_distribution": self._method_metrics}
            else:
                return {}
    
    def _get_latency_metrics(self):
        """Métriques de latence détaillées"""
        if not self._request_history:
            return {"no_data": True}
        
        latencies = [r.processing_time_ms for r in self._request_history]
        
        return {
            "avg_latency_ms": round(statistics.mean(latencies), 2),
            "median_latency_ms": round(statistics.median(latencies), 2),
            "p95_latency_ms": round(statistics.quantiles(latencies, n=20)[18], 2) if len(latencies) > 20 else round(max(latencies), 2),
            "p99_latency_ms": round(statistics.quantiles(latencies, n=100)[98], 2) if len(latencies) > 100 else round(max(latencies), 2),
            "min_latency_ms": round(min(latencies), 2),
            "max_latency_ms": round(max(latencies), 2)
        }
    
    def _get_accuracy_metrics(self):
        """Métriques de précision détaillées"""
        if not self._request_history:
            return {"no_data": True}
        
        confidences = [r.confidence for r in self._request_history]
        successful_requests = [r for r in self._request_history if not r.error]
        
        return {
            "avg_confidence": round(statistics.mean(confidences), 3),
            "success_rate": round(len(successful_requests) / len(self._request_history), 3),
            "high_confidence_rate": round(
                sum(1 for c in confidences if c >= 0.8) / len(confidences), 3
            ),
            "low_confidence_rate": round(
                sum(1 for c in confidences if c < 0.5) / len(confidences), 3
            )
        }
    
    def _get_cost_metrics(self):
        """Métriques de coûts détaillées"""
        if not self._request_history:
            return {"no_data": True}
        
        total_cost = sum(r.cost for r in self._request_history)
        
        return {
            "total_cost": round(total_cost, 4),
            "avg_cost_per_request": round(total_cost / len(self._request_history), 6),
            "cost_by_method": {
                method: round(metrics["total_cost"], 4)
                for method, metrics in self._method_metrics.items()
            }
        }
        """Exporte métriques en format CSV"""
        import csv
        import io
        
        with self._lock:
            cutoff_time = time.time() - (hours * 3600)
            recent_requests = [
                r for r in self._request_history 
                if r.timestamp >= cutoff_time
            ]
            
            output = io.StringIO()
            writer = csv.writer(output)
            
            # En-têtes
            writer.writerow([
                'timestamp', 'query', 'intent', 'confidence', 'method',
                'processing_time_ms', 'cost', 'user_id', 'entities_count',
                'cached', 'error'
            ])
            
            # Données
            for request in recent_requests:
                writer.writerow([
                    request.timestamp,
                    request.query,
                    request.intent,
                    request.confidence,
                    request.method,
                    request.processing_time_ms,
                    request.cost,
                    request.user_id,
                    request.entities_count,
                    request.cached,
                    request.error or ''
                ])
            
            return output.getvalue()
    
    def reset_metrics(self):
        """Remet à zéro toutes les métriques"""
        with self._lock:
            self._request_history.clear()
            for window in self._windows.values():
                window.requests.clear()
            
            self._intent_metrics.clear()
            self._method_metrics.clear()
            self._user_metrics.clear()
            
            self._global_stats = {
                "total_requests": 0,
                "total_errors": 0,
                "total_cost": 0.0,
                "service_start_time": time.time(),
                "peak_requests_per_hour": 0,
                "peak_avg_latency": 0.0
            }
            
            self.logger.info("Toutes les métriques ont été remises à zéro")


# Instance singleton du collecteur de métriques
_metrics_collector_instance = None

def get_metrics_collector() -> IntentMetricsCollector:
    """Factory function pour récupérer instance collecteur singleton"""
    global _metrics_collector_instance
    if _metrics_collector_instance is None:
        _metrics_collector_instance = IntentMetricsCollector()
    return _metrics_collector_instance


# Fonction utilitaire d'enregistrement rapide
def record_intent_request(
    query: str,
    result: Dict[str, Any],
    user_id: Optional[str] = None
):
    """Enregistrement rapide d'une requête de détection d'intention"""
    collector = get_metrics_collector()
    
    collector.record_request(
        query=query,
        intent=result.get("intent", "UNKNOWN"),
        confidence=result.get("confidence", 0.0),
        method=result.get("method_used", "unknown"),
        processing_time_ms=result.get("processing_time_ms", 0.0),
        cost=result.get("cost_estimate", 0.0),
        user_id=user_id,
        entities_count=len(result.get("entities", {})),
        cached=result.get("cached", False),
        error=result.get("error")
    )


# Exports publics
__all__ = [
    "IntentMetricsCollector",
    "RequestMetric",
    "PerformanceWindow",
    "get_metrics_collector",
    "record_intent_request"
]