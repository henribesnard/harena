# -*- coding: utf-8 -*-
"""
Performance Monitor - Phase 6 Production
Architecture v2.0 - Monitoring temps reel

Responsabilite : Monitoring performance et metriques temps reel
- Collecte metriques performance pipeline
- Alertes automatiques seuils critiques
- Dashboard temps reel via WebSocket
- Export Prometheus/Grafana
- Historique performance trends
"""

import logging
import asyncio
import json
import time
from typing import Dict, List, Optional, Any, AsyncIterator
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict

logger = logging.getLogger(__name__)

class MetricType(Enum):
    """Types de metriques monitores"""
    LATENCY = "latency"
    THROUGHPUT = "throughput"  
    ERROR_RATE = "error_rate"
    RESOURCE_USAGE = "resource_usage"
    QUEUE_SIZE = "queue_size"
    CUSTOM = "custom"

class AlertSeverity(Enum):
    """Niveaux de severite alertes"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

@dataclass
class MetricDataPoint:
    """Point de donnee metrique"""
    timestamp: datetime
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    source_component: str = ""

@dataclass
class PerformanceThreshold:
    """Seuil de performance avec alertes"""
    metric_name: str
    warning_threshold: float
    critical_threshold: float
    emergency_threshold: Optional[float] = None
    window_seconds: int = 60  # Fenetre de calcul
    min_samples: int = 5      # Minimum d'echantillons

@dataclass
class Alert:
    """Alerte performance"""
    id: str
    severity: AlertSeverity
    metric_name: str
    current_value: float
    threshold_value: float
    message: str
    timestamp: datetime
    source_component: str
    resolved: bool = False
    resolved_at: Optional[datetime] = None

@dataclass
class PerformanceDashboard:
    """Snapshot dashboard performance"""
    timestamp: datetime
    pipeline_metrics: Dict[str, Any]
    component_metrics: Dict[str, Any]
    active_alerts: List[Alert]
    throughput_per_minute: float
    avg_latency_ms: float
    error_rate_percent: float
    resource_usage: Dict[str, float]

class RealTimeMetricsCollector:
    """Collecteur metriques temps reel avec buffer circulaire"""
    
    def __init__(self, buffer_size: int = 1000):
        self.buffer_size = buffer_size
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=buffer_size))
        self.lock = asyncio.Lock()
    
    async def record_metric(
        self, 
        metric_name: str, 
        value: float, 
        labels: Dict[str, str] = None,
        source_component: str = ""
    ):
        """Enregistre une metrique"""
        
        async with self.lock:
            data_point = MetricDataPoint(
                timestamp=datetime.now(),
                value=value,
                labels=labels or {},
                source_component=source_component
            )
            
            self.metrics[metric_name].append(data_point)
    
    async def get_metric_history(
        self, 
        metric_name: str, 
        window_seconds: int = 300
    ) -> List[MetricDataPoint]:
        """Recupere historique metrique"""
        
        async with self.lock:
            if metric_name not in self.metrics:
                return []
            
            cutoff_time = datetime.now() - timedelta(seconds=window_seconds)
            
            return [
                point for point in self.metrics[metric_name]
                if point.timestamp >= cutoff_time
            ]
    
    async def get_metric_aggregate(
        self,
        metric_name: str,
        window_seconds: int = 60,
        aggregation: str = "avg"  # avg, min, max, sum, count
    ) -> Optional[float]:
        """Calcule agregation metrique"""
        
        history = await self.get_metric_history(metric_name, window_seconds)
        
        if not history:
            return None
        
        values = [point.value for point in history]
        
        if aggregation == "avg":
            return sum(values) / len(values)
        elif aggregation == "min":
            return min(values)
        elif aggregation == "max":
            return max(values)
        elif aggregation == "sum":
            return sum(values)
        elif aggregation == "count":
            return float(len(values))
        else:
            return None

class AlertManager:
    """Gestionnaire alertes avec historique"""
    
    def __init__(self):
        self.thresholds: Dict[str, PerformanceThreshold] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.alert_callbacks: List[callable] = []
    
    def configure_threshold(self, threshold: PerformanceThreshold):
        """Configure seuil alerte pour une metrique"""
        self.thresholds[threshold.metric_name] = threshold
        logger.info(f"Seuil configure pour {threshold.metric_name}: "
                   f"warn={threshold.warning_threshold}, "
                   f"crit={threshold.critical_threshold}")
    
    def add_alert_callback(self, callback: callable):
        """Ajoute callback pour notifications alertes"""
        self.alert_callbacks.append(callback)
    
    async def check_metric_thresholds(
        self,
        metrics_collector: RealTimeMetricsCollector
    ):
        """Verifie seuils et declenche alertes si necessaire"""
        
        for metric_name, threshold in self.thresholds.items():
            try:
                # Calcul valeur moyenne sur fenetre
                current_value = await metrics_collector.get_metric_aggregate(
                    metric_name,
                    threshold.window_seconds,
                    "avg"
                )
                
                if current_value is None:
                    continue
                
                # Verification nombre minimum echantillons
                history = await metrics_collector.get_metric_history(
                    metric_name, 
                    threshold.window_seconds
                )
                
                if len(history) < threshold.min_samples:
                    continue
                
                # Detection niveau alerte
                alert_severity = None
                
                if (threshold.emergency_threshold and 
                    current_value >= threshold.emergency_threshold):
                    alert_severity = AlertSeverity.EMERGENCY
                elif current_value >= threshold.critical_threshold:
                    alert_severity = AlertSeverity.CRITICAL
                elif current_value >= threshold.warning_threshold:
                    alert_severity = AlertSeverity.WARNING
                
                # Gestion alerte
                if alert_severity:
                    await self._trigger_alert(
                        metric_name, 
                        alert_severity, 
                        current_value, 
                        threshold
                    )
                else:
                    # Resolution alerte si existante
                    await self._resolve_alert(metric_name)
                    
            except Exception as e:
                logger.error(f"Erreur verification seuil {metric_name}: {str(e)}")
    
    async def _trigger_alert(
        self,
        metric_name: str,
        severity: AlertSeverity,
        current_value: float,
        threshold: PerformanceThreshold
    ):
        """Declenche alerte"""
        
        alert_id = f"{metric_name}_{severity.value}"
        
        # Eviter spam d'alertes identiques
        if alert_id in self.active_alerts:
            existing_alert = self.active_alerts[alert_id]
            if existing_alert.severity == severity:
                return  # Alerte deja active
        
        # Creation nouvelle alerte
        threshold_value = threshold.critical_threshold
        if severity == AlertSeverity.WARNING:
            threshold_value = threshold.warning_threshold
        elif severity == AlertSeverity.EMERGENCY and threshold.emergency_threshold:
            threshold_value = threshold.emergency_threshold
        
        alert = Alert(
            id=alert_id,
            severity=severity,
            metric_name=metric_name,
            current_value=current_value,
            threshold_value=threshold_value,
            message=f"{metric_name} = {current_value:.2f} (seuil {severity.value}: {threshold_value:.2f})",
            timestamp=datetime.now(),
            source_component=metric_name.split('_')[0] if '_' in metric_name else "unknown"
        )
        
        # Enregistrement alerte
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)
        
        # Callbacks notifications
        for callback in self.alert_callbacks:
            try:
                await callback(alert)
            except Exception as e:
                logger.error(f"Erreur callback alerte: {str(e)}")
        
        logger.warning(f"ALERTE {severity.value.upper()}: {alert.message}")
    
    async def _resolve_alert(self, metric_name: str):
        """Resout alertes pour une metrique"""
        
        resolved_alerts = []
        
        for alert_id, alert in list(self.active_alerts.items()):
            if alert.metric_name == metric_name and not alert.resolved:
                alert.resolved = True
                alert.resolved_at = datetime.now()
                resolved_alerts.append(alert)
                del self.active_alerts[alert_id]
        
        for alert in resolved_alerts:
            logger.info(f"Alerte resolue: {alert.message}")
    
    def get_active_alerts(self) -> List[Alert]:
        """Recupere alertes actives"""
        return list(self.active_alerts.values())
    
    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """Recupere historique alertes"""
        return self.alert_history[-limit:]

class PerformanceMonitor:
    """
    Monitor principal performance temps reel
    
    Collecte metriques, gere alertes, fournit dashboard
    """
    
    def __init__(
        self,
        metrics_buffer_size: int = 2000,
        monitoring_interval_seconds: float = 5.0
    ):
        self.metrics_collector = RealTimeMetricsCollector(metrics_buffer_size)
        self.alert_manager = AlertManager()
        self.monitoring_interval = monitoring_interval_seconds
        
        # Tache background monitoring
        self._monitoring_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Statistiques globales
        self.stats = {
            "monitoring_start_time": datetime.now(),
            "total_metrics_collected": 0,
            "total_alerts_triggered": 0,
            "uptime_seconds": 0
        }
        
        # Configuration seuils par defaut
        self._configure_default_thresholds()
        
        logger.info("PerformanceMonitor initialise")
    
    def _configure_default_thresholds(self):
        """Configure seuils par defaut selon criteres Phase 6"""
        
        # Latence pipeline (<2s critere Phase 5, alertes graduelles)
        self.alert_manager.configure_threshold(PerformanceThreshold(
            metric_name="pipeline_latency_ms",
            warning_threshold=1500,    # 1.5s warning
            critical_threshold=2000,   # 2s critical (critere Phase 5)
            emergency_threshold=5000,  # 5s emergency
            window_seconds=60,
            min_samples=3
        ))
        
        # Taux erreur pipeline
        self.alert_manager.configure_threshold(PerformanceThreshold(
            metric_name="pipeline_error_rate",
            warning_threshold=0.05,    # 5% warning
            critical_threshold=0.10,   # 10% critical
            emergency_threshold=0.25,  # 25% emergency
            window_seconds=300,        # 5 minutes
            min_samples=10
        ))
        
        # Throughput (conversations/minute)
        self.alert_manager.configure_threshold(PerformanceThreshold(
            metric_name="conversations_per_minute",
            warning_threshold=0,       # Pas de warning pour throughput bas
            critical_threshold=0,      # Seulement si 0 pendant longtemps
            window_seconds=180,        # 3 minutes
            min_samples=5
        ))
        
        # Usage memoire (si disponible)
        self.alert_manager.configure_threshold(PerformanceThreshold(
            metric_name="memory_usage_percent",
            warning_threshold=80,      # 80% warning
            critical_threshold=90,     # 90% critical
            emergency_threshold=95,    # 95% emergency
            window_seconds=60,
            min_samples=5
        ))
        
        # Queue size (conversations en attente)
        self.alert_manager.configure_threshold(PerformanceThreshold(
            metric_name="conversation_queue_size",
            warning_threshold=10,      # 10 conversations en queue
            critical_threshold=25,     # 25 conversations
            emergency_threshold=50,    # 50 conversations
            window_seconds=30,
            min_samples=3
        ))
    
    async def start_monitoring(self):
        """Demarre monitoring background"""
        
        if self._running:
            logger.warning("Monitoring deja demarre")
            return
        
        self._running = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info(f"Monitoring demarre (interval: {self.monitoring_interval}s)")
    
    async def stop_monitoring(self):
        """Arrete monitoring"""
        
        self._running = False
        
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Monitoring arrete")
    
    async def _monitoring_loop(self):
        """Boucle principale monitoring"""
        
        try:
            while self._running:
                start_time = time.time()
                
                # Verification seuils alertes
                await self.alert_manager.check_metric_thresholds(self.metrics_collector)
                
                # Mise a jour stats uptime
                self.stats["uptime_seconds"] = (
                    datetime.now() - self.stats["monitoring_start_time"]
                ).total_seconds()
                
                # Attente prochain cycle
                elapsed = time.time() - start_time
                sleep_time = max(0, self.monitoring_interval - elapsed)
                
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                    
        except asyncio.CancelledError:
            logger.info("Monitoring loop cancelled")
        except Exception as e:
            logger.error(f"Erreur monitoring loop: {str(e)}")
    
    async def record_pipeline_metrics(
        self,
        conversation_id: str,
        pipeline_result,  # ConversationResult
        start_time: datetime
    ):
        """Enregistre metriques d'une conversation complete"""
        
        try:
            # Latence totale
            latency_ms = pipeline_result.metrics.total_processing_time_ms
            await self.metrics_collector.record_metric(
                "pipeline_latency_ms",
                latency_ms,
                {"conversation_id": conversation_id},
                "pipeline"
            )
            
            # Success/Error rate
            success_rate = 1.0 if pipeline_result.success else 0.0
            error_rate = 0.0 if pipeline_result.success else 1.0
            
            await self.metrics_collector.record_metric(
                "pipeline_success_rate",
                success_rate,
                {"conversation_id": conversation_id},
                "pipeline"
            )
            
            await self.metrics_collector.record_metric(
                "pipeline_error_rate", 
                error_rate,
                {"conversation_id": conversation_id},
                "pipeline"
            )
            
            # Throughput (conversations par minute)
            await self.metrics_collector.record_metric(
                "conversations_per_minute",
                1.0,  # 1 conversation traitee
                {},
                "pipeline"
            )
            
            # Tokens consommes
            if hasattr(pipeline_result.metrics, 'tokens_used'):
                await self.metrics_collector.record_metric(
                    "tokens_consumed",
                    pipeline_result.metrics.tokens_used,
                    {"model": pipeline_result.metrics.model_used},
                    "llm"
                )
            
            # Metriques par stage
            if hasattr(pipeline_result.metrics, 'stage_timings'):
                for stage, timing_ms in pipeline_result.metrics.stage_timings.items():
                    await self.metrics_collector.record_metric(
                        f"stage_{stage}_latency_ms",
                        timing_ms,
                        {"conversation_id": conversation_id},
                        stage
                    )
            
            self.stats["total_metrics_collected"] += 1
            
        except Exception as e:
            logger.error(f"Erreur enregistrement metriques: {str(e)}")
    
    async def record_component_metric(
        self,
        component_name: str,
        metric_name: str,
        value: float,
        labels: Dict[str, str] = None
    ):
        """Enregistre metrique d'un composant specifique"""
        
        full_metric_name = f"{component_name}_{metric_name}"
        await self.metrics_collector.record_metric(
            full_metric_name,
            value,
            labels or {},
            component_name
        )
    
    async def get_performance_dashboard(self) -> PerformanceDashboard:
        """Genere dashboard performance actuel"""
        
        try:
            # Metriques pipeline principales
            pipeline_metrics = {}
            
            # Latence moyenne derniere minute
            avg_latency = await self.metrics_collector.get_metric_aggregate(
                "pipeline_latency_ms", 60, "avg"
            )
            pipeline_metrics["avg_latency_ms"] = avg_latency or 0
            
            # Throughput derniere minute  
            throughput_count = await self.metrics_collector.get_metric_aggregate(
                "conversations_per_minute", 60, "count"
            )
            pipeline_metrics["throughput_per_minute"] = throughput_count or 0
            
            # Taux erreur derniere minute
            error_rate = await self.metrics_collector.get_metric_aggregate(
                "pipeline_error_rate", 60, "avg"
            )
            pipeline_metrics["error_rate_percent"] = (error_rate or 0) * 100
            
            # Metriques composants
            component_metrics = {}
            
            # LLM metrics
            tokens_used = await self.metrics_collector.get_metric_aggregate(
                "tokens_consumed", 300, "sum"  # 5 minutes
            )
            component_metrics["llm_tokens_consumed_5min"] = tokens_used or 0
            
            # Alertes actives
            active_alerts = self.alert_manager.get_active_alerts()
            
            # Resource usage (mock pour l'instant)
            resource_usage = {
                "memory_usage_percent": 0,  # A implementer avec psutil
                "cpu_usage_percent": 0,
                "open_connections": 0
            }
            
            dashboard = PerformanceDashboard(
                timestamp=datetime.now(),
                pipeline_metrics=pipeline_metrics,
                component_metrics=component_metrics,
                active_alerts=active_alerts,
                throughput_per_minute=pipeline_metrics["throughput_per_minute"],
                avg_latency_ms=pipeline_metrics["avg_latency_ms"],
                error_rate_percent=pipeline_metrics["error_rate_percent"],
                resource_usage=resource_usage
            )
            
            return dashboard
            
        except Exception as e:
            logger.error(f"Erreur generation dashboard: {str(e)}")
            return PerformanceDashboard(
                timestamp=datetime.now(),
                pipeline_metrics={},
                component_metrics={},
                active_alerts=[],
                throughput_per_minute=0,
                avg_latency_ms=0,
                error_rate_percent=0,
                resource_usage={}
            )
    
    async def stream_dashboard_updates(self) -> AsyncIterator[str]:
        """Stream dashboard updates pour WebSocket"""
        
        try:
            while self._running:
                dashboard = await self.get_performance_dashboard()
                
                # Serialisation dashboard
                dashboard_data = {
                    "timestamp": dashboard.timestamp.isoformat(),
                    "pipeline_metrics": dashboard.pipeline_metrics,
                    "component_metrics": dashboard.component_metrics,
                    "active_alerts": [
                        {
                            "id": alert.id,
                            "severity": alert.severity.value,
                            "metric_name": alert.metric_name,
                            "current_value": alert.current_value,
                            "threshold_value": alert.threshold_value,
                            "message": alert.message,
                            "timestamp": alert.timestamp.isoformat(),
                            "source_component": alert.source_component
                        }
                        for alert in dashboard.active_alerts
                    ],
                    "throughput_per_minute": dashboard.throughput_per_minute,
                    "avg_latency_ms": dashboard.avg_latency_ms,
                    "error_rate_percent": dashboard.error_rate_percent,
                    "resource_usage": dashboard.resource_usage
                }
                
                yield f"data: {json.dumps(dashboard_data)}\n\n"
                
                # Attente avant prochain update
                await asyncio.sleep(self.monitoring_interval)
                
        except asyncio.CancelledError:
            logger.info("Dashboard streaming stopped")
        except Exception as e:
            logger.error(f"Erreur dashboard streaming: {str(e)}")
            yield f"data: {{\"error\": \"Streaming error: {str(e)}\"}}\n\n"
    
    def get_monitoring_stats(self) -> Dict[str, Any]:
        """Recupere statistiques monitoring"""
        
        return {
            "monitoring_active": self._running,
            "uptime_seconds": self.stats["uptime_seconds"],
            "total_metrics_collected": self.stats["total_metrics_collected"],
            "total_alerts_triggered": len(self.alert_manager.alert_history),
            "active_alerts_count": len(self.alert_manager.active_alerts),
            "configured_thresholds": len(self.alert_manager.thresholds),
            "metrics_buffer_size": self.metrics_collector.buffer_size,
            "monitoring_interval_seconds": self.monitoring_interval
        }
    
    async def export_prometheus_metrics(self) -> str:
        """Export metriques format Prometheus"""
        
        try:
            metrics_lines = []
            
            # Header
            metrics_lines.append("# HELP conversation_service_metrics Performance metrics")
            metrics_lines.append("# TYPE conversation_service_metrics gauge")
            
            # Metriques principales
            dashboard = await self.get_performance_dashboard()
            
            timestamp_ms = int(dashboard.timestamp.timestamp() * 1000)
            
            # Latence moyenne
            metrics_lines.append(
                f"conversation_service_latency_ms {dashboard.avg_latency_ms} {timestamp_ms}"
            )
            
            # Throughput
            metrics_lines.append(
                f"conversation_service_throughput_per_minute {dashboard.throughput_per_minute} {timestamp_ms}"
            )
            
            # Taux erreur
            metrics_lines.append(
                f"conversation_service_error_rate {dashboard.error_rate_percent/100} {timestamp_ms}"
            )
            
            # Alertes actives
            metrics_lines.append(
                f"conversation_service_active_alerts {len(dashboard.active_alerts)} {timestamp_ms}"
            )
            
            return "\n".join(metrics_lines)
            
        except Exception as e:
            logger.error(f"Erreur export Prometheus: {str(e)}")
            return f"# Error exporting metrics: {str(e)}"

# Instance globale monitor
performance_monitor = PerformanceMonitor()

async def initialize_performance_monitoring():
    """Initialise monitoring performance global"""
    
    try:
        await performance_monitor.start_monitoring()
        logger.info("Performance monitoring initialized")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize performance monitoring: {str(e)}")
        return False

async def cleanup_performance_monitoring():
    """Cleanup monitoring"""
    
    try:
        await performance_monitor.stop_monitoring()
        logger.info("Performance monitoring cleaned up")
    except Exception as e:
        logger.error(f"Error cleaning up performance monitoring: {str(e)}")