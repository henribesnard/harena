# -*- coding: utf-8 -*-
"""
Metrics Dashboard - Phase 6 Production
Architecture v2.0 - Dashboard et alertes

Responsabilite : Dashboard metriques et systeme alertes
- Interface web temps reel metriques
- Integration performance + health monitoring
- Alertes Slack/Email configurables
- Export formats multiples (JSON, Prometheus)
- API REST pour integration externe
"""

import logging
import asyncio
import json
from typing import Dict, List, Optional, Any, AsyncIterator
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum

from .performance_monitor import performance_monitor, PerformanceDashboard
from .health_monitor import health_monitor, HealthStatus

logger = logging.getLogger(__name__)

class AlertChannel(Enum):
    """Canaux alertes supportes"""
    LOG = "log"
    WEBHOOK = "webhook" 
    EMAIL = "email"
    SLACK = "slack"

class DashboardView(Enum):
    """Vues dashboard disponibles"""
    OVERVIEW = "overview"
    PERFORMANCE = "performance"  
    HEALTH = "health"
    ALERTS = "alerts"
    DETAILED = "detailed"

@dataclass
class AlertChannelConfig:
    """Configuration canal alerte"""
    channel: AlertChannel
    enabled: bool
    config: Dict[str, Any]

@dataclass
class MetricsDashboardConfig:
    """Configuration dashboard metriques"""
    refresh_interval_seconds: int = 5
    history_retention_hours: int = 24
    alert_channels: List[AlertChannelConfig] = None
    enable_prometheus_export: bool = True
    enable_grafana_integration: bool = False

@dataclass
class SystemOverview:
    """Vue d'ensemble systeme"""
    timestamp: datetime
    overall_health_status: str
    pipeline_performance: Dict[str, Any]
    active_alerts_count: int
    throughput_last_hour: float
    avg_latency_last_hour: float
    error_rate_last_hour: float
    healthy_components: int
    total_components: int
    uptime_seconds: float

class AlertNotificationService:
    """Service notifications alertes"""
    
    def __init__(self, config: MetricsDashboardConfig):
        self.config = config
        self.alert_channels: List[AlertChannelConfig] = config.alert_channels or []
        
        # Historique notifications pour eviter spam
        self.notification_history: Dict[str, datetime] = {}
        self.notification_cooldown_seconds = 300  # 5 minutes
        
    async def send_alert_notification(self, alert, dashboard_data: Dict[str, Any]):
        """Envoie notification alerte via canaux configures"""
        
        alert_key = f"{alert.metric_name}_{alert.severity.value}"
        
        # Verification cooldown
        if self._is_in_cooldown(alert_key):
            return
        
        # Preparation message
        alert_message = self._format_alert_message(alert, dashboard_data)
        
        # Envoi via chaque canal active
        for channel_config in self.alert_channels:
            if not channel_config.enabled:
                continue
                
            try:
                await self._send_via_channel(channel_config, alert_message, alert)
                self.notification_history[alert_key] = datetime.now()
                
            except Exception as e:
                logger.error(f"Erreur envoi alerte via {channel_config.channel.value}: {str(e)}")
    
    def _is_in_cooldown(self, alert_key: str) -> bool:
        """Verifie si alerte est en cooldown"""
        
        if alert_key not in self.notification_history:
            return False
            
        last_notification = self.notification_history[alert_key]
        return (datetime.now() - last_notification).total_seconds() < self.notification_cooldown_seconds
    
    def _format_alert_message(self, alert, dashboard_data: Dict[str, Any]) -> Dict[str, str]:
        """Formate message alerte pour envoi"""
        
        severity_emojis = {
            "warning": " ",
            "critical": "=4", 
            "emergency": "=¨"
        }
        
        emoji = severity_emojis.get(alert.severity.value, "=Ê")
        
        return {
            "title": f"{emoji} Alerte {alert.severity.value.upper()}",
            "message": alert.message,
            "details": f"Composant: {alert.source_component}\n"
                     f"Métrique: {alert.metric_name}\n"
                     f"Valeur: {alert.current_value:.2f}\n"
                     f"Seuil: {alert.threshold_value:.2f}\n"
                     f"Timestamp: {alert.timestamp.strftime('%H:%M:%S')}",
            "system_status": dashboard_data.get("overview", {}).get("overall_health_status", "unknown"),
            "url": f"/monitoring/dashboard?alert_id={alert.id}"
        }
    
    async def _send_via_channel(
        self, 
        channel_config: AlertChannelConfig, 
        message: Dict[str, str], 
        alert
    ):
        """Envoie message via canal specifique"""
        
        if channel_config.channel == AlertChannel.LOG:
            # Log simple
            logger.warning(f"ALERT: {message['title']} - {message['message']}")
            
        elif channel_config.channel == AlertChannel.WEBHOOK:
            # Webhook HTTP
            webhook_url = channel_config.config.get("url")
            if webhook_url:
                await self._send_webhook(webhook_url, message, alert)
                
        elif channel_config.channel == AlertChannel.SLACK:
            # Slack webhook
            slack_webhook = channel_config.config.get("webhook_url")
            if slack_webhook:
                await self._send_slack_notification(slack_webhook, message, alert)
                
        elif channel_config.channel == AlertChannel.EMAIL:
            # Email (placeholder - necessiterait SMTP config)
            logger.info(f"Email alert would be sent: {message['title']}")
    
    async def _send_webhook(self, webhook_url: str, message: Dict[str, str], alert):
        """Envoie webhook HTTP generique"""
        
        import aiohttp
        
        payload = {
            "alert_id": alert.id,
            "severity": alert.severity.value,
            "metric": alert.metric_name,
            "value": alert.current_value,
            "threshold": alert.threshold_value,
            "timestamp": alert.timestamp.isoformat(),
            "message": message
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    webhook_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        logger.info(f"Webhook alert sent successfully to {webhook_url}")
                    else:
                        logger.error(f"Webhook alert failed: {response.status}")
                        
        except Exception as e:
            logger.error(f"Webhook error: {str(e)}")
    
    async def _send_slack_notification(self, webhook_url: str, message: Dict[str, str], alert):
        """Envoie notification Slack"""
        
        import aiohttp
        
        # Format Slack
        slack_payload = {
            "text": message["title"],
            "attachments": [
                {
                    "color": "warning" if alert.severity.value == "warning" else "danger",
                    "fields": [
                        {"title": "Message", "value": message["message"], "short": False},
                        {"title": "Détails", "value": message["details"], "short": False},
                        {"title": "Status Système", "value": message["system_status"], "short": True}
                    ],
                    "footer": "Monitoring System",
                    "ts": int(alert.timestamp.timestamp())
                }
            ]
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    webhook_url,
                    json=slack_payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        logger.info("Slack alert sent successfully")
                    else:
                        logger.error(f"Slack alert failed: {response.status}")
                        
        except Exception as e:
            logger.error(f"Slack error: {str(e)}")

class MetricsDashboard:
    """
    Dashboard principal metriques et monitoring
    
    Centralise performance monitor + health monitor + alertes
    """
    
    def __init__(self, config: MetricsDashboardConfig = None):
        self.config = config or MetricsDashboardConfig()
        
        # Service notifications
        self.alert_service = AlertNotificationService(self.config)
        
        # Callbacks alertes
        performance_monitor.alert_manager.add_alert_callback(self._handle_performance_alert)
        
        # Cache dashboard pour performance
        self._dashboard_cache: Dict[str, Any] = {}
        self._cache_timestamp: Optional[datetime] = None
        self._cache_ttl_seconds = 2  # Cache 2s pour eviter spam
        
        logger.info("MetricsDashboard initialise")
    
    async def _handle_performance_alert(self, alert):
        """Gestionnaire alertes performance"""
        
        try:
            # Recuperer contexte dashboard pour alerte
            dashboard_data = await self.get_dashboard_data(DashboardView.OVERVIEW)
            
            # Envoyer notification
            await self.alert_service.send_alert_notification(alert, dashboard_data)
            
        except Exception as e:
            logger.error(f"Erreur gestion alerte: {str(e)}")
    
    async def get_dashboard_data(self, view: DashboardView = DashboardView.OVERVIEW) -> Dict[str, Any]:
        """Recupere donnees dashboard selon vue"""
        
        # Verification cache
        if self._is_cache_valid():
            cached_data = self._dashboard_cache.get(view.value)
            if cached_data:
                return cached_data
        
        # Generation donnees selon vue
        if view == DashboardView.OVERVIEW:
            data = await self._get_overview_data()
        elif view == DashboardView.PERFORMANCE:
            data = await self._get_performance_data()
        elif view == DashboardView.HEALTH:
            data = await self._get_health_data()
        elif view == DashboardView.ALERTS:
            data = await self._get_alerts_data()
        elif view == DashboardView.DETAILED:
            data = await self._get_detailed_data()
        else:
            data = {"error": f"Unknown view: {view.value}"}
        
        # Mise en cache
        self._dashboard_cache[view.value] = data
        self._cache_timestamp = datetime.now()
        
        return data
    
    def _is_cache_valid(self) -> bool:
        """Verifie validite cache"""
        
        if not self._cache_timestamp:
            return False
            
        age_seconds = (datetime.now() - self._cache_timestamp).total_seconds()
        return age_seconds < self._cache_ttl_seconds
    
    async def _get_overview_data(self) -> Dict[str, Any]:
        """Vue d'ensemble systeme"""
        
        # Performance dashboard
        perf_dashboard = await performance_monitor.get_performance_dashboard()
        
        # Health overview  
        health_overview = await health_monitor.get_system_health()
        
        # Overview combine
        overview = SystemOverview(
            timestamp=datetime.now(),
            overall_health_status=health_overview["overall_status"],
            pipeline_performance={
                "avg_latency_ms": perf_dashboard.avg_latency_ms,
                "throughput_per_minute": perf_dashboard.throughput_per_minute,
                "error_rate_percent": perf_dashboard.error_rate_percent
            },
            active_alerts_count=len(perf_dashboard.active_alerts),
            throughput_last_hour=await self._get_throughput_last_hour(),
            avg_latency_last_hour=await self._get_avg_latency_last_hour(),
            error_rate_last_hour=await self._get_error_rate_last_hour(),
            healthy_components=health_overview["summary"]["healthy"],
            total_components=health_overview["summary"]["total"],
            uptime_seconds=health_overview["monitoring_stats"]["uptime_seconds"]
        )
        
        return {
            "view": "overview",
            "timestamp": overview.timestamp.isoformat(),
            "overview": asdict(overview),
            "recent_alerts": perf_dashboard.active_alerts[:5],  # 5 alertes recentes
            "quick_stats": {
                "pipeline_healthy": overview.overall_health_status == "healthy",
                "latency_ok": overview.avg_latency_last_hour < 2000,  # <2s critere Phase 5
                "error_rate_low": overview.error_rate_last_hour < 5,   # <5%
                "throughput_positive": overview.throughput_last_hour > 0
            }
        }
    
    async def _get_performance_data(self) -> Dict[str, Any]:
        """Donnees performance detaillees"""
        
        perf_dashboard = await performance_monitor.get_performance_dashboard()
        perf_stats = performance_monitor.get_monitoring_stats()
        
        # Historique metriques
        latency_history = await performance_monitor.metrics_collector.get_metric_history(
            "pipeline_latency_ms", 3600  # 1 heure
        )
        
        throughput_history = await performance_monitor.metrics_collector.get_metric_history(
            "conversations_per_minute", 3600
        )
        
        return {
            "view": "performance",
            "timestamp": datetime.now().isoformat(),
            "current_metrics": {
                "latency_ms": perf_dashboard.avg_latency_ms,
                "throughput_per_minute": perf_dashboard.throughput_per_minute,
                "error_rate_percent": perf_dashboard.error_rate_percent
            },
            "history": {
                "latency": [
                    {"timestamp": p.timestamp.isoformat(), "value": p.value}
                    for p in latency_history[-60:]  # Derniere heure, 1 point/min
                ],
                "throughput": [
                    {"timestamp": p.timestamp.isoformat(), "value": p.value}
                    for p in throughput_history[-60:]
                ]
            },
            "pipeline_metrics": perf_dashboard.pipeline_metrics,
            "component_metrics": perf_dashboard.component_metrics,
            "monitoring_stats": perf_stats
        }
    
    async def _get_health_data(self) -> Dict[str, Any]:
        """Donnees health detaillees"""
        
        health_overview = await health_monitor.get_system_health()
        
        # Details par composant
        component_details = {}
        for component_name in health_overview["components"].keys():
            component_health = await health_monitor.get_component_health(component_name)
            if component_health:
                component_details[component_name] = component_health
        
        return {
            "view": "health",
            "timestamp": datetime.now().isoformat(),
            "overall_status": health_overview["overall_status"],
            "summary": health_overview["summary"],
            "components": component_details,
            "monitoring_stats": health_overview["monitoring_stats"]
        }
    
    async def _get_alerts_data(self) -> Dict[str, Any]:
        """Donnees alertes"""
        
        # Alertes actives performance
        active_alerts = performance_monitor.alert_manager.get_active_alerts()
        
        # Historique alertes
        alert_history = performance_monitor.alert_manager.get_alert_history(50)
        
        return {
            "view": "alerts",
            "timestamp": datetime.now().isoformat(),
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
                for alert in active_alerts
            ],
            "alert_history": [
                {
                    "id": alert.id,
                    "severity": alert.severity.value,
                    "metric_name": alert.metric_name,
                    "message": alert.message,
                    "timestamp": alert.timestamp.isoformat(),
                    "resolved": alert.resolved,
                    "resolved_at": alert.resolved_at.isoformat() if alert.resolved_at else None
                }
                for alert in alert_history
            ],
            "alert_stats": {
                "total_active": len(active_alerts),
                "by_severity": {
                    "warning": len([a for a in active_alerts if a.severity.value == "warning"]),
                    "critical": len([a for a in active_alerts if a.severity.value == "critical"]),
                    "emergency": len([a for a in active_alerts if a.severity.value == "emergency"])
                },
                "total_in_history": len(alert_history)
            }
        }
    
    async def _get_detailed_data(self) -> Dict[str, Any]:
        """Donnees completes (toutes vues combinees)"""
        
        overview_data = await self._get_overview_data()
        performance_data = await self._get_performance_data()
        health_data = await self._get_health_data()
        alerts_data = await self._get_alerts_data()
        
        return {
            "view": "detailed",
            "timestamp": datetime.now().isoformat(),
            "overview": overview_data["overview"],
            "performance": performance_data,
            "health": health_data,
            "alerts": alerts_data
        }
    
    # === HELPER METHODS ===
    
    async def _get_throughput_last_hour(self) -> float:
        """Calcule throughput derniere heure"""
        
        throughput_data = await performance_monitor.metrics_collector.get_metric_history(
            "conversations_per_minute", 3600  # 1 heure
        )
        
        if not throughput_data:
            return 0.0
            
        # Somme conversations derniere heure
        return sum(point.value for point in throughput_data)
    
    async def _get_avg_latency_last_hour(self) -> float:
        """Calcule latence moyenne derniere heure"""
        
        latency_data = await performance_monitor.metrics_collector.get_metric_history(
            "pipeline_latency_ms", 3600
        )
        
        if not latency_data:
            return 0.0
            
        return sum(point.value for point in latency_data) / len(latency_data)
    
    async def _get_error_rate_last_hour(self) -> float:
        """Calcule taux erreur derniere heure"""
        
        error_data = await performance_monitor.metrics_collector.get_metric_history(
            "pipeline_error_rate", 3600
        )
        
        if not error_data:
            return 0.0
            
        # Moyenne des taux erreur
        avg_error_rate = sum(point.value for point in error_data) / len(error_data)
        return avg_error_rate * 100  # Conversion en pourcentage
    
    # === STREAMING API ===
    
    async def stream_dashboard_updates(
        self, 
        view: DashboardView = DashboardView.OVERVIEW
    ) -> AsyncIterator[str]:
        """Stream updates dashboard pour WebSocket/SSE"""
        
        try:
            while True:
                dashboard_data = await self.get_dashboard_data(view)
                
                yield f"data: {json.dumps(dashboard_data)}\n\n"
                
                # Attente selon intervalle config
                await asyncio.sleep(self.config.refresh_interval_seconds)
                
        except asyncio.CancelledError:
            logger.info(f"Dashboard streaming stopped for view {view.value}")
        except Exception as e:
            logger.error(f"Dashboard streaming error: {str(e)}")
            yield f"data: {{\"error\": \"Streaming error: {str(e)}\"}}\n\n"
    
    # === EXPORT FORMATS ===
    
    async def export_prometheus_metrics(self) -> str:
        """Export metriques format Prometheus"""
        
        # Utiliser export performance monitor + health data
        perf_metrics = await performance_monitor.export_prometheus_metrics()
        
        # Ajouter metriques health
        health_data = await health_monitor.get_system_health()
        
        health_metrics_lines = [
            "# HELP conversation_service_health Health status metrics",
            "# TYPE conversation_service_health gauge"
        ]
        
        timestamp_ms = int(datetime.now().timestamp() * 1000)
        
        # Status global (1=healthy, 0.5=degraded, 0=unhealthy)
        status_value = {
            "healthy": 1.0,
            "degraded": 0.5,
            "unhealthy": 0.0
        }.get(health_data["overall_status"], 0.0)
        
        health_metrics_lines.append(
            f"conversation_service_health_status {status_value} {timestamp_ms}"
        )
        
        # Composants healthy
        healthy_ratio = health_data["summary"]["healthy"] / max(health_data["summary"]["total"], 1)
        health_metrics_lines.append(
            f"conversation_service_components_healthy_ratio {healthy_ratio} {timestamp_ms}"
        )
        
        health_metrics = "\n".join(health_metrics_lines)
        
        return f"{perf_metrics}\n{health_metrics}"
    
    async def export_json_summary(self) -> Dict[str, Any]:
        """Export resume JSON pour APIs externes"""
        
        overview_data = await self.get_dashboard_data(DashboardView.OVERVIEW)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "service": "conversation_service_v2",
            "version": "2.0",
            "status": overview_data["overview"]["overall_health_status"],
            "metrics": {
                "latency_ms": overview_data["overview"]["pipeline_performance"]["avg_latency_ms"],
                "throughput_per_minute": overview_data["overview"]["pipeline_performance"]["throughput_per_minute"],
                "error_rate_percent": overview_data["overview"]["pipeline_performance"]["error_rate_percent"],
                "active_alerts": overview_data["overview"]["active_alerts_count"],
                "uptime_seconds": overview_data["overview"]["uptime_seconds"]
            },
            "health": {
                "healthy_components": overview_data["overview"]["healthy_components"],
                "total_components": overview_data["overview"]["total_components"],
                "component_health_ratio": overview_data["overview"]["healthy_components"] / max(overview_data["overview"]["total_components"], 1)
            }
        }

# Instance globale dashboard
metrics_dashboard = MetricsDashboard()

async def initialize_metrics_dashboard(config: MetricsDashboardConfig = None):
    """Initialise dashboard metriques"""
    
    try:
        global metrics_dashboard
        if config:
            metrics_dashboard = MetricsDashboard(config)
        
        logger.info("Metrics dashboard initialized")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize metrics dashboard: {str(e)}")
        return False