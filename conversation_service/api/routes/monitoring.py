# -*- coding: utf-8 -*-
"""
Monitoring API Routes - Phase 6 Production
Architecture v2.0 - API monitoring et dashboard

Routes API pour monitoring, metriques et dashboard temps reel
"""

import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, Query, Response
from fastapi.responses import StreamingResponse, PlainTextResponse
from pydantic import BaseModel

from ...monitoring import (
    metrics_dashboard,
    performance_monitor,
    health_monitor,
    DashboardView
)

logger = logging.getLogger(__name__)

# === MODELS ===

class MonitoringStatusResponse(BaseModel):
    """Response monitoring status"""
    monitoring_active: bool
    performance_monitoring: bool
    health_monitoring: bool
    dashboard_ready: bool
    uptime_seconds: float

# === ROUTER ===

router = APIRouter(
    prefix="/api/v2/monitoring",
    tags=["monitoring"],
    responses={
        500: {"description": "Erreur monitoring interne"},
        503: {"description": "Monitoring indisponible"}
    }
)

# === ROUTES STATUS ===

@router.get("/status", response_model=MonitoringStatusResponse)
async def get_monitoring_status():
    """Status global du monitoring"""
    
    try:
        # Stats monitoring
        perf_stats = performance_monitor.get_monitoring_stats()
        
        return MonitoringStatusResponse(
            monitoring_active=performance_monitor._running and health_monitor._running,
            performance_monitoring=performance_monitor._running,
            health_monitoring=health_monitor._running,
            dashboard_ready=True,
            uptime_seconds=perf_stats["uptime_seconds"]
        )
        
    except Exception as e:
        logger.error(f"Erreur status monitoring: {str(e)}")
        return MonitoringStatusResponse(
            monitoring_active=False,
            performance_monitoring=False,
            health_monitoring=False,
            dashboard_ready=False,
            uptime_seconds=0
        )

# === ROUTES DASHBOARD ===

@router.get("/dashboard")
async def get_dashboard(
    view: str = Query("overview", description="Vue dashboard"),
    format: str = Query("json", description="Format export")
):
    """Dashboard principal monitoring"""
    
    try:
        # Parse vue
        dashboard_view = DashboardView.OVERVIEW
        if view in ["performance", "health", "alerts", "detailed"]:
            dashboard_view = DashboardView(view)
        
        # Recuperer donnees
        dashboard_data = await metrics_dashboard.get_dashboard_data(dashboard_view)
        
        # Format export
        if format == "prometheus":
            metrics_text = await metrics_dashboard.export_prometheus_metrics()
            return PlainTextResponse(content=metrics_text, media_type="text/plain")
        elif format == "summary":
            summary_data = await metrics_dashboard.export_json_summary()
            return summary_data
        else:
            return dashboard_data
            
    except Exception as e:
        logger.error(f"Erreur dashboard: {str(e)}")
        return {"error": f"Dashboard error: {str(e)}"}

@router.get("/dashboard/stream")
async def stream_dashboard(
    view: str = Query("overview", description="Vue dashboard"),
    interval: int = Query(5, description="Interval seconds", ge=1, le=60)
):
    """Stream dashboard temps reel via SSE"""
    
    try:
        # Parse vue
        dashboard_view = DashboardView.OVERVIEW
        if view in ["performance", "health", "alerts", "detailed"]:
            dashboard_view = DashboardView(view)
        
        # Override interval dashboard
        original_interval = metrics_dashboard.config.refresh_interval_seconds
        metrics_dashboard.config.refresh_interval_seconds = interval
        
        async def generate_dashboard_stream():
            try:
                async for chunk in metrics_dashboard.stream_dashboard_updates(dashboard_view):
                    yield chunk
            finally:
                # Restaurer interval original
                metrics_dashboard.config.refresh_interval_seconds = original_interval
        
        return StreamingResponse(
            generate_dashboard_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*"
            }
        )
        
    except Exception as e:
        logger.error(f"Erreur dashboard streaming: {str(e)}")
        
        async def error_stream():
            yield f"data: {{\"error\": \"Dashboard streaming error: {str(e)}\"}}\n\n"
        
        return StreamingResponse(error_stream(), media_type="text/event-stream")

# === ROUTES PERFORMANCE ===

@router.get("/performance")
async def get_performance_metrics():
    """Metriques performance detaillees"""
    
    try:
        dashboard = await performance_monitor.get_performance_dashboard()
        stats = performance_monitor.get_monitoring_stats()
        
        return {
            "timestamp": dashboard.timestamp.isoformat(),
            "metrics": {
                "throughput_per_minute": dashboard.throughput_per_minute,
                "avg_latency_ms": dashboard.avg_latency_ms,
                "error_rate_percent": dashboard.error_rate_percent
            },
            "pipeline_metrics": dashboard.pipeline_metrics,
            "component_metrics": dashboard.component_metrics,
            "data_visualizations": dashboard.data_visualizations,
            "monitoring_stats": stats
        }
        
    except Exception as e:
        logger.error(f"Erreur metriques performance: {str(e)}")
        return {"error": f"Performance metrics error: {str(e)}"}

@router.get("/performance/history")
async def get_performance_history(
    metric: str = Query("pipeline_latency_ms", description="Nom metrique"),
    window_seconds: int = Query(3600, description="Fenetre historique", ge=60, le=86400)
):
    """Historique metrique performance"""
    
    try:
        history = await performance_monitor.metrics_collector.get_metric_history(
            metric, window_seconds
        )
        
        return {
            "metric": metric,
            "window_seconds": window_seconds,
            "data_points": len(history),
            "history": [
                {
                    "timestamp": point.timestamp.isoformat(),
                    "value": point.value,
                    "labels": point.labels,
                    "source": point.source_component
                }
                for point in history
            ]
        }
        
    except Exception as e:
        logger.error(f"Erreur historique performance: {str(e)}")
        return {"error": f"Performance history error: {str(e)}"}

# === ROUTES HEALTH ===

@router.get("/health")
async def get_system_health():
    """Health check global systeme"""
    
    try:
        return await health_monitor.get_system_health()
        
    except Exception as e:
        logger.error(f"Erreur health check: {str(e)}")
        return {
            "overall_status": "unhealthy",
            "error": f"Health check error: {str(e)}",
            "timestamp": "unknown"
        }

@router.get("/health/{component_name}")
async def get_component_health(component_name: str):
    """Health check composant specifique"""
    
    try:
        component_health = await health_monitor.get_component_health(component_name)
        
        if not component_health:
            return {"error": f"Component {component_name} not found"}
        
        return component_health
        
    except Exception as e:
        logger.error(f"Erreur health component {component_name}: {str(e)}")
        return {"error": f"Component health error: {str(e)}"}

# === ROUTES ALERTES ===

@router.get("/alerts")
async def get_alerts(
    active_only: bool = Query(True, description="Alertes actives uniquement"),
    limit: int = Query(50, description="Limite alertes", ge=1, le=200)
):
    """Liste alertes systeme"""
    
    try:
        if active_only:
            alerts = performance_monitor.alert_manager.get_active_alerts()
        else:
            alerts = performance_monitor.alert_manager.get_alert_history(limit)
        
        return {
            "timestamp": "2024-01-01T00:00:00",  # Placeholder
            "active_only": active_only,
            "total_alerts": len(alerts),
            "alerts": [
                {
                    "id": alert.id,
                    "severity": alert.severity.value,
                    "metric_name": alert.metric_name,
                    "current_value": alert.current_value,
                    "threshold_value": alert.threshold_value,
                    "message": alert.message,
                    "timestamp": alert.timestamp.isoformat(),
                    "source_component": alert.source_component,
                    "resolved": getattr(alert, 'resolved', False),
                    "resolved_at": getattr(alert, 'resolved_at', None)
                }
                for alert in alerts
            ]
        }
        
    except Exception as e:
        logger.error(f"Erreur alertes: {str(e)}")
        return {"error": f"Alerts error: {str(e)}"}

@router.get("/alerts/stream")
async def stream_alerts():
    """Stream alertes temps reel"""
    
    async def generate_alert_stream():
        try:
            # Stream basic - dans une implementation complete,
            # ceci utiliserait un systeme pub/sub pour alertes temps reel
            import asyncio
            
            while True:
                alerts = performance_monitor.alert_manager.get_active_alerts()
                
                alert_data = {
                    "timestamp": "2024-01-01T00:00:00",  # Placeholder
                    "active_alerts_count": len(alerts),
                    "alerts": [
                        {
                            "id": alert.id,
                            "severity": alert.severity.value,
                            "message": alert.message,
                            "timestamp": alert.timestamp.isoformat()
                        }
                        for alert in alerts
                    ]
                }
                
                yield f"data: {alert_data}\n\n"
                await asyncio.sleep(10)  # Update toutes les 10s
                
        except Exception as e:
            yield f"data: {{\"error\": \"Alert streaming error: {str(e)}\"}}\n\n"
    
    return StreamingResponse(
        generate_alert_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )

# === ROUTES EXPORT ===

@router.get("/export/prometheus", response_class=PlainTextResponse)
async def export_prometheus():
    """Export metriques format Prometheus"""
    
    try:
        metrics = await metrics_dashboard.export_prometheus_metrics()
        return PlainTextResponse(content=metrics, media_type="text/plain")
        
    except Exception as e:
        logger.error(f"Erreur export Prometheus: {str(e)}")
        return PlainTextResponse(f"# Error: {str(e)}", media_type="text/plain")

@router.get("/export/json")
async def export_json():
    """Export resume JSON"""
    
    try:
        return await metrics_dashboard.export_json_summary()
        
    except Exception as e:
        logger.error(f"Erreur export JSON: {str(e)}")
        return {"error": f"JSON export error: {str(e)}"}

# === ROUTES CONFIGURATION ===

@router.get("/config")
async def get_monitoring_config():
    """Configuration monitoring actuelle"""
    
    try:
        return {
            "performance_monitor": {
                "monitoring_active": performance_monitor._running,
                "monitoring_interval": performance_monitor.monitoring_interval,
                "buffer_size": performance_monitor.metrics_collector.buffer_size,
                "configured_thresholds": len(performance_monitor.alert_manager.thresholds)
            },
            "health_monitor": {
                "monitoring_active": health_monitor._running,
                "check_interval": health_monitor.check_interval,
                "configured_checks": len(health_monitor.health_checks),
                "components_monitored": len(health_monitor.component_states)
            },
            "dashboard": {
                "refresh_interval": metrics_dashboard.config.refresh_interval_seconds,
                "cache_ttl": metrics_dashboard._cache_ttl_seconds
            }
        }
        
    except Exception as e:
        logger.error(f"Erreur config monitoring: {str(e)}")
        return {"error": f"Config error: {str(e)}"}

# === ROUTES UTILITAIRES ===

@router.post("/test-alert")
async def trigger_test_alert():
    """Declenche alerte test (dev/debug uniquement)"""
    
    try:
        # Enregistrer metrique artificielle pour declencher alerte
        await performance_monitor.metrics_collector.record_metric(
            "test_metric",
            9999,  # Valeur elevee pour declencher alerte
            {"source": "test"},
            "test_component"
        )
        
        return {"message": "Test alert triggered", "timestamp": "2024-01-01T00:00:00"}
        
    except Exception as e:
        logger.error(f"Erreur test alert: {str(e)}")
        return {"error": f"Test alert error: {str(e)}"}

@router.post("/clear-cache")
async def clear_monitoring_cache():
    """Vide cache monitoring"""
    
    try:
        # Clear dashboard cache
        metrics_dashboard._dashboard_cache.clear()
        metrics_dashboard._cache_timestamp = None
        
        return {"message": "Monitoring cache cleared", "timestamp": "2024-01-01T00:00:00"}
        
    except Exception as e:
        logger.error(f"Erreur clear cache: {str(e)}")
        return {"error": f"Clear cache error: {str(e)}"}