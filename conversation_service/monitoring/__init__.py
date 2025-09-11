"""
Monitoring Package - Phase 6 Production
Architecture v2.0

Module monitoring complet pour production :
- PerformanceMonitor : Metriques temps reel + alertes
- HealthMonitor : Health checks detailles composants
- MetricsDashboard : Dashboard unifie + notifications
"""

from .performance_monitor import (
    PerformanceMonitor,
    RealTimeMetricsCollector,
    AlertManager,
    performance_monitor,
    initialize_performance_monitoring,
    cleanup_performance_monitoring
)

from .health_monitor import (
    HealthMonitor,
    HealthStatus,
    ComponentType,
    health_monitor,
    initialize_health_monitoring,
    cleanup_health_monitoring
)

from .metrics_dashboard import (
    MetricsDashboard,
    MetricsDashboardConfig,
    AlertChannelConfig,
    AlertChannel,
    DashboardView,
    metrics_dashboard,
    initialize_metrics_dashboard
)

__all__ = [
    # Performance monitoring
    "PerformanceMonitor",
    "RealTimeMetricsCollector", 
    "AlertManager",
    "performance_monitor",
    "initialize_performance_monitoring",
    "cleanup_performance_monitoring",
    
    # Health monitoring
    "HealthMonitor",
    "HealthStatus",
    "ComponentType", 
    "health_monitor",
    "initialize_health_monitoring",
    "cleanup_health_monitoring",
    
    # Dashboard & alertes
    "MetricsDashboard",
    "MetricsDashboardConfig",
    "AlertChannelConfig",
    "AlertChannel",
    "DashboardView",
    "metrics_dashboard",
    "initialize_metrics_dashboard"
]