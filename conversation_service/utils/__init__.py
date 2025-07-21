"""
🛠️ Métriques MVP et monitoring performance

Utilitaires centralisés pour métriques, health checks et monitoring
performance Intent Detection Engine avec seuils par niveau.
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

# ==========================================
# CLASSES MÉTRIQUES
# ==========================================

@dataclass
class LevelMetrics:
    """Métriques performance par niveau L0/L1/L2"""
    
    level: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    
    # Latences
    total_latency_ms: float = 0.0
    min_latency_ms: Optional[float] = None
    max_latency_ms: Optional[float] = None
    latency_history: deque = field(default_factory=lambda: deque(maxlen=100))
    
    # Cache
    cache_hits: int = 0
    cache_misses: int = 0
    
    # Timestamps
    first_request_time: Optional[float] = None
    last_request_time: Optional[float] = None
    
    def record_request(self, latency_ms: float, success: bool, cache_hit: bool = False):
        """Enregistre nouvelle requête avec métriques"""
        current_time = time.time()
        
        # Compteurs
        self.total_requests += 1
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
        
        # Latence
        if success:
            self.total_latency_ms += latency_ms
            self.latency_history.append(latency_ms)
            
            if self.min_latency_ms is None or latency_ms < self.min_latency_ms:
                self.min_latency_ms = latency_ms
            
            if self.max_latency_ms is None or latency_ms > self.max_latency_ms:
                self.max_latency_ms = latency_ms
        
        # Cache
        if cache_hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
        
        # Timestamps
        if self.first_request_time is None:
            self.first_request_time = current_time
        self.last_request_time = current_time
    
    def get_success_rate(self) -> float:
        """Taux de succès"""
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests
    
    def get_average_latency(self) -> float:
        """Latence moyenne"""
        if self.successful_requests == 0:
            return 0.0
        return self.total_latency_ms / self.successful_requests
    
    def get_cache_hit_rate(self) -> float:
        """Taux hit cache"""
        total_cache_requests = self.cache_hits + self.cache_misses
        if total_cache_requests == 0:
            return 0.0
        return self.cache_hits / total_cache_requests
    
    def get_percentile_latency(self, percentile: float) -> float:
        """Latence percentile (50, 90, 95, 99)"""
        if not self.latency_history:
            return 0.0
        
        sorted_latencies = sorted(self.latency_history)
        index = int((percentile / 100) * len(sorted_latencies))
        index = min(index, len(sorted_latencies) - 1)
        
        return sorted_latencies[index]
    
    def meets_performance_target(self) -> bool:
        """Vérifie si niveau respecte cibles performance"""
        targets = {
            "L0_PATTERN": 10.0,      # <10ms
            "L1_LIGHTWEIGHT": 30.0,  # <30ms
            "L2_LLM": 500.0         # <500ms
        }
        
        target_latency = targets.get(self.level, 1000.0)
        avg_latency = self.get_average_latency()
        
        return avg_latency <= target_latency
    
    def get_performance_grade(self) -> str:
        """Grade performance A/B/C/D/F"""
        targets = {
            "L0_PATTERN": 10.0,
            "L1_LIGHTWEIGHT": 30.0,  
            "L2_LLM": 500.0
        }
        
        target = targets.get(self.level, 1000.0)
        avg_latency = self.get_average_latency()
        
        if avg_latency == 0:
            return "N/A"
        
        ratio = avg_latency / target
        
        if ratio <= 0.5:
            return "A"
        elif ratio <= 1.0:
            return "B"
        elif ratio <= 1.5:
            return "C"
        elif ratio <= 2.0:
            return "D"
        else:
            return "F"

class BasicMetrics:
    """
    📊 Gestionnaire métriques global Intent Detection Engine
    
    Centralise métriques performance par niveau avec calculs
    automatiques moyennes, percentiles et cibles.
    """
    
    def __init__(self):
        # Métriques par niveau
        self.level_metrics: Dict[str, LevelMetrics] = {
            "L0_PATTERN": LevelMetrics("L0_PATTERN"),
            "L1_LIGHTWEIGHT": LevelMetrics("L1_LIGHTWEIGHT"),
            "L2_LLM": LevelMetrics("L2_LLM"),
            "ERROR_TIMEOUT": LevelMetrics("ERROR_TIMEOUT"),
            "ERROR_FALLBACK": LevelMetrics("ERROR_FALLBACK"),
            "ERROR_SYSTEM": LevelMetrics("ERROR_SYSTEM")
        }
        
        # Métriques globales
        self.start_time = time.time()
        self.total_requests = 0
        self.system_errors = 0
        
        # Historique performance (pour tendances)
        self.performance_snapshots = deque(maxlen=100)
        
        logger.info("📊 BasicMetrics initialisé")
    
    def record_intent_performance(self, level: str, latency_ms: float, user_id: str, success: bool = True, cache_hit: bool = False):
        """Enregistre performance détection intention"""
        self.total_requests += 1
        
        # Enregistrement niveau spécifique
        if level in self.level_metrics:
            self.level_metrics[level].record_request(latency_ms, success, cache_hit)
        else:
            logger.warning(f"⚠️ Niveau métrique inconnu: {level}")
        
        # Erreur système si échec non géré
        if not success and level not in ["ERROR_TIMEOUT", "ERROR_FALLBACK"]:
            self.system_errors += 1
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Résumé performance global pour monitoring"""
        uptime_seconds = time.time() - self.start_time
        
        # Calculs globaux
        total_successful = sum(m.successful_requests for m in self.level_metrics.values())
        total_failed = sum(m.failed_requests for m in self.level_metrics.values())
        global_success_rate = total_successful / max(1, self.total_requests)
        
        # Distribution niveaux
        level_distribution = {}
        level_percentages = {}
        
        for level, metrics in self.level_metrics.items():
            level_distribution[level] = metrics.total_requests
            if self.total_requests > 0:
                level_percentages[level] = round((metrics.total_requests / self.total_requests) * 100, 1)
        
        # Performance par niveau
        level_performance = {}
        for level, metrics in self.level_metrics.items():
            if metrics.total_requests > 0:
                level_performance[level] = {
                    "requests": metrics.total_requests,
                    "success_rate": round(metrics.get_success_rate(), 3),
                    "avg_latency_ms": round(metrics.get_average_latency(), 2),
                    "min_latency_ms": metrics.min_latency_ms,
                    "max_latency_ms": metrics.max_latency_ms,
                    "p50_latency_ms": round(metrics.get_percentile_latency(50), 2),
                    "p95_latency_ms": round(metrics.get_percentile_latency(95), 2),
                    "cache_hit_rate": round(metrics.get_cache_hit_rate(), 3),
                    "meets_target": metrics.meets_performance_target(),
                    "grade": metrics.get_performance_grade()
                }
        
        # Calcul latence moyenne globale
        total_latency = sum(m.total_latency_ms for m in self.level_metrics.values())
        avg_latency_global = total_latency / max(1, total_successful)
        
        # Status global basé sur performance
        status = self._determine_global_status(level_performance, global_success_rate)
        
        return {
            # Métriques globales
            "status": status,
            "uptime_seconds": round(uptime_seconds, 1),
            "total_requests": self.total_requests,
            "successful_requests": total_successful,
            "failed_requests": total_failed,
            "system_errors": self.system_errors,
            "global_success_rate": round(global_success_rate, 3),
            "avg_latency_ms": round(avg_latency_global, 2),
            
            # Distribution niveaux
            "level_distribution": level_distribution,
            "level_percentages": level_percentages,
            
            # Performance détaillée
            "level_performance": level_performance,
            
            # Validation cibles globales
            "targets_validation": self._validate_global_targets(level_percentages, level_performance)
        }
    
    def _determine_global_status(self, level_performance: Dict[str, Any], success_rate: float) -> str:
        """Détermine status global basé sur métriques"""
        
        # Erreur critique si taux succès <80%
        if success_rate < 0.80:
            return "critical"
        
        # Dégradé si problèmes performance
        performance_issues = []
        
        # Vérification cibles par niveau
        for level, perf in level_performance.items():
            if level.startswith("ERROR_"):
                continue
                
            if not perf.get("meets_target", True):
                performance_issues.append(f"{level}_latency")
            
            if perf.get("success_rate", 1.0) < 0.90:
                performance_issues.append(f"{level}_reliability")
        
        if performance_issues:
            return "degraded"
        
        # Avertissement si distribution sous-optimale
        if self.total_requests > 100:  # Seulement avec volume suffisant
            l0_percentage = level_performance.get("L0_PATTERN", {}).get("requests", 0) / max(1, self.total_requests) * 100
            l2_percentage = level_performance.get("L2_LLM", {}).get("requests", 0) / max(1, self.total_requests) * 100
            
            # L0 devrait être ~85%, L2 ~3%
            if l0_percentage < 70 or l2_percentage > 10:
                return "warning"
        
        return "healthy"
    
    def _validate_global_targets(self, level_percentages: Dict[str, float], level_performance: Dict[str, Any]) -> Dict[str, Any]:
        """Validation cibles architecture L0→L1→L2"""
        
        validation = {
            "overall_target_met": True,
            "issues": [],
            "recommendations": []
        }
        
        # Cible distribution L0: 85%
        l0_percentage = level_percentages.get("L0_PATTERN", 0)
        if l0_percentage < 80:
            validation["overall_target_met"] = False
            validation["issues"].append(f"L0 usage trop bas: {l0_percentage}% (cible: 85%)")
            validation["recommendations"].append("Optimiser patterns L0 pour requêtes fréquentes")
        
        # Cible distribution L2: <5%
        l2_percentage = level_percentages.get("L2_LLM", 0)
        if l2_percentage > 8:
            validation["overall_target_met"] = False
            validation["issues"].append(f"L2 usage trop élevé: {l2_percentage}% (cible: <5%)")
            validation["recommendations"].append("Améliorer classification L1 ou patterns L0")
        
        # Cibles latence par niveau
        for level, targets in [("L0_PATTERN", 10), ("L1_LIGHTWEIGHT", 30), ("L2_LLM", 500)]:
            if level in level_performance:
                avg_latency = level_performance[level].get("avg_latency_ms", 0)
                if avg_latency > targets:
                    validation["overall_target_met"] = False
                    validation["issues"].append(f"{level} latence: {avg_latency}ms (cible: {targets}ms)")
        
        # Performance globale cible: 95% <100ms
        if self.total_requests > 50:
            fast_requests = 0
            for level, perf in level_performance.items():
                if level.startswith("ERROR_"):
                    continue
                avg_latency = perf.get("avg_latency_ms", 0)
                requests = perf.get("requests", 0)
                if avg_latency < 100:
                    fast_requests += requests
            
            fast_percentage = (fast_requests / max(1, self.total_requests)) * 100
            if fast_percentage < 95:
                validation["issues"].append(f"Requêtes <100ms: {fast_percentage:.1f}% (cible: 95%)")
        
        return validation
    
    def take_performance_snapshot(self):
        """Capture snapshot performance pour historique"""
        snapshot = {
            "timestamp": time.time(),
            "total_requests": self.total_requests,
            "global_success_rate": sum(m.successful_requests for m in self.level_metrics.values()) / max(1, self.total_requests),
            "level_breakdown": {
                level: {
                    "requests": metrics.total_requests,
                    "avg_latency": metrics.get_average_latency()
                }
                for level, metrics in self.level_metrics.items()
                if metrics.total_requests > 0
            }
        }
        
        self.performance_snapshots.append(snapshot)
    
    def get_performance_trends(self, minutes: int = 60) -> Dict[str, Any]:
        """Tendances performance sur période"""
        if len(self.performance_snapshots) < 2:
            return {"insufficient_data": True}
        
        cutoff_time = time.time() - (minutes * 60)
        recent_snapshots = [s for s in self.performance_snapshots if s["timestamp"] > cutoff_time]
        
        if len(recent_snapshots) < 2:
            return {"insufficient_recent_data": True}
        
        # Calcul tendances
        first_snapshot = recent_snapshots[0]
        last_snapshot = recent_snapshots[-1]
        
        request_growth = last_snapshot["total_requests"] - first_snapshot["total_requests"]
        success_rate_change = last_snapshot["global_success_rate"] - first_snapshot["global_success_rate"]
        
        return {
            "period_minutes": minutes,
            "snapshots_analyzed": len(recent_snapshots),
            "request_growth": request_growth,
            "requests_per_minute": request_growth / max(1, minutes),
            "success_rate_change": round(success_rate_change, 3),
            "trend_status": "improving" if success_rate_change > 0.01 else "stable" if abs(success_rate_change) < 0.01 else "degrading"
        }
    
    def reset_metrics(self):
        """Reset toutes les métriques"""
        for metrics in self.level_metrics.values():
            metrics.__init__(metrics.level)
        
        self.start_time = time.time()
        self.total_requests = 0
        self.system_errors = 0
        self.performance_snapshots.clear()
        
        logger.info("🔄 Métriques reset")

# ==========================================
# INSTANCE GLOBALE ET FONCTIONS HELPER
# ==========================================

# Instance globale BasicMetrics
_global_metrics = BasicMetrics()

async def record_intent_performance(level: str, latency_ms: float, user_id: str, success: bool = True, cache_hit: bool = False):
    """Helper pour enregistrement performance intention"""
    _global_metrics.record_intent_performance(level, latency_ms, user_id, success, cache_hit)

async def get_performance_summary() -> Dict[str, Any]:
    """Helper pour récupération résumé performance"""
    return _global_metrics.get_performance_summary()

def get_performance_trends(minutes: int = 60) -> Dict[str, Any]:
    """Helper pour tendances performance"""
    return _global_metrics.get_performance_trends(minutes)

def take_performance_snapshot():
    """Helper pour snapshot performance"""
    _global_metrics.take_performance_snapshot()

# ==========================================
# HEALTH CHECKS
# ==========================================

async def simple_health_check() -> Dict[str, Any]:
    """
    Health check basique service + dépendances
    
    Vérifie:
    - Status Intent Detection Engine
    - Connexion Redis si activé
    - Performance globale
    """
    health_status = {
        "healthy": True,
        "timestamp": time.time(),
        "service": "conversation_service",
        "version": "1.0.0"
    }
    
    try:
        # Métriques performance
        performance = await get_performance_summary()
        health_status["performance"] = {
            "status": performance["status"],
            "total_requests": performance["total_requests"],
            "success_rate": performance["global_success_rate"],
            "avg_latency_ms": performance["avg_latency_ms"]
        }
        
        # Status global basé sur performance
        if performance["status"] in ["critical", "degraded"]:
            health_status["healthy"] = False
            health_status["issues"] = [f"Performance status: {performance['status']}"]
        
        # Vérification Redis si activé
        redis_health = await _check_redis_health()
        health_status["redis"] = redis_health
        
        if not redis_health["available"] and redis_health["required"]:
            health_status["healthy"] = False
            health_status["issues"] = health_status.get("issues", []) + ["Redis indisponible"]
        
        # Vérification composants Intent Detection
        engine_health = await _check_intent_engine_health()
        health_status["intent_engine"] = engine_health
        
        if not engine_health["available"]:
            health_status["healthy"] = False
            health_status["issues"] = health_status.get("issues", []) + ["Intent Engine indisponible"]
    
    except Exception as e:
        logger.error(f"❌ Erreur health check: {e}")
        health_status.update({
            "healthy": False,
            "error": str(e)
        })
    
    return health_status

async def _check_redis_health() -> Dict[str, Any]:
    """Vérification santé Redis"""
    from config_service.config import settings
    
    redis_config = settings.get_cache_config()
    redis_required = redis_config["redis"]["enabled"]
    
    if not redis_required:
        return {
            "available": True,
            "required": False,
            "status": "disabled"
        }
    
    try:
        # Test connexion Redis via cache manager
        import aioredis
        
        redis_client = await aioredis.from_url(
            redis_config["redis"]["url"],
            password=redis_config["redis"]["password"],
            socket_connect_timeout=3,
            socket_timeout=2
        )
        
        # Test ping
        await redis_client.ping()
        await redis_client.close()
        
        return {
            "available": True,
            "required": True,
            "status": "connected",
            "url": redis_config["redis"]["url"]
        }
        
    except Exception as e:
        logger.warning(f"⚠️ Redis health check failed: {e}")
        return {
            "available": False,
            "required": True,
            "status": "error",
            "error": str(e)
        }

async def _check_intent_engine_health() -> Dict[str, Any]:
    """Vérification santé Intent Detection Engine"""
    try:
        # Tentative import engine pour vérifier disponibilité
        from conversation_service.intent_detection.engine import IntentDetectionEngine
        
        # Note: Ici on pourrait tester l'engine s'il était accessible globalement
        # Pour l'instant, on vérifie juste l'importabilité
        
        return {
            "available": True,
            "status": "importable",
            "components": {
                "engine": True,
                "cache_manager": True,
                "pattern_matcher": True
            }
        }
        
    except ImportError as e:
        logger.error(f"❌ Intent Engine import failed: {e}")
        return {
            "available": False,
            "status": "import_error",
            "error": str(e)
        }
    except Exception as e:
        logger.error(f"❌ Intent Engine health check failed: {e}")
        return {
            "available": False,
            "status": "error", 
            "error": str(e)
        }

# ==========================================
# MONITORING ET ALERTES
# ==========================================

class PerformanceMonitor:
    """
    📈 Monitoring avancé performance avec alertes
    
    Surveille métriques en temps réel et déclenche alertes
    selon seuils configurés.
    """
    
    def __init__(self):
        self.alert_thresholds = {
            "success_rate_min": 0.90,
            "avg_latency_max_ms": 200,
            "l0_usage_min_percent": 70,
            "l2_usage_max_percent": 10,
            "error_rate_max": 0.05
        }
        
        self.alerts_history = deque(maxlen=100)
        self.last_check_time = time.time()
    
    async def check_performance_alerts(self) -> List[Dict[str, Any]]:
        """Vérification alertes performance basées sur seuils"""
        alerts = []
        
        try:
            performance = await get_performance_summary()
            
            # Alerte taux de succès
            success_rate = performance.get("global_success_rate", 1.0)
            if success_rate < self.alert_thresholds["success_rate_min"]:
                alerts.append({
                    "type": "success_rate_low",
                    "severity": "critical",
                    "message": f"Taux de succès bas: {success_rate:.1%}",
                    "current_value": success_rate,
                    "threshold": self.alert_thresholds["success_rate_min"]
                })
            
            # Alerte latence moyenne
            avg_latency = performance.get("avg_latency_ms", 0)
            if avg_latency > self.alert_thresholds["avg_latency_max_ms"]:
                alerts.append({
                    "type": "latency_high",
                    "severity": "warning",
                    "message": f"Latence moyenne élevée: {avg_latency:.1f}ms",
                    "current_value": avg_latency,
                    "threshold": self.alert_thresholds["avg_latency_max_ms"]
                })
            
            # Alerte distribution L0
            l0_percentage = performance.get("level_percentages", {}).get("L0_PATTERN", 0)
            if l0_percentage < self.alert_thresholds["l0_usage_min_percent"]:
                alerts.append({
                    "type": "l0_usage_low",
                    "severity": "warning",
                    "message": f"Usage L0 bas: {l0_percentage:.1f}%",
                    "current_value": l0_percentage,
                    "threshold": self.alert_thresholds["l0_usage_min_percent"]
                })
            
            # Alerte distribution L2
            l2_percentage = performance.get("level_percentages", {}).get("L2_LLM", 0)
            if l2_percentage > self.alert_thresholds["l2_usage_max_percent"]:
                alerts.append({
                    "type": "l2_usage_high",
                    "severity": "warning",
                    "message": f"Usage L2 élevé: {l2_percentage:.1f}%",
                    "current_value": l2_percentage,
                    "threshold": self.alert_thresholds["l2_usage_max_percent"]
                })
            
            # Stockage historique alertes
            for alert in alerts:
                alert["timestamp"] = time.time()
                self.alerts_history.append(alert)
            
            self.last_check_time = time.time()
            
        except Exception as e:
            logger.error(f"❌ Erreur vérification alertes: {e}")
        
        return alerts
    
    def get_alerts_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Résumé alertes sur période"""
        cutoff_time = time.time() - (hours * 3600)
        recent_alerts = [a for a in self.alerts_history if a["timestamp"] > cutoff_time]
        
        # Groupement par type
        alerts_by_type = defaultdict(int)
        alerts_by_severity = defaultdict(int)
        
        for alert in recent_alerts:
            alerts_by_type[alert["type"]] += 1
            alerts_by_severity[alert["severity"]] += 1
        
        return {
            "period_hours": hours,
            "total_alerts": len(recent_alerts),
            "by_type": dict(alerts_by_type),
            "by_severity": dict(alerts_by_severity),
            "most_recent": recent_alerts[-5:] if recent_alerts else []
        }

# Instance globale monitoring
_performance_monitor = PerformanceMonitor()

async def check_performance_alerts() -> List[Dict[str, Any]]:
    """Helper vérification alertes performance"""
    return await _performance_monitor.check_performance_alerts()

def get_alerts_summary(hours: int = 24) -> Dict[str, Any]:
    """Helper résumé alertes"""
    return _performance_monitor.get_alerts_summary(hours)

# ==========================================
# BACKGROUND TASKS
# ==========================================

async def start_performance_monitoring():
    """Démarre monitoring performance en arrière-plan"""
    logger.info("📈 Démarrage monitoring performance...")
    
    async def monitoring_loop():
        while True:
            try:
                # Snapshot toutes les 5 minutes
                take_performance_snapshot()
                
                # Vérification alertes toutes les 2 minutes
                alerts = await check_performance_alerts()
                if alerts:
                    logger.warning(f"⚠️ {len(alerts)} alertes performance détectées")
                    for alert in alerts:
                        logger.warning(f"  - {alert['type']}: {alert['message']}")
                
                await asyncio.sleep(120)  # 2 minutes
                
            except Exception as e:
                logger.error(f"❌ Erreur monitoring loop: {e}")
                await asyncio.sleep(60)  # Retry dans 1 minute
    
    # Lancement task background
    asyncio.create_task(monitoring_loop())

# ==========================================
# EXPORT FONCTIONS PRINCIPALES
# ==========================================

__all__ = [
    "record_intent_performance",
    "get_performance_summary", 
    "get_performance_trends",
    "simple_health_check",
    "check_performance_alerts",
    "get_alerts_summary",
    "start_performance_monitoring",
    "BasicMetrics",
    "LevelMetrics",
    "PerformanceMonitor"
]