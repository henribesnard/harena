"""
Health checks complets pour monitoring infrastructure
"""

import asyncio
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from conversation_service.config import settings
from .logging import get_logger

logger = get_logger(__name__)


@dataclass
class HealthCheckResult:
    """Résultat check santé composant"""
    component: str
    status: str  # healthy, degraded, unhealthy
    latency_ms: Optional[int]
    details: Dict[str, Any]
    timestamp: datetime


class HealthChecker:
    """
    Vérifications santé système avec tests proactifs
    """
    
    def __init__(self, redis_manager=None, intent_engine=None):
        self.redis_manager = redis_manager
        self.intent_engine = intent_engine
        self.last_checks: Dict[str, HealthCheckResult] = {}
    
    async def check_redis_health(self) -> HealthCheckResult:
        """Vérification santé Redis avec tests fonctionnels"""
        
        start_time = time.time()
        
        try:
            if not self.redis_manager or not self.redis_manager.redis_client:
                return HealthCheckResult(
                    component="redis",
                    status="unhealthy",
                    latency_ms=None,
                    details={"error": "Redis manager not initialized"},
                    timestamp=datetime.utcnow()
                )
            
            # Test 1: Ping basique
            await self.redis_manager.redis_client.ping()
            ping_latency = int((time.time() - start_time) * 1000)
            
            # Test 2: Set/Get fonctionnel
            test_key = "health_check_test"
            test_value = {"timestamp": time.time(), "test": True}
            
            set_success = await self.redis_manager.set(test_key, test_value, ttl=60)
            if not set_success:
                raise Exception("Redis SET failed")
            
            retrieved_value = await self.redis_manager.get(test_key)
            if not retrieved_value or retrieved_value.get("test") is not True:
                raise Exception("Redis GET failed or data corrupted")
            
            # Nettoyage
            await self.redis_manager.delete(test_key)
            
            total_latency = int((time.time() - start_time) * 1000)
            
            # Évaluation status basé latence
            if total_latency < 50:
                status = "healthy"
            elif total_latency < 200:
                status = "degraded"
            else:
                status = "unhealthy"
            
            return HealthCheckResult(
                component="redis",
                status=status,
                latency_ms=total_latency,
                details={
                    "ping_latency_ms": ping_latency,
                    "functional_test": "passed",
                    "circuit_breaker_open": self.redis_manager._is_circuit_open
                },
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            total_latency = int((time.time() - start_time) * 1000)
            
            return HealthCheckResult(
                component="redis",
                status="unhealthy",
                latency_ms=total_latency,
                details={
                    "error": str(e),
                    "test_failed": "functional_test"
                },
                timestamp=datetime.utcnow()
            )
    
    async def check_intent_engine_health(self) -> HealthCheckResult:
        """Vérification santé moteur intention"""
        
        start_time = time.time()
        
        try:
            if not self.intent_engine:
                return HealthCheckResult(
                    component="intent_engine",
                    status="unhealthy", 
                    latency_ms=None,
                    details={"error": "Intent engine not initialized"},
                    timestamp=datetime.utcnow()
                )
            
            # Test détection intention simple
            test_result = await self.intent_engine.detect_intent(
                "solde compte test", 
                user_id=999999  # User ID test
            )
            
            total_latency = int((time.time() - start_time) * 1000)
            
            if not test_result:
                return HealthCheckResult(
                    component="intent_engine",
                    status="unhealthy",
                    latency_ms=total_latency,
                    details={"error": "Intent detection returned null"},
                    timestamp=datetime.utcnow()
                )
            
            # Évaluation qualité résultat
            if test_result.confidence.score > 0.8 and total_latency < 100:
                status = "healthy"
            elif test_result.confidence.score > 0.6 and total_latency < 500:
                status = "degraded"
            else:
                status = "unhealthy"
            
            return HealthCheckResult(
                component="intent_engine",
                status=status,
                latency_ms=total_latency,
                details={
                    "test_intent": test_result.intent_type,
                    "test_confidence": test_result.confidence.score,
                    "detection_level": test_result.level.value,
                    "cache_hit": test_result.metadata.get("cache_hit", False)
                },
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            total_latency = int((time.time() - start_time) * 1000)
            
            return HealthCheckResult(
                component="intent_engine",
                status="unhealthy",
                latency_ms=total_latency,
                details={"error": str(e)},
                timestamp=datetime.utcnow()
            )
    
    async def check_external_dependencies(self) -> HealthCheckResult:
        """Vérification dépendances externes (DeepSeek API)"""
        
        start_time = time.time()
        
        try:
            # Test simple DeepSeek API si configuré
            if not settings.DEEPSEEK_API_KEY:
                return HealthCheckResult(
                    component="external_apis",
                    status="degraded",
                    latency_ms=0,
                    details={"warning": "DeepSeek API not configured"},
                    timestamp=datetime.utcnow()
                )
            
            # Test connexion uniquement (pas d'appel réel pour économiser coûts)
            import httpx
            
            async with httpx.AsyncClient(timeout=5) as client:
                # Test connectivité DNS/réseau uniquement
                response = await client.head(
                    "https://api.deepseek.com",
                    headers={"Authorization": f"Bearer {settings.DEEPSEEK_API_KEY}"}
                )
            
            total_latency = int((time.time() - start_time) * 1000)
            
            return HealthCheckResult(
                component="external_apis",
                status="healthy",
                latency_ms=total_latency,
                details={
                    "deepseek_connectivity": "ok",
                    "network_latency_ms": total_latency
                },
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            total_latency = int((time.time() - start_time) * 1000)
            
            return HealthCheckResult(
                component="external_apis",
                status="degraded",  # Pas critique car fallback disponible
                latency_ms=total_latency,
                details={
                    "error": str(e),
                    "impact": "L2_LLM_fallback_unavailable"
                },
                timestamp=datetime.utcnow()
            )
    
    async def run_comprehensive_health_check(self) -> Dict[str, Any]:
        """Exécution checks santé complets"""
        
        checks = await asyncio.gather(
            self.check_redis_health(),
            self.check_intent_engine_health(),
            self.check_external_dependencies(),
            return_exceptions=True
        )
        
        results = {}
        overall_status = "healthy"
        critical_issues = []
        
        for check in checks:
            if isinstance(check, Exception):
                results["unknown_component"] = {
                    "status": "unhealthy",
                    "error": str(check)
                }
                overall_status = "unhealthy"
                critical_issues.append(f"Health check exception: {check}")
                continue
            
            results[check.component] = {
                "status": check.status,
                "latency_ms": check.latency_ms,
                "details": check.details,
                "timestamp": check.timestamp.isoformat()
            }
            
            # Mise à jour cache résultats
            self.last_checks[check.component] = check
            
            # Évaluation impact sur status global
            if check.component in ["redis", "intent_engine"] and check.status == "unhealthy":
                overall_status = "unhealthy"
                critical_issues.append(f"{check.component} is unhealthy")
            elif check.status == "degraded" and overall_status == "healthy":
                overall_status = "degraded"
        
        return {
            "overall_status": overall_status,
            "components": results,
            "critical_issues": critical_issues,
            "timestamp": datetime.utcnow().isoformat(),
            "summary": {
                "healthy_components": len([r for r in results.values() if r["status"] == "healthy"]),
                "degraded_components": len([r for r in results.values() if r["status"] == "degraded"]),
                "unhealthy_components": len([r for r in results.values() if r["status"] == "unhealthy"])
            }
        }
    
    async def get_last_health_status(self) -> Dict[str, Any]:
        """Récupération dernier status santé (cache)"""
        
        if not self.last_checks:
            return {"status": "unknown", "message": "No health checks performed yet"}
        
        # Status le plus récent par composant
        results = {}
        for component, check in self.last_checks.items():
            age_seconds = (datetime.utcnow() - check.timestamp).total_seconds()
            
            results[component] = {
                "status": check.status,
                "age_seconds": age_seconds,
                "stale": age_seconds > 300  # Stale après 5 minutes
            }
        
        return results
