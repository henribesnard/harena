# -*- coding: utf-8 -*-
"""
Health Monitor - Phase 6 Production
Architecture v2.0 - Health checks detailles

Responsabilite : Monitoring sante composants et dependances
- Health checks detailles par composant
- Detection pannes et degradations
- Tests connectivity externes (search_service, LLM APIs)
- Recuperation automatique quand possible
- Dashboard sante temps reel
"""

import logging
import asyncio
import aiohttp
import json
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    """Status de sante composants"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

class ComponentType(Enum):
    """Types de composants monitores"""
    CORE_SERVICE = "core_service"
    EXTERNAL_API = "external_api"
    DATABASE = "database"
    CACHE = "cache"
    QUEUE = "queue"
    LLM_PROVIDER = "llm_provider"

@dataclass
class HealthCheck:
    """Configuration health check pour un composant"""
    name: str
    component_type: ComponentType
    check_function: callable
    timeout_seconds: int = 10
    interval_seconds: int = 30
    failure_threshold: int = 3  # Echecs avant unhealthy
    recovery_threshold: int = 2  # Succes avant healthy
    enabled: bool = True

@dataclass
class HealthResult:
    """Resultat health check"""
    component_name: str
    status: HealthStatus
    timestamp: datetime
    response_time_ms: int
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

@dataclass
class ComponentHealth:
    """Etat sante composant"""
    name: str
    status: HealthStatus
    last_check: datetime
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    total_checks: int = 0
    total_failures: int = 0
    avg_response_time_ms: float = 0
    last_error: Optional[str] = None
    history: List[HealthResult] = field(default_factory=list)

class ExternalServiceHealthChecker:
    """Health checks services externes"""
    
    @staticmethod
    async def check_search_service(base_url: str, timeout: int = 10) -> HealthResult:
        """Check search_service health"""
        
        start_time = time.time()
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
                # Test endpoint health
                health_url = f"{base_url.rstrip('/')}/health"
                
                async with session.get(health_url) as response:
                    response_time_ms = int((time.time() - start_time) * 1000)
                    
                    if response.status == 200:
                        health_data = await response.json()
                        
                        return HealthResult(
                            component_name="search_service",
                            status=HealthStatus.HEALTHY,
                            timestamp=datetime.now(),
                            response_time_ms=response_time_ms,
                            message="Search service responding normally",
                            details={
                                "endpoint": health_url,
                                "service_status": health_data.get("status", "unknown"),
                                "service_info": health_data
                            }
                        )
                    else:
                        return HealthResult(
                            component_name="search_service",
                            status=HealthStatus.DEGRADED,
                            timestamp=datetime.now(),
                            response_time_ms=response_time_ms,
                            message=f"Search service returned HTTP {response.status}",
                            error=f"HTTP {response.status}"
                        )
        
        except asyncio.TimeoutError:
            return HealthResult(
                component_name="search_service",
                status=HealthStatus.UNHEALTHY,
                timestamp=datetime.now(),
                response_time_ms=timeout * 1000,
                message="Search service timeout",
                error="Connection timeout"
            )
        
        except Exception as e:
            return HealthResult(
                component_name="search_service",
                status=HealthStatus.UNHEALTHY,
                timestamp=datetime.now(),
                response_time_ms=int((time.time() - start_time) * 1000),
                message="Search service connection failed",
                error=str(e)
            )
    
    @staticmethod
    async def check_llm_provider(
        provider_name: str, 
        base_url: str, 
        api_key: str, 
        timeout: int = 15
    ) -> HealthResult:
        """Check LLM provider health"""
        
        start_time = time.time()
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
                headers = {
                    "Content-Type": "application/json"
                }
                
                # Configuration par provider
                if provider_name.lower() == "deepseek":
                    headers["Authorization"] = f"Bearer {api_key}"
                    test_url = f"{base_url}/models"
                elif provider_name.lower() == "openai":
                    headers["Authorization"] = f"Bearer {api_key}"
                    test_url = f"{base_url}/models"
                elif provider_name.lower() == "local":
                    # Ollama endpoint
                    test_url = f"{base_url}/api/tags"
                else:
                    # Generic test
                    test_url = f"{base_url}/health"
                
                async with session.get(test_url, headers=headers) as response:
                    response_time_ms = int((time.time() - start_time) * 1000)
                    
                    if response.status == 200:
                        try:
                            data = await response.json()
                            model_count = 0
                            
                            # Parse response selon provider
                            if provider_name.lower() in ["deepseek", "openai"]:
                                model_count = len(data.get("data", []))
                            elif provider_name.lower() == "local":
                                model_count = len(data.get("models", []))
                            
                            return HealthResult(
                                component_name=f"llm_{provider_name}",
                                status=HealthStatus.HEALTHY,
                                timestamp=datetime.now(),
                                response_time_ms=response_time_ms,
                                message=f"{provider_name} API responding normally",
                                details={
                                    "provider": provider_name,
                                    "endpoint": test_url,
                                    "models_available": model_count,
                                    "response_data": data if len(str(data)) < 500 else "truncated"
                                }
                            )
                        except json.JSONDecodeError:
                            return HealthResult(
                                component_name=f"llm_{provider_name}",
                                status=HealthStatus.DEGRADED,
                                timestamp=datetime.now(),
                                response_time_ms=response_time_ms,
                                message=f"{provider_name} API returned invalid JSON",
                                error="Invalid JSON response"
                            )
                    else:
                        return HealthResult(
                            component_name=f"llm_{provider_name}",
                            status=HealthStatus.DEGRADED,
                            timestamp=datetime.now(),
                            response_time_ms=response_time_ms,
                            message=f"{provider_name} API returned HTTP {response.status}",
                            error=f"HTTP {response.status}"
                        )
        
        except asyncio.TimeoutError:
            return HealthResult(
                component_name=f"llm_{provider_name}",
                status=HealthStatus.UNHEALTHY,
                timestamp=datetime.now(),
                response_time_ms=timeout * 1000,
                message=f"{provider_name} API timeout",
                error="Connection timeout"
            )
        
        except Exception as e:
            return HealthResult(
                component_name=f"llm_{provider_name}",
                status=HealthStatus.UNHEALTHY,
                timestamp=datetime.now(),
                response_time_ms=int((time.time() - start_time) * 1000),
                message=f"{provider_name} API connection failed",
                error=str(e)
            )

class InternalComponentHealthChecker:
    """Health checks composants internes"""
    
    @staticmethod
    async def check_orchestrator(orchestrator) -> HealthResult:
        """Check conversation orchestrator health"""
        
        start_time = time.time()
        
        try:
            # Test health check orchostrateur
            health_data = await orchestrator.health_check()
            response_time_ms = int((time.time() - start_time) * 1000)
            
            status_map = {
                "healthy": HealthStatus.HEALTHY,
                "degraded": HealthStatus.DEGRADED,
                "unhealthy": HealthStatus.UNHEALTHY
            }
            
            status = status_map.get(health_data.get("status"), HealthStatus.UNKNOWN)
            
            return HealthResult(
                component_name="conversation_orchestrator",
                status=status,
                timestamp=datetime.now(),
                response_time_ms=response_time_ms,
                message=f"Orchestrator status: {health_data.get('status')}",
                details=health_data
            )
            
        except Exception as e:
            return HealthResult(
                component_name="conversation_orchestrator",
                status=HealthStatus.UNHEALTHY,
                timestamp=datetime.now(),
                response_time_ms=int((time.time() - start_time) * 1000),
                message="Orchestrator health check failed",
                error=str(e)
            )
    
    @staticmethod
    async def check_template_engine(template_engine) -> HealthResult:
        """Check template engine health"""
        
        start_time = time.time()
        
        try:
            # Test basique template engine
            stats = template_engine.get_cache_stats()
            response_time_ms = int((time.time() - start_time) * 1000)
            
            # Determine status selon stats
            status = HealthStatus.HEALTHY
            message = "Template engine operational"
            
            if not template_engine.initialized:
                status = HealthStatus.UNHEALTHY
                message = "Template engine not initialized"
            elif stats.get("cache_size", 0) == 0:
                status = HealthStatus.DEGRADED
                message = "Template engine has no cached templates"
            
            return HealthResult(
                component_name="template_engine",
                status=status,
                timestamp=datetime.now(),
                response_time_ms=response_time_ms,
                message=message,
                details=stats
            )
            
        except Exception as e:
            return HealthResult(
                component_name="template_engine",
                status=HealthStatus.UNHEALTHY,
                timestamp=datetime.now(),
                response_time_ms=int((time.time() - start_time) * 1000),
                message="Template engine check failed",
                error=str(e)
            )
    
    @staticmethod
    async def check_context_manager(context_manager) -> HealthResult:
        """Check context manager health"""
        
        start_time = time.time()
        
        try:
            # Test basique context manager
            stats = context_manager.get_stats()
            response_time_ms = int((time.time() - start_time) * 1000)
            
            status = HealthStatus.HEALTHY
            message = "Context manager operational"
            
            # Verification storage backend
            if not context_manager.storage_initialized:
                status = HealthStatus.DEGRADED
                message = "Context manager storage not fully initialized"
            
            return HealthResult(
                component_name="context_manager",
                status=status,
                timestamp=datetime.now(),
                response_time_ms=response_time_ms,
                message=message,
                details=stats
            )
            
        except Exception as e:
            return HealthResult(
                component_name="context_manager",
                status=HealthStatus.UNHEALTHY,
                timestamp=datetime.now(),
                response_time_ms=int((time.time() - start_time) * 1000),
                message="Context manager check failed", 
                error=str(e)
            )

class HealthMonitor:
    """
    Monitor principal sante systeme
    
    Orchestre health checks composants et services externes
    """
    
    def __init__(
        self,
        check_interval_seconds: int = 30,
        history_retention_hours: int = 24
    ):
        self.check_interval = check_interval_seconds
        self.history_retention = timedelta(hours=history_retention_hours)
        
        # Health checks configures
        self.health_checks: Dict[str, HealthCheck] = {}
        
        # Etat composants
        self.component_states: Dict[str, ComponentHealth] = {}
        
        # Tache background
        self._monitoring_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Dependencies injection (sera configure)
        self.dependencies: Dict[str, Any] = {}
        
        # Statistiques
        self.stats = {
            "monitoring_start_time": datetime.now(),
            "total_checks_performed": 0,
            "total_failures_detected": 0
        }
        
        logger.info("HealthMonitor initialise")
    
    def configure_dependencies(self, dependencies: Dict[str, Any]):
        """Configure dependencies pour health checks"""
        self.dependencies = dependencies
        
        # Auto-configuration health checks selon dependencies disponibles
        self._auto_configure_health_checks()
    
    def _auto_configure_health_checks(self):
        """Auto-configure health checks selon dependencies"""
        
        # Orchestrateur (toujours present)
        if "orchestrator" in self.dependencies:
            self.add_health_check(HealthCheck(
                name="conversation_orchestrator",
                component_type=ComponentType.CORE_SERVICE,
                check_function=self._check_orchestrator,
                interval_seconds=30,
                timeout_seconds=10
            ))
        
        # Template engine
        if "template_engine" in self.dependencies:
            self.add_health_check(HealthCheck(
                name="template_engine",
                component_type=ComponentType.CORE_SERVICE,
                check_function=self._check_template_engine,
                interval_seconds=60,
                timeout_seconds=5
            ))
        
        # Context manager
        if "context_manager" in self.dependencies:
            self.add_health_check(HealthCheck(
                name="context_manager",
                component_type=ComponentType.CACHE,
                check_function=self._check_context_manager,
                interval_seconds=45,
                timeout_seconds=5
            ))
        
        # Search service
        if "search_service_url" in self.dependencies:
            self.add_health_check(HealthCheck(
                name="search_service",
                component_type=ComponentType.EXTERNAL_API,
                check_function=self._check_search_service,
                interval_seconds=30,
                timeout_seconds=10,
                failure_threshold=2
            ))
        
        # LLM providers
        llm_configs = self.dependencies.get("llm_configs", {})
        for provider_name, config in llm_configs.items():
            if config.get("enabled", False):
                self.add_health_check(HealthCheck(
                    name=f"llm_{provider_name}",
                    component_type=ComponentType.LLM_PROVIDER,
                    check_function=lambda pn=provider_name, cfg=config: self._check_llm_provider(pn, cfg),
                    interval_seconds=60,
                    timeout_seconds=15,
                    failure_threshold=2
                ))
        
        logger.info(f"Auto-configured {len(self.health_checks)} health checks")
    
    def add_health_check(self, health_check: HealthCheck):
        """Ajoute health check"""
        self.health_checks[health_check.name] = health_check
        
        # Initialiser etat composant
        if health_check.name not in self.component_states:
            self.component_states[health_check.name] = ComponentHealth(
                name=health_check.name,
                status=HealthStatus.UNKNOWN,
                last_check=datetime.now()
            )
    
    async def start_monitoring(self):
        """Demarre monitoring health"""
        
        if self._running:
            logger.warning("Health monitoring deja demarre")
            return
        
        self._running = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info(f"Health monitoring demarre ({len(self.health_checks)} checks)")
    
    async def stop_monitoring(self):
        """Arrete monitoring health"""
        
        self._running = False
        
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Health monitoring arrete")
    
    async def _monitoring_loop(self):
        """Boucle principale monitoring health"""
        
        try:
            while self._running:
                # Execute health checks
                check_tasks = []
                
                for check_name, health_check in self.health_checks.items():
                    if not health_check.enabled:
                        continue
                    
                    component_state = self.component_states[check_name]
                    
                    # Verifier si il faut executer ce check
                    if self._should_run_check(component_state, health_check):
                        task = asyncio.create_task(
                            self._execute_health_check(check_name, health_check)
                        )
                        check_tasks.append(task)
                
                # Attendre tous les checks avec timeout
                if check_tasks:
                    await asyncio.gather(*check_tasks, return_exceptions=True)
                
                # Nettoyage historique ancien
                self._cleanup_old_history()
                
                # Attente avant prochain cycle
                await asyncio.sleep(self.check_interval)
                
        except asyncio.CancelledError:
            logger.info("Health monitoring loop cancelled")
        except Exception as e:
            logger.error(f"Erreur health monitoring loop: {str(e)}")
    
    def _should_run_check(self, component_state: ComponentHealth, health_check: HealthCheck) -> bool:
        """Determine si un health check doit etre execute"""
        
        time_since_last_check = datetime.now() - component_state.last_check
        
        return time_since_last_check.total_seconds() >= health_check.interval_seconds
    
    async def _execute_health_check(self, check_name: str, health_check: HealthCheck):
        """Execute un health check specifique"""
        
        try:
            # Execute check avec timeout
            result = await asyncio.wait_for(
                health_check.check_function(),
                timeout=health_check.timeout_seconds
            )
            
            # Mise a jour etat composant
            self._update_component_state(check_name, health_check, result)
            
            self.stats["total_checks_performed"] += 1
            
        except asyncio.TimeoutError:
            # Health check timeout
            result = HealthResult(
                component_name=check_name,
                status=HealthStatus.UNHEALTHY,
                timestamp=datetime.now(),
                response_time_ms=health_check.timeout_seconds * 1000,
                message=f"Health check timeout after {health_check.timeout_seconds}s",
                error="Timeout"
            )
            
            self._update_component_state(check_name, health_check, result)
            self.stats["total_failures_detected"] += 1
            
        except Exception as e:
            # Erreur health check
            result = HealthResult(
                component_name=check_name,
                status=HealthStatus.UNHEALTHY,
                timestamp=datetime.now(),
                response_time_ms=0,
                message=f"Health check failed: {str(e)}",
                error=str(e)
            )
            
            self._update_component_state(check_name, health_check, result)
            self.stats["total_failures_detected"] += 1
            
            logger.error(f"Health check error for {check_name}: {str(e)}")
    
    def _update_component_state(
        self, 
        check_name: str, 
        health_check: HealthCheck, 
        result: HealthResult
    ):
        """Met a jour etat composant avec resultat check"""
        
        component_state = self.component_states[check_name]
        
        # Mise a jour compteurs
        component_state.total_checks += 1
        component_state.last_check = result.timestamp
        
        if result.status == HealthStatus.HEALTHY:
            component_state.consecutive_successes += 1
            component_state.consecutive_failures = 0
        else:
            component_state.consecutive_failures += 1
            component_state.consecutive_successes = 0
            component_state.total_failures += 1
            component_state.last_error = result.error
        
        # Calcul nouveau status selon seuils
        old_status = component_state.status
        
        if (result.status == HealthStatus.HEALTHY and 
            component_state.consecutive_successes >= health_check.recovery_threshold):
            component_state.status = HealthStatus.HEALTHY
            
        elif (result.status != HealthStatus.HEALTHY and
              component_state.consecutive_failures >= health_check.failure_threshold):
            component_state.status = result.status
        
        # Log changement status
        if component_state.status != old_status:
            logger.info(f"Component {check_name} status changed: {old_status.value} -> {component_state.status.value}")
        
        # Mise a jour temps reponse moyen
        if component_state.total_checks > 1:
            component_state.avg_response_time_ms = (
                (component_state.avg_response_time_ms * (component_state.total_checks - 1) + result.response_time_ms) /
                component_state.total_checks
            )
        else:
            component_state.avg_response_time_ms = result.response_time_ms
        
        # Ajout a l'historique
        component_state.history.append(result)
        
        # Limiter taille historique
        if len(component_state.history) > 100:
            component_state.history = component_state.history[-100:]
    
    def _cleanup_old_history(self):
        """Nettoie historique ancien"""
        
        cutoff_time = datetime.now() - self.history_retention
        
        for component_state in self.component_states.values():
            component_state.history = [
                result for result in component_state.history
                if result.timestamp >= cutoff_time
            ]
    
    # === HEALTH CHECK FUNCTIONS ===
    
    async def _check_orchestrator(self) -> HealthResult:
        """Health check orchestrateur"""
        return await InternalComponentHealthChecker.check_orchestrator(
            self.dependencies.get("orchestrator")
        )
    
    async def _check_template_engine(self) -> HealthResult:
        """Health check template engine"""
        return await InternalComponentHealthChecker.check_template_engine(
            self.dependencies.get("template_engine")
        )
    
    async def _check_context_manager(self) -> HealthResult:
        """Health check context manager"""
        return await InternalComponentHealthChecker.check_context_manager(
            self.dependencies.get("context_manager")
        )
    
    async def _check_search_service(self) -> HealthResult:
        """Health check search service"""
        return await ExternalServiceHealthChecker.check_search_service(
            self.dependencies.get("search_service_url")
        )
    
    async def _check_llm_provider(self, provider_name: str, config: Dict[str, Any]) -> HealthResult:
        """Health check LLM provider"""
        return await ExternalServiceHealthChecker.check_llm_provider(
            provider_name,
            config.get("base_url", ""),
            config.get("api_key", ""),
            15
        )
    
    # === PUBLIC API ===
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Recupere sante globale systeme"""
        
        overall_status = HealthStatus.HEALTHY
        healthy_components = 0
        degraded_components = 0
        unhealthy_components = 0
        
        components_summary = {}
        
        for name, state in self.component_states.items():
            components_summary[name] = {
                "status": state.status.value,
                "last_check": state.last_check.isoformat(),
                "consecutive_failures": state.consecutive_failures,
                "avg_response_time_ms": state.avg_response_time_ms,
                "total_checks": state.total_checks,
                "total_failures": state.total_failures,
                "last_error": state.last_error
            }
            
            if state.status == HealthStatus.HEALTHY:
                healthy_components += 1
            elif state.status == HealthStatus.DEGRADED:
                degraded_components += 1
            else:
                unhealthy_components += 1
        
        # Determination status global
        if unhealthy_components > 0:
            overall_status = HealthStatus.UNHEALTHY
        elif degraded_components > 0:
            overall_status = HealthStatus.DEGRADED
        
        return {
            "overall_status": overall_status.value,
            "timestamp": datetime.now().isoformat(),
            "components": components_summary,
            "summary": {
                "healthy": healthy_components,
                "degraded": degraded_components,
                "unhealthy": unhealthy_components,
                "total": len(self.component_states)
            },
            "monitoring_stats": {
                **self.stats,
                "uptime_seconds": (datetime.now() - self.stats["monitoring_start_time"]).total_seconds(),
                "checks_configured": len(self.health_checks)
            }
        }
    
    async def get_component_health(self, component_name: str) -> Optional[Dict[str, Any]]:
        """Recupere sante composant specifique"""
        
        if component_name not in self.component_states:
            return None
        
        state = self.component_states[component_name]
        
        return {
            "name": state.name,
            "status": state.status.value,
            "last_check": state.last_check.isoformat(),
            "consecutive_failures": state.consecutive_failures,
            "consecutive_successes": state.consecutive_successes,
            "total_checks": state.total_checks,
            "total_failures": state.total_failures,
            "avg_response_time_ms": state.avg_response_time_ms,
            "last_error": state.last_error,
            "recent_history": [
                {
                    "timestamp": result.timestamp.isoformat(),
                    "status": result.status.value,
                    "response_time_ms": result.response_time_ms,
                    "message": result.message,
                    "error": result.error
                }
                for result in state.history[-10:]  # 10 derniers checks
            ]
        }

# Instance globale health monitor
health_monitor = HealthMonitor()

async def initialize_health_monitoring(dependencies: Dict[str, Any]):
    """Initialise monitoring sante avec dependencies"""
    
    try:
        health_monitor.configure_dependencies(dependencies)
        await health_monitor.start_monitoring()
        logger.info("Health monitoring initialized")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize health monitoring: {str(e)}")
        return False

async def cleanup_health_monitoring():
    """Cleanup monitoring sante"""
    
    try:
        await health_monitor.stop_monitoring()
        logger.info("Health monitoring cleaned up")
    except Exception as e:
        logger.error(f"Error cleaning up health monitoring: {str(e)}")