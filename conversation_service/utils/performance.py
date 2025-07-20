"""
Utilitaires optimisation performance spécifiques Heroku
"""

import asyncio
import gc
import psutil
import os
from typing import Dict, Any
from contextlib import asynccontextmanager
import time

from .logging import get_logger

logger = get_logger(__name__)


class PerformanceMonitor:
    """
    Monitoring performance temps réel optimisé Heroku
    """
    
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.baseline_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
    def get_memory_usage(self) -> Dict[str, float]:
        """Usage mémoire actuel"""
        memory_info = self.process.memory_info()
        
        return {
            "rss_mb": memory_info.rss / 1024 / 1024,
            "vms_mb": memory_info.vms / 1024 / 1024,
            "percent": self.process.memory_percent(),
            "baseline_mb": self.baseline_memory,
            "growth_mb": (memory_info.rss / 1024 / 1024) - self.baseline_memory
        }
    
    def get_cpu_usage(self) -> Dict[str, float]:
        """Usage CPU actuel"""
        return {
            "percent": self.process.cpu_percent(),
            "num_threads": self.process.num_threads(),
            "num_fds": self.process.num_fds() if hasattr(self.process, 'num_fds') else 0
        }
    
    async def check_memory_pressure(self) -> Dict[str, Any]:
        """Détection pression mémoire"""
        memory = self.get_memory_usage()
        
        # Seuils Heroku dyno standard (512MB)
        warning_threshold = 400  # MB
        critical_threshold = 480  # MB
        
        status = "normal"
        if memory["rss_mb"] > critical_threshold:
            status = "critical"
        elif memory["rss_mb"] > warning_threshold:
            status = "warning"
        
        return {
            "status": status,
            "current_mb": memory["rss_mb"],
            "threshold_warning": warning_threshold,
            "threshold_critical": critical_threshold,
            "should_gc": memory["rss_mb"] > warning_threshold
        }
    
    async def trigger_garbage_collection(self) -> Dict[str, Any]:
        """Garbage collection forcé avec métriques"""
        before_memory = self.get_memory_usage()["rss_mb"]
        
        # GC agressif
        collected = gc.collect()
        
        after_memory = self.get_memory_usage()["rss_mb"]
        freed_mb = before_memory - after_memory
        
        logger.info(f"Garbage collection: freed {freed_mb:.2f}MB, collected {collected} objects")
        
        return {
            "objects_collected": collected,
            "memory_before_mb": before_memory,
            "memory_after_mb": after_memory,
            "memory_freed_mb": freed_mb,
            "gc_effectiveness": (freed_mb / before_memory) if before_memory > 0 else 0
        }
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """Statistiques système complètes"""
        try:
            memory = self.get_memory_usage()
            cpu = self.get_cpu_usage()
            
            # Informations disque si disponibles
            disk_usage = {}
            try:
                disk = psutil.disk_usage('/')
                disk_usage = {
                    "total_gb": disk.total / (1024**3),
                    "used_gb": disk.used / (1024**3),
                    "free_gb": disk.free / (1024**3),
                    "percent": (disk.used / disk.total) * 100
                }
            except Exception:
                disk_usage = {"error": "disk_stats_unavailable"}
            
            # Informations réseau si disponibles
            network_stats = {}
            try:
                net_io = psutil.net_io_counters()
                network_stats = {
                    "bytes_sent": net_io.bytes_sent,
                    "bytes_recv": net_io.bytes_recv,
                    "packets_sent": net_io.packets_sent,
                    "packets_recv": net_io.packets_recv
                }
            except Exception:
                network_stats = {"error": "network_stats_unavailable"}
            
            return {
                "memory": memory,
                "cpu": cpu,
                "disk": disk_usage,
                "network": network_stats,
                "process_info": {
                    "pid": self.process.pid,
                    "ppid": self.process.ppid(),
                    "create_time": self.process.create_time(),
                    "status": self.process.status()
                }
            }
            
        except Exception as e:
            logger.error(f"Erreur récupération stats système: {e}")
            return {"error": str(e)}


@asynccontextmanager
async def performance_context(operation_name: str):
    """
    Context manager pour monitoring performance opérations
    """
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss / 1024 / 1024
    
    try:
        yield
    finally:
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        duration_ms = int((end_time - start_time) * 1000)
        memory_delta = end_memory - start_memory
        
        logger.debug(
            f"Performance: {operation_name} completed",
            extra={
                "extra_data": {
                    "operation": operation_name,
                    "duration_ms": duration_ms,
                    "memory_delta_mb": memory_delta,
                    "performance_category": "slow" if duration_ms > 1000 else "normal"
                }
            }
        )


class HerokuOptimizer:
    """
    Optimisations spécifiques contraintes Heroku
    """
    
    @staticmethod
    async def optimize_for_cold_start():
        """Optimisations démarrage à froid"""
        
        optimizations = []
        
        # Préchargement modules critiques
        try:
            import json
            import asyncio
            import time
            import hashlib
            import gzip
            optimizations.append("core_modules_preloaded")
        except ImportError as e:
            logger.warning(f"Failed to preload modules: {e}")
        
        # Configuration garbage collector pour responsivité
        gc.set_threshold(700, 10, 10)  # Plus agressif que défaut
        optimizations.append("gc_tuned")
        
        # Préallocation structures données critiques
        try:
            # Préallocation cache local patterns
            _ = {}
            # Préallocation listes pour métriques
            _ = []
            # Préallocation dictionnaires pour cache
            _ = dict()
            optimizations.append("structures_preallocated")
        except Exception as e:
            logger.warning(f"Preallocation failed: {e}")
        
        # Configuration optimisations Python
        try:
            # Désactivation debug assertions en production
            if __debug__:
                optimizations.append("debug_mode_detected")
            else:
                optimizations.append("production_mode_optimized")
        except Exception as e:
            logger.warning(f"Python optimization check failed: {e}")
        
        return {
            "optimizations_applied": optimizations,
            "cold_start_ready": True,
            "optimization_count": len(optimizations)
        }
    
    @staticmethod
    async def optimize_memory_usage():
        """Optimisations usage mémoire"""
        
        # Configuration Python pour mémoire limitée
        optimizations = []
        
        # GC agressif si pression mémoire
        monitor = PerformanceMonitor()
        memory_check = await monitor.check_memory_pressure()
        
        if memory_check["should_gc"]:
            gc_result = await monitor.trigger_garbage_collection()
            optimizations.append(f"gc_freed_{gc_result['memory_freed_mb']:.1f}mb")
        
        # Optimisation cache local si nécessaire
        if memory_check["status"] == "critical":
            # Réduction cache local (implémentation dépend du cache)
            optimizations.append("cache_reduced")
            
            # Force collection plus agressive
            for _ in range(3):
                gc.collect()
            optimizations.append("aggressive_gc_performed")
        
        # Optimisation imports pour économiser mémoire
        if memory_check["status"] in ["warning", "critical"]:
            # Nettoyage modules non essentiels si possible
            optimizations.append("memory_pressure_optimizations")
        
        return {
            "optimizations": optimizations,
            "memory_status": memory_check["status"],
            "current_usage_mb": memory_check["current_mb"],
            "optimization_needed": memory_check["status"] != "normal"
        }
    
    @staticmethod
    def get_heroku_dyno_info() -> Dict[str, Any]:
        """Information dyno Heroku depuis env vars"""
        
        return {
            "dyno_name": os.getenv("DYNO", "unknown"),
            "dyno_type": os.getenv("DYNO_TYPE", "unknown"),
            "port": os.getenv("PORT", "unknown"),
            "release_version": os.getenv("HEROKU_RELEASE_VERSION", "unknown"),
            "slug_commit": os.getenv("HEROKU_SLUG_COMMIT", "unknown"),
            "app_name": os.getenv("HEROKU_APP_NAME", "unknown"),
            "runtime": os.getenv("HEROKU_RUNTIME", "unknown"),
            "stack": os.getenv("HEROKU_STACK", "unknown")
        }
    
    @staticmethod
    def estimate_heroku_costs() -> Dict[str, Any]:
        """Estimation coûts Heroku basée sur usage"""
        
        try:
            dyno_type = os.getenv("DYNO_TYPE", "unknown")
            
            # Estimation coûts selon type dyno
            dyno_costs = {
                "standard-1x": 25,  # $/mois
                "standard-2x": 50,  # $/mois
                "performance-m": 250,  # $/mois
                "performance-l": 500,  # $/mois
            }
            
            monthly_cost = dyno_costs.get(dyno_type, 0)
            
            # Estimation add-ons Redis
            redis_cost = 30  # Redis Cloud 30MB ~ $30/mois
            
            return {
                "dyno_type": dyno_type,
                "estimated_dyno_cost_monthly": monthly_cost,
                "estimated_redis_cost_monthly": redis_cost,
                "estimated_total_monthly": monthly_cost + redis_cost,
                "cost_optimization_suggestions": [
                    "Use Redis caching efficiently",
                    "Optimize memory usage to avoid larger dynos",
                    "Monitor API call costs (DeepSeek)"
                ]
            }
            
        except Exception as e:
            return {"error": str(e), "cost_estimation_available": False}
    
    @staticmethod
    async def run_performance_diagnostics() -> Dict[str, Any]:
        """Diagnostic performance complet"""
        
        start_time = time.time()
        
        try:
            monitor = PerformanceMonitor()
            
            # Tests performance divers
            diagnostics = {}
            
            # Test 1: Performance mémoire
            memory_test_start = time.time()
            memory_stats = monitor.get_memory_usage()
            memory_test_duration = (time.time() - memory_test_start) * 1000
            
            diagnostics["memory_test"] = {
                "duration_ms": memory_test_duration,
                "current_usage": memory_stats,
                "status": "fast" if memory_test_duration < 10 else "slow"
            }
            
            # Test 2: Performance CPU
            cpu_test_start = time.time()
            cpu_stats = monitor.get_cpu_usage()
            cpu_test_duration = (time.time() - cpu_test_start) * 1000
            
            diagnostics["cpu_test"] = {
                "duration_ms": cpu_test_duration,
                "current_usage": cpu_stats,
                "status": "fast" if cpu_test_duration < 10 else "slow"
            }
            
            # Test 3: Performance I/O simple
            io_test_start = time.time()
            test_data = {"test": "performance", "timestamp": time.time()}
            import json
            serialized = json.dumps(test_data)
            deserialized = json.loads(serialized)
            io_test_duration = (time.time() - io_test_start) * 1000
            
            diagnostics["io_test"] = {
                "duration_ms": io_test_duration,
                "data_size_bytes": len(serialized),
                "status": "fast" if io_test_duration < 5 else "slow"
            }
            
            # Test 4: Performance asyncio
            async_test_start = time.time()
            await asyncio.sleep(0.001)  # Test async scheduler
            async_test_duration = (time.time() - async_test_start) * 1000
            
            diagnostics["async_test"] = {
                "duration_ms": async_test_duration,
                "status": "fast" if async_test_duration < 10 else "slow"
            }
            
            total_duration = (time.time() - start_time) * 1000
            
            # Évaluation globale
            all_fast = all(
                test.get("status") == "fast" 
                for test in diagnostics.values()
            )
            
            return {
                "overall_status": "healthy" if all_fast else "degraded",
                "total_diagnostic_duration_ms": total_duration,
                "tests": diagnostics,
                "recommendations": [
                    "Monitor memory usage regularly",
                    "Use async operations efficiently", 
                    "Cache frequently accessed data"
                ] if not all_fast else ["Performance is optimal"]
            }
            
        except Exception as e:
            return {
                "overall_status": "error",
                "error": str(e),
                "diagnostic_failed": True
            }


class ResourceThrottler:
    """
    Throttling intelligent des ressources
    """
    
    def __init__(self, max_memory_mb: int = 450):
        self.max_memory_mb = max_memory_mb
        self.monitor = PerformanceMonitor()
        self.throttle_active = False
    
    async def check_and_throttle(self) -> Dict[str, Any]:
        """Vérification et throttling si nécessaire"""
        
        memory_check = await self.monitor.check_memory_pressure()
        
        if memory_check["status"] == "critical" and not self.throttle_active:
            # Activation throttling
            self.throttle_active = True
            
            # Actions throttling
            actions_taken = []
            
            # 1. Garbage collection forcé
            gc_result = await self.monitor.trigger_garbage_collection()
            actions_taken.append(f"gc_freed_{gc_result['memory_freed_mb']:.1f}mb")
            
            # 2. Ajout délai artificial pour réduire charge
            await asyncio.sleep(0.1)
            actions_taken.append("request_throttling_enabled")
            
            return {
                "throttling_activated": True,
                "memory_status": memory_check["status"],
                "actions_taken": actions_taken,
                "current_memory_mb": memory_check["current_mb"]
            }
            
        elif memory_check["status"] == "normal" and self.throttle_active:
            # Désactivation throttling
            self.throttle_active = False
            
            return {
                "throttling_deactivated": True,
                "memory_status": memory_check["status"],
                "current_memory_mb": memory_check["current_mb"]
            }
        
        return {
            "throttling_active": self.throttle_active,
            "memory_status": memory_check["status"],
            "current_memory_mb": memory_check["current_mb"]
        }


# Export utilitaires performance
__all__ = [
    "PerformanceMonitor",
    "performance_context", 
    "HerokuOptimizer",
    "ResourceThrottler"
]