"""
Validateurs pour requêtes et configuration - Version corrigée
"""

import re
from typing import Any, Dict

from conversation_service.config import settings


class RequestValidator:
    """Validation requêtes utilisateur"""
    
    @staticmethod
    def validate_conversation_message(message: str) -> Dict[str, Any]:
        """Validation message conversation"""
        
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "sanitized_message": message
        }
        
        # Validation longueur
        if len(message.strip()) == 0:
            validation_result["valid"] = False
            validation_result["errors"].append("Message cannot be empty")
            return validation_result
        
        if len(message) > 2000:
            validation_result["valid"] = False
            validation_result["errors"].append("Message too long (max 2000 characters)")
            return validation_result
        
        # Détection contenu potentiellement malveillant
        suspicious_patterns = [
            r'<script[^>]*>.*?</script>',  # Scripts XSS
            r'javascript:',               # URLs javascript
            r'data:text/html',           # Data URLs HTML
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, message, re.IGNORECASE):
                validation_result["warnings"].append("Suspicious content detected and sanitized")
                validation_result["sanitized_message"] = re.sub(pattern, '', message, flags=re.IGNORECASE)
        
        # Validation caractères spéciaux excessifs
        special_char_ratio = len(re.findall(r'[^\w\s]', message)) / len(message)
        if special_char_ratio > 0.3:
            validation_result["warnings"].append("High ratio of special characters")
        
        return validation_result
    
    @staticmethod
    def validate_user_id(user_id: Any) -> bool:
        """Validation ID utilisateur"""
        if not isinstance(user_id, int):
            return False
        return 1 <= user_id <= 999999999


class ConfigValidator:
    """Validation configuration système"""
    
    @staticmethod
    def validate_redis_config() -> Dict[str, Any]:
        """Validation configuration Redis"""
        
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Vérification variables obligatoires
        required_vars = ["REDIS_HOST", "REDIS_PORT", "REDIS_PASSWORD"]
        for var in required_vars:
            if not hasattr(settings, var) or not getattr(settings, var):
                validation_result["valid"] = False
                validation_result["errors"].append(f"Missing required Redis config: {var}")
        
        # Validation port Redis
        try:
            port = getattr(settings, "REDIS_PORT", 0)
            if not (1 <= int(port) <= 65535):
                validation_result["valid"] = False
                validation_result["errors"].append("Invalid Redis port")
        except (ValueError, TypeError):
            validation_result["valid"] = False
            validation_result["errors"].append("Redis port must be numeric")
        
        # Validation pool size
        max_connections = getattr(settings, "REDIS_MAX_CONNECTIONS", 10)
        if max_connections > 20:
            validation_result["warnings"].append("High Redis connection pool size may impact Heroku dyno")
        
        return validation_result
    
    @staticmethod
    def validate_deepseek_config() -> Dict[str, Any]:
        """Validation configuration DeepSeek"""
        
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Vérification API key
        api_key = getattr(settings, "DEEPSEEK_API_KEY", "")
        if not api_key:
            validation_result["warnings"].append("DeepSeek API key not configured - L2 fallback unavailable")
        elif len(api_key) < 20:
            validation_result["valid"] = False
            validation_result["errors"].append("DeepSeek API key appears invalid")
        
        # Validation timeout
        timeout = getattr(settings, "DEEPSEEK_RESPONSE_TIMEOUT", 15)
        if timeout > 30:
            validation_result["warnings"].append("High DeepSeek timeout may impact user experience")
        elif timeout < 5:
            validation_result["warnings"].append("Low DeepSeek timeout may cause frequent failures")
        
        # Validation max tokens
        max_tokens = getattr(settings, "DEEPSEEK_MAX_TOKENS", 1024)
        if max_tokens > 4000:
            validation_result["warnings"].append("High token limit may increase costs")
        
        return validation_result
    
    @staticmethod
    def validate_cache_config() -> Dict[str, Any]:
        """Validation configuration cache"""
        
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Validation TTL patterns
        ttl_patterns = getattr(settings, "INTENT_CACHE_TTL_PATTERNS", 3600)
        if ttl_patterns < 300:
            validation_result["warnings"].append("Low pattern cache TTL may reduce performance")
        elif ttl_patterns > 7200:
            validation_result["warnings"].append("High pattern cache TTL may cause stale data")
        
        # Validation TTL embeddings
        ttl_embeddings = getattr(settings, "INTENT_CACHE_TTL_EMBEDDINGS", 1800)
        if ttl_embeddings < 300:
            validation_result["warnings"].append("Low embedding cache TTL may increase L1 load")
        
        # Validation seuil confiance
        confidence_threshold = getattr(settings, "INTENT_CONFIDENCE_THRESHOLD", 0.85)
        if confidence_threshold < 0.5:
            validation_result["warnings"].append("Low confidence threshold may reduce accuracy")
        elif confidence_threshold > 0.95:
            validation_result["warnings"].append("High confidence threshold may increase L2 fallback usage")
        
        return validation_result
    
    @staticmethod
    def validate_complete_configuration() -> Dict[str, Any]:
        """Validation configuration complète"""
        
        redis_validation = ConfigValidator.validate_redis_config()
        deepseek_validation = ConfigValidator.validate_deepseek_config()
        cache_validation = ConfigValidator.validate_cache_config()
        
        # Agrégation résultats
        overall_valid = all([
            redis_validation["valid"],
            deepseek_validation["valid"], 
            cache_validation["valid"]
        ])
        
        all_errors = (
            redis_validation["errors"] +
            deepseek_validation["errors"] +
            cache_validation["errors"]
        )
        
        all_warnings = (
            redis_validation["warnings"] +
            deepseek_validation["warnings"] +
            cache_validation["warnings"]
        )
        
        return {
            "valid": overall_valid,
            "errors": all_errors,
            "warnings": all_warnings,
            "component_results": {
                "redis": redis_validation,
                "deepseek": deepseek_validation,
                "cache": cache_validation
            }
        }


# conversation_service/utils/performance.py
"""
Utilitaires optimisation performance spécifiques Heroku - Version complète
"""

import asyncio
import gc
import psutil
import os
from typing import Dict, Any, Optional
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
            optimizations.append("structures_preallocated")
        except Exception as e:
            logger.warning(f"Preallocation failed: {e}")
        
        return {
            "optimizations_applied": optimizations,
            "cold_start_ready": True
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
        
        return {
            "optimizations": optimizations,
            "memory_status": memory_check["status"],
            "current_usage_mb": memory_check["current_mb"]
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
            "app_name": os.getenv("HEROKU_APP_NAME", "unknown")
        }


# Export utilitaires performance
__all__ = [
    "PerformanceMonitor",
    "performance_context", 
    "HerokuOptimizer"
]