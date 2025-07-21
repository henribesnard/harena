"""
📋 Logging structuré pour observabilité

Configuration logging FastAPI + Heroku avec logs structurés JSON,
contexte requête automatique et helpers spécialisés.
"""

import json
import logging
import sys
import time
from typing import Dict, Any, Optional
from contextvars import ContextVar
from datetime import datetime

from config_service.config import settings

# ==========================================
# CONTEXTE LOGGING
# ==========================================

# Variables contexte pour tracking requêtes
request_id_context: ContextVar[Optional[str]] = ContextVar('request_id', default=None)
user_id_context: ContextVar[Optional[str]] = ContextVar('user_id', default=None)

def set_request_context(request_id: str, user_id: str = None):
    """Définit contexte requête pour logs"""
    request_id_context.set(request_id)
    if user_id:
        user_id_context.set(user_id)

def get_request_context() -> Dict[str, Optional[str]]:
    """Récupère contexte requête actuel"""
    return {
        "request_id": request_id_context.get(),
        "user_id": user_id_context.get()
    }

# ==========================================
# FORMATTERS STRUCTURÉS
# ==========================================

class StructuredJSONFormatter(logging.Formatter):
    """
    Formatter JSON structuré pour logs observables
    
    Format de sortie optimisé pour parsing automatique
    et recherche dans systèmes de logging centralisés.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """Formatage log en JSON structuré"""
        
        # Données de base
        log_entry = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Contexte requête si disponible
        context = get_request_context()
        if context["request_id"]:
            log_entry["request_id"] = context["request_id"]
        if context["user_id"]:
            log_entry["user_id"] = context["user_id"]
        
        # Métadonnées service
        log_entry["service"] = "conversation_service"
        log_entry["version"] = "1.0.0"
        
        # Exception si présente
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": self.formatException(record.exc_info)
            }
        
        # Attributs personnalisés
        extra_fields = {}
        for key, value in record.__dict__.items():
            if key not in [
                'name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                'filename', 'module', 'lineno', 'funcName', 'created',
                'msecs', 'relativeCreated', 'thread', 'threadName',
                'processName', 'process', 'getMessage', 'exc_info', 'exc_text',
                'stack_info'
            ]:
                extra_fields[key] = value
        
        if extra_fields:
            log_entry["extra"] = extra_fields
        
        return json.dumps(log_entry, ensure_ascii=False, default=str)

class DevelopmentFormatter(logging.Formatter):
    """Formatter lisible pour développement local"""
    
    def format(self, record: logging.LogRecord) -> str:
        """Formatage lisible développement"""
        
        # Couleurs pour niveaux
        colors = {
            'DEBUG': '\033[36m',    # Cyan
            'INFO': '\033[32m',     # Vert
            'WARNING': '\033[33m',  # Jaune
            'ERROR': '\033[31m',    # Rouge
            'CRITICAL': '\033[91m'  # Rouge brillant
        }
        reset_color = '\033[0m'
        
        # Timestamp formaté
        timestamp = datetime.fromtimestamp(record.created).strftime('%H:%M:%S.%f')[:-3]
        
        # Couleur niveau
        level_color = colors.get(record.levelname, '')
        level_display = f"{level_color}{record.levelname:<8}{reset_color}"
        
        # Contexte requête
        context = get_request_context()
        context_str = ""
        if context["request_id"]:
            context_str = f"[{context['request_id'][:8]}]"
            if context["user_id"]:
                context_str += f"[{context['user_id'][:10]}]"
        
        # Message principal
        base_message = f"{timestamp} {level_display} {record.name:<20} {context_str} {record.getMessage()}"
        
        # Exception si présente
        if record.exc_info:
            exception_text = self.formatException(record.exc_info)
            base_message += f"\n{exception_text}"
        
        return base_message

# ==========================================
# CONFIGURATION LOGGING
# ==========================================

def setup_logging():
    """Configuration logging structuré selon environnement"""
    
    # Niveau logging depuis configuration
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    
    # Nettoyage handlers existants
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    
    # Handler console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    
    # Formatter selon environnement
    if settings.DEBUG:
        # Développement : format lisible
        formatter = DevelopmentFormatter()
    else:
        # Production : JSON structuré
        formatter = StructuredJSONFormatter()
    
    console_handler.setFormatter(formatter)
    
    # Configuration root logger
    root_logger.addHandler(console_handler)
    root_logger.setLevel(log_level)
    
    # Suppression logs verbeux librairies
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    
    # Logger conversation service plus verbeux
    logging.getLogger("conversation_service").setLevel(log_level)
    
    print(f"📋 Logging configuré - Niveau: {settings.LOG_LEVEL}, Mode: {'DEV' if settings.DEBUG else 'PROD'}")

# ==========================================
# HELPERS LOGGING SPÉCIALISÉS
# ==========================================

def log_intent_detection(
    event: str,
    level: str = None,
    intent: str = None,
    confidence: float = None,
    latency_ms: float = None,
    cache_hit: bool = None,
    user_id: str = None,
    request_id: str = None,
    error: str = None,
    **kwargs
):
    """
    Helper logging événements détection intention
    
    Args:
        event: Type événement (classification_start, l0_success, etc.)
        level: Niveau détection (L0/L1/L2)
        intent: Intention détectée
        confidence: Score confiance
        latency_ms: Latence traitement
        cache_hit: Hit cache
        user_id: ID utilisateur
        request_id: ID requête
        error: Message erreur
        **kwargs: Métadonnées supplémentaires
    """
    logger = logging.getLogger("conversation_service.intent_detection")
    
    # Construction données log
    log_data = {
        "event": event,
        "component": "intent_detection"
    }
    
    # Ajout données si fournies
    if level:
        log_data["level"] = level
    if intent:
        log_data["intent"] = intent
    if confidence is not None:
        log_data["confidence"] = round(confidence, 3)
    if latency_ms is not None:
        log_data["latency_ms"] = round(latency_ms, 2)
    if cache_hit is not None:
        log_data["cache_hit"] = cache_hit
    if user_id:
        log_data["user_id"] = user_id
    if request_id:
        log_data["request_id"] = request_id
    if error:
        log_data["error"] = error
    
    # Métadonnées supplémentaires
    log_data.update(kwargs)
    
    # Niveau log selon événement
    if event.endswith("_error") or error:
        logger.error("Intent detection event", extra=log_data)
    elif event.endswith("_warning"):
        logger.warning("Intent detection event", extra=log_data)
    elif event.endswith("_success") or event == "classification_success":
        logger.info("Intent detection event", extra=log_data)
    else:
        logger.debug("Intent detection event", extra=log_data)

def log_cache_operation(
    operation: str,
    cache_type: str,
    key: str = None,
    hit: bool = None,
    ttl_seconds: int = None,
    size_bytes: int = None,
    error: str = None,
    **kwargs
):
    """
    Helper logging opérations cache
    
    Args:
        operation: Type opération (get, set, clear, etc.)
        cache_type: Type cache (redis, local, l0, l1, l2)
        key: Clé cache (tronquée)
        hit: Hit/miss pour GET
        ttl_seconds: TTL pour SET
        size_bytes: Taille donnée
        error: Message erreur
        **kwargs: Métadonnées supplémentaires
    """
    logger = logging.getLogger("conversation_service.cache")
    
    log_data = {
        "event": "cache_operation",
        "operation": operation,
        "cache_type": cache_type,
        "component": "cache"
    }
    
    if key:
        log_data["cache_key"] = key[:50] + "..." if len(key) > 50 else key
    if hit is not None:
        log_data["cache_hit"] = hit
    if ttl_seconds:
        log_data["ttl_seconds"] = ttl_seconds
    if size_bytes:
        log_data["size_bytes"] = size_bytes
    if error:
        log_data["error"] = error
    
    log_data.update(kwargs)
    
    if error:
        logger.warning("Cache operation", extra=log_data)
    else:
        logger.debug("Cache operation", extra=log_data)

def log_api_call(
    api_name: str,
    method: str,
    endpoint: str,
    status_code: int = None,
    latency_ms: float = None,
    request_size: int = None,
    response_size: int = None,
    error: str = None,
    **kwargs
):
    """
    Helper logging appels API externes
    
    Args:
        api_name: Nom API (deepseek, search_service, etc.)
        method: Méthode HTTP
        endpoint: Endpoint appelé
        status_code: Code retour HTTP
        latency_ms: Latence appel
        request_size: Taille requête
        response_size: Taille réponse
        error: Message erreur
        **kwargs: Métadonnées supplémentaires
    """
    logger = logging.getLogger(f"conversation_service.api.{api_name}")
    
    log_data = {
        "event": "api_call",
        "api_name": api_name,
        "method": method,
        "endpoint": endpoint,
        "component": "api_client"
    }
    
    if status_code:
        log_data["status_code"] = status_code
    if latency_ms is not None:
        log_data["latency_ms"] = round(latency_ms, 2)
    if request_size:
        log_data["request_size"] = request_size
    if response_size:
        log_data["response_size"] = response_size
    if error:
        log_data["error"] = error
    
    log_data.update(kwargs)
    
    # Niveau selon résultat
    if error or (status_code and status_code >= 500):
        logger.error("API call", extra=log_data)
    elif status_code and status_code >= 400:
        logger.warning("API call", extra=log_data)
    else:
        logger.info("API call", extra=log_data)

def log_performance_metric(
    metric_name: str,
    value: float,
    unit: str = None,
    component: str = None,
    level: str = None,
    threshold: float = None,
    threshold_exceeded: bool = None,
    **kwargs
):
    """
    Helper logging métriques performance
    
    Args:
        metric_name: Nom métrique (latency, success_rate, etc.)
        value: Valeur métrique
        unit: Unité (ms, percent, count)
        component: Composant concerné
        level: Niveau si applicable (L0/L1/L2)
        threshold: Seuil si applicable
        threshold_exceeded: Seuil dépassé
        **kwargs: Métadonnées supplémentaires
    """
    logger = logging.getLogger("conversation_service.metrics")
    
    log_data = {
        "event": "performance_metric",
        "metric_name": metric_name,
        "value": round(value, 3) if isinstance(value, float) else value,
        "component": "metrics"
    }
    
    if unit:
        log_data["unit"] = unit
    if component:
        log_data["target_component"] = component
    if level:
        log_data["level"] = level
    if threshold is not None:
        log_data["threshold"] = threshold
        log_data["threshold_exceeded"] = threshold_exceeded
    
    log_data.update(kwargs)
    
    # Niveau selon seuil
    if threshold_exceeded:
        logger.warning("Performance metric", extra=log_data)
    else:
        logger.debug("Performance metric", extra=log_data)

def log_business_event(
    event_type: str,
    user_id: str = None,
    intent: str = None,
    success: bool = None,
    value: float = None,
    metadata: Dict[str, Any] = None,
    **kwargs
):
    """
    Helper logging événements métier/business
    
    Args:
        event_type: Type événement business (user_query, intent_classified, etc.)
        user_id: ID utilisateur
        intent: Intention traitée
        success: Succès opération
        value: Valeur métier associée
        metadata: Métadonnées business
        **kwargs: Données supplémentaires
    """
    logger = logging.getLogger("conversation_service.business")
    
    log_data = {
        "event": "business_event",
        "event_type": event_type,
        "component": "business"
    }
    
    if user_id:
        log_data["user_id"] = user_id
    if intent:
        log_data["intent"] = intent
    if success is not None:
        log_data["success"] = success
    if value is not None:
        log_data["business_value"] = value
    if metadata:
        log_data["metadata"] = metadata
    
    log_data.update(kwargs)
    
    # Toujours niveau INFO pour événements business
    logger.info("Business event", extra=log_data)

# ==========================================
# HELPERS CONTEXTE REQUÊTE
# ==========================================

class RequestContextManager:
    """
    Context manager pour gestion automatique contexte requête
    
    Usage:
        async with RequestContextManager(request_id, user_id):
            # Logs incluront automatiquement request_id et user_id
            logger.info("Message avec contexte")
    """
    
    def __init__(self, request_id: str, user_id: str = None):
        self.request_id = request_id
        self.user_id = user_id
        self.previous_request_id = None
        self.previous_user_id = None
    
    def __enter__(self):
        # Sauvegarde contexte précédent
        self.previous_request_id = request_id_context.get()
        self.previous_user_id = user_id_context.get()
        
        # Définition nouveau contexte
        set_request_context(self.request_id, self.user_id)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restauration contexte précédent
        request_id_context.set(self.previous_request_id)
        user_id_context.set(self.previous_user_id)

# ==========================================
# MONITORING LOGS
# ==========================================

class LogMetricsCollector:
    """
    Collecteur métriques logs pour monitoring
    
    Suit volume logs par niveau/composant pour alertes
    """
    
    def __init__(self):
        self.log_counts = {
            "DEBUG": 0,
            "INFO": 0, 
            "WARNING": 0,
            "ERROR": 0,
            "CRITICAL": 0
        }
        self.component_counts = {}
        self.error_patterns = {}
        self.start_time = time.time()
    
    def record_log(self, level: str, logger_name: str, message: str):
        """Enregistre log pour métriques"""
        self.log_counts[level] = self.log_counts.get(level, 0) + 1
        
        # Composant depuis nom logger
        component = logger_name.split('.')[-1] if '.' in logger_name else logger_name
        self.component_counts[component] = self.component_counts.get(component, 0) + 1
        
        # Patterns erreurs fréquentes
        if level in ["ERROR", "CRITICAL"]:
            pattern_key = message[:50]  # 50 premiers caractères
            self.error_patterns[pattern_key] = self.error_patterns.get(pattern_key, 0) + 1
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Résumé métriques logs"""
        uptime_hours = (time.time() - self.start_time) / 3600
        total_logs = sum(self.log_counts.values())
        
        return {
            "uptime_hours": round(uptime_hours, 2),
            "total_logs": total_logs,
            "logs_per_hour": round(total_logs / max(uptime_hours, 0.01), 1),
            "by_level": self.log_counts.copy(),
            "by_component": dict(sorted(self.component_counts.items(), key=lambda x: x[1], reverse=True)[:10]),
            "error_rate": (self.log_counts.get("ERROR", 0) + self.log_counts.get("CRITICAL", 0)) / max(total_logs, 1),
            "top_error_patterns": dict(sorted(self.error_patterns.items(), key=lambda x: x[1], reverse=True)[:5])
        }
    
    def reset_metrics(self):
        """Reset métriques"""
        self.log_counts = {level: 0 for level in self.log_counts}
        self.component_counts.clear()
        self.error_patterns.clear()
        self.start_time = time.time()

# Instance globale collecteur
_log_metrics = LogMetricsCollector()

class MetricsLoggingHandler(logging.Handler):
    """Handler pour collecte métriques logs"""
    
    def emit(self, record):
        """Collecte métrique pour chaque log"""
        _log_metrics.record_log(record.levelname, record.name, record.getMessage())

def get_log_metrics() -> Dict[str, Any]:
    """Récupération métriques logs"""
    return _log_metrics.get_metrics_summary()

def setup_log_metrics_collection():
    """Activation collecte métriques logs"""
    metrics_handler = MetricsLoggingHandler()
    metrics_handler.setLevel(logging.DEBUG)
    
    root_logger = logging.getLogger()
    root_logger.addHandler(metrics_handler)

# ==========================================
# HELPERS DEBUG ET DEVELOPMENT
# ==========================================

def log_request_timing(operation_name: str):
    """
    Décorateur pour logging timing opérations
    
    Usage:
        @log_request_timing("detect_intent")
        async def detect_intent(self, query):
            ...
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            logger = logging.getLogger(f"conversation_service.timing.{operation_name}")
            
            try:
                result = await func(*args, **kwargs)
                latency_ms = (time.time() - start_time) * 1000
                
                logger.info(
                    f"Operation completed: {operation_name}",
                    extra={
                        "operation": operation_name,
                        "latency_ms": round(latency_ms, 2),
                        "success": True
                    }
                )
                
                return result
                
            except Exception as e:
                latency_ms = (time.time() - start_time) * 1000
                
                logger.error(
                    f"Operation failed: {operation_name}",
                    extra={
                        "operation": operation_name,
                        "latency_ms": round(latency_ms, 2),
                        "success": False,
                        "error": str(e)
                    }
                )
                
                raise
        
        return wrapper
    return decorator

def debug_log_context():
    """Debug contexte logging actuel"""
    context = get_request_context()
    logger = logging.getLogger("conversation_service.debug")
    
    logger.debug(
        "Current logging context",
        extra={
            "current_context": context,
            "debug_event": "context_check"
        }
    )

def log_system_resources():
    """Log ressources système pour debug"""
    import psutil
    import os
    
    logger = logging.getLogger("conversation_service.system")
    
    try:
        memory_info = psutil.virtual_memory()
        disk_info = psutil.disk_usage('/')
        
        logger.info(
            "System resources",
            extra={
                "memory_percent": memory_info.percent,
                "memory_available_mb": round(memory_info.available / 1024 / 1024, 1),
                "disk_percent": disk_info.percent,
                "disk_free_gb": round(disk_info.free / 1024 / 1024 / 1024, 1),
                "process_count": len(psutil.pids()),
                "load_average": os.getloadavg() if hasattr(os, 'getloadavg') else None
            }
        )
    except Exception as e:
        logger.warning(f"Could not collect system resources: {e}")

# ==========================================
# CONFIGURATION PRODUCTION
# ==========================================

def configure_production_logging():
    """Configuration logging optimisée production"""
    
    # Niveau WARNING pour réduire volume
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.WARNING)
    
    # Logger conversation service reste INFO
    logging.getLogger("conversation_service").setLevel(logging.INFO)
    
    # Suppression logs debug composants
    debug_loggers = [
        "conversation_service.intent_detection.pattern_matcher",
        "conversation_service.cache",
        "httpx", "httpcore", "asyncio"
    ]
    
    for logger_name in debug_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    
    print("📋 Logging production configuré - Volume optimisé")

def configure_development_logging():
    """Configuration logging détaillée développement"""
    
    # Niveau DEBUG pour développement
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # Tous les composants verbeux
    logging.getLogger("conversation_service").setLevel(logging.DEBUG)
    
    print("📋 Logging développement configuré - Mode verbeux")

# ==========================================
# EXPORT FONCTIONS PRINCIPALES
# ==========================================

__all__ = [
    "setup_logging",
    "configure_production_logging", 
    "configure_development_logging",
    "log_intent_detection",
    "log_cache_operation",
    "log_api_call", 
    "log_performance_metric",
    "log_business_event",
    "set_request_context",
    "get_request_context",
    "RequestContextManager",
    "get_log_metrics",
    "setup_log_metrics_collection",
    "log_request_timing",
    "debug_log_context",
    "log_system_resources"
]