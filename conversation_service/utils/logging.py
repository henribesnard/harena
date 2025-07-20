"""
Configuration logging centralisée optimisée pour Heroku
Structure logs pour observabilité et debugging
"""

import sys
import json
import logging
from typing import  Optional
from datetime import datetime
from contextvars import ContextVar

from conversation_service.config import settings

# Variables contextuelles pour correlation logs
request_id_var: ContextVar[Optional[str]] = ContextVar('request_id', default=None)
user_id_var: ContextVar[Optional[int]] = ContextVar('user_id', default=None)


class StructuredFormatter(logging.Formatter):
    """
    Formatter logs structurés JSON pour Heroku Logs/Datadog
    """
    
    def __init__(self):
        super().__init__()
        
    def format(self, record: logging.LogRecord) -> str:
        """Format log en JSON structuré"""
        
        # Données de base
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "service": "conversation_service",
            "version": "1.0.0"
        }
        
        # Contexte requête si disponible
        request_id = request_id_var.get()
        if request_id:
            log_entry["request_id"] = request_id
            
        user_id = user_id_var.get()
        if user_id:
            log_entry["user_id"] = user_id
        
        # Métadonnées additionnelles du record
        if hasattr(record, 'extra_data'):
            log_entry.update(record.extra_data)
        
        # Information exception si présente
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": self.formatException(record.exc_info)
            }
        
        # Métadonnées performance si disponibles
        if hasattr(record, 'duration_ms'):
            log_entry["duration_ms"] = record.duration_ms
            
        if hasattr(record, 'cache_hit'):
            log_entry["cache_hit"] = record.cache_hit
        
        return json.dumps(log_entry, ensure_ascii=False)


class ConversationServiceLogger:
    """
    Logger spécialisé avec méthodes helper pour cas d'usage métier
    """
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        
    def log_intent_detection(
        self, 
        query: str,
        intent_type: str,
        confidence: float,
        level: str,
        duration_ms: int,
        cache_hit: bool = False
    ):
        """Log spécialisé détection intention"""
        extra_data = {
            "event_type": "intent_detection",
            "query_length": len(query),
            "intent_type": intent_type,
            "confidence": confidence,
            "detection_level": level,
            "duration_ms": duration_ms,
            "cache_hit": cache_hit
        }
        
        self.logger.info(
            f"Intent detected: {intent_type} (confidence: {confidence:.2f})",
            extra={"extra_data": extra_data}
        )
    
    def log_cache_operation(
        self,
        operation: str,
        key: str,
        hit: bool,
        duration_ms: Optional[int] = None
    ):
        """Log opérations cache"""
        extra_data = {
            "event_type": "cache_operation",
            "operation": operation,
            "cache_key": key,
            "cache_hit": hit
        }
        
        if duration_ms is not None:
            extra_data["duration_ms"] = duration_ms
        
        self.logger.debug(
            f"Cache {operation}: {'HIT' if hit else 'MISS'} for {key}",
            extra={"extra_data": extra_data}
        )
    
    def log_api_call(
        self,
        provider: str,
        endpoint: str,
        status_code: int,
        duration_ms: int,
        tokens_used: Optional[int] = None
    ):
        """Log appels API externes"""
        extra_data = {
            "event_type": "external_api_call",
            "provider": provider,
            "endpoint": endpoint,
            "status_code": status_code,
            "duration_ms": duration_ms
        }
        
        if tokens_used:
            extra_data["tokens_used"] = tokens_used
        
        level = "info" if 200 <= status_code < 300 else "warning"
        getattr(self.logger, level)(
            f"API call {provider}: {status_code} in {duration_ms}ms",
            extra={"extra_data": extra_data}
        )
    
    def log_conversation(
        self,
        conversation_id: str,
        user_query: str,
        response_length: int,
        processing_time_ms: int
    ):
        """Log conversation complète"""
        extra_data = {
            "event_type": "conversation",
            "conversation_id": conversation_id,
            "query_length": len(user_query),
            "response_length": response_length,
            "processing_time_ms": processing_time_ms
        }
        
        self.logger.info(
            f"Conversation {conversation_id} processed in {processing_time_ms}ms",
            extra={"extra_data": extra_data}
        )
    
    # Délégation méthodes standard
    def info(self, message: str, **kwargs):
        self.logger.info(message, **kwargs)
        
    def warning(self, message: str, **kwargs):
        self.logger.warning(message, **kwargs)
        
    def error(self, message: str, **kwargs):
        self.logger.error(message, **kwargs)
        
    def debug(self, message: str, **kwargs):
        self.logger.debug(message, **kwargs)


def setup_logging():
    """
    Configuration logging globale optimisée Heroku
    """
    
    # Configuration niveau global
    log_level = getattr(logging, settings.LOG_LEVEL.upper())
    
    # Handler console pour Heroku (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(StructuredFormatter())
    
    # Configuration logger racine
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(console_handler)
    
    # Silenciation logs verbeux libraries externes
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("redis").setLevel(logging.INFO)
    
    # Logger conversation service plus verbeux en dev
    app_logger = logging.getLogger("conversation_service")
    if settings.ENVIRONMENT == "development":
        app_logger.setLevel(logging.DEBUG)


def get_logger(name: str) -> ConversationServiceLogger:
    """
    Factory logger avec métadonnées service
    """
    return ConversationServiceLogger(name)


def set_request_context(request_id: str, user_id: Optional[int] = None):
    """Helper pour définir contexte requête dans logs"""
    request_id_var.set(request_id)
    if user_id:
        user_id_var.set(user_id)


def clear_request_context():
    """Helper pour nettoyer contexte requête"""
    request_id_var.set(None)
    user_id_var.set(None)