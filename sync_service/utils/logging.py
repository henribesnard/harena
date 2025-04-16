# sync_service/utils/logging.py
import json
import logging
import os
import sys
from datetime import datetime
from typing import Dict, Any, Optional

class StructuredLogRecord(logging.LogRecord):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.timestamp = datetime.utcnow().isoformat()

class StructuredFormatter(logging.Formatter):
    def format(self, record):
        log_obj = {
            "timestamp": getattr(record, "timestamp", datetime.utcnow().isoformat()),
            "level": record.levelname,
            "module": record.module,
            "message": record.getMessage(),
        }
        
        # Ajouter des méta-données contextuelles si présentes
        if hasattr(record, "user_id"):
            log_obj["user_id"] = record.user_id
            
        if hasattr(record, "bridge_item_id"):
            log_obj["bridge_item_id"] = record.bridge_item_id
            
        if hasattr(record, "account_id"):
            log_obj["account_id"] = record.account_id
            
        if hasattr(record, "request_id"):
            log_obj["request_id"] = record.request_id
            
        if hasattr(record, "event_type"):
            log_obj["event_type"] = record.event_type
        
        # Ajouter les informations d'exception si présentes
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
            
        # Ajouter des données additionnelles (utilisé via extra={})
        for key, value in record.__dict__.items():
            if key.startswith('data_') and not key.startswith('_'):
                clean_key = key[5:]  # enlever le préfixe 'data_'
                log_obj[clean_key] = value
            
        return json.dumps(log_obj)

class ColoredConsoleFormatter(logging.Formatter):
    """Formateur de logs avec couleurs pour la console"""
    
    COLORS = {
        'DEBUG': '\033[94m',    # Bleu
        'INFO': '\033[92m',     # Vert
        'WARNING': '\033[93m',  # Jaune
        'ERROR': '\033[91m',    # Rouge
        'CRITICAL': '\033[91m\033[1m', # Rouge gras
        'ENDC': '\033[0m',      # Reset
    }
    
    def format(self, record):
        log_message = super().format(record)
        color = self.COLORS.get(record.levelname, self.COLORS['ENDC'])
        return f"{color}{log_message}{self.COLORS['ENDC']}"

def setup_structured_logging(level: Optional[str] = None):
    """Configure structured logging for the sync service.
    
    Args:
        level: Optional log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
               Defaults to the LOG_LEVEL environment variable or INFO
    """
    # Détermine le niveau de log
    if level is None:
        level = os.environ.get("LOG_LEVEL", "INFO").upper()
    
    numeric_level = getattr(logging, level, logging.INFO)
    
    # Remplacer la factory de LogRecord par notre version structurée
    logging.setLogRecordFactory(StructuredLogRecord)
    
    # Configurer le handler pour la sortie standard
    console_handler = logging.StreamHandler(sys.stdout)
    
    # Déterminer le formateur en fonction de l'environnement
    if os.environ.get("ENVIRONMENT", "").lower() == "development":
        # En développement, utiliser un format plus lisible avec couleurs
        formatter = ColoredConsoleFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    else:
        # En production, utiliser le format JSON structuré
        formatter = StructuredFormatter()
    
    console_handler.setFormatter(formatter)
    
    # Configurer le logger root
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Supprimer les handlers existants pour éviter les duplications
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    root_logger.addHandler(console_handler)
    
    # Configurer le logger spécifique du service
    logger = logging.getLogger("sync_service")
    logger.setLevel(numeric_level)
    
    # Si en mode DEBUG, configurer des niveaux spécifiques pour certains modules
    if numeric_level == logging.DEBUG:
        # Réduire le bruit des logs httpx en mode DEBUG
        logging.getLogger("httpx").setLevel(logging.INFO)
        logging.getLogger("httpcore").setLevel(logging.INFO)
        # Réduire les logs de SQLAlchemy
        logging.getLogger("sqlalchemy.engine").setLevel(logging.INFO)
    
    logger.info(f"Logging initialized at {level} level")
    return logger

class LoggerAdapter(logging.LoggerAdapter):
    """Adapter pour ajouter automatiquement des données contextuelles aux logs"""
    
    def __init__(self, logger, extra=None):
        """
        Initialise l'adapter avec des métadonnées contextuelles
        
        Args:
            logger: Logger de base
            extra: Dictionnaire de métadonnées à injecter dans chaque log
        """
        super().__init__(logger, extra or {})
    
    def process(self, msg, kwargs):
        """Injecte les métadonnées dans chaque entrée de log"""
        # S'assurer que 'extra' existe dans kwargs
        if 'extra' not in kwargs:
            kwargs['extra'] = {}
        
        # Ajouter nos métadonnées
        for key, value in self.extra.items():
            kwargs['extra'][key] = value
        
        return msg, kwargs

def get_contextual_logger(name: str, **context) -> logging.Logger:
    """
    Crée un logger contextuel qui inclut les métadonnées spécifiées.
    
    Args:
        name: Nom du logger
        **context: Métadonnées contextuelles (user_id, bridge_item_id, etc.)
    
    Returns:
        Un logger adapté avec contexte
    """
    logger = logging.getLogger(name)
    return LoggerAdapter(logger, context)

def log_method_call(logger, method_name: str, **params):
    """
    Enregistre un appel de méthode avec ses paramètres
    
    Args:
        logger: Logger à utiliser
        method_name: Nom de la méthode appelée
        **params: Paramètres de la méthode
    """
    # Masquer les informations sensibles
    safe_params = {}
    for key, value in params.items():
        if key.lower() in ('token', 'password', 'secret', 'auth', 'key'):
            safe_params[key] = '******'
        else:
            safe_params[key] = value
    
    logger.debug(f"Calling {method_name} with params: {safe_params}")