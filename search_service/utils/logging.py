"""
Configuration du logging structuré pour le service de recherche.

Ce module fournit des fonctionnalités pour configurer les logs structurés
au format JSON en production et colorés en développement.
"""
import json
import logging
import os
import sys
from datetime import datetime
from typing import Dict, Any, Optional
import uuid
from contextvars import ContextVar

# Variable de contexte pour stocker l'ID de requête actuel
request_id_var = ContextVar("request_id", default=None)

class StructuredLogRecord(logging.LogRecord):
    """LogRecord personnalisé qui ajoute des champs structurés aux logs."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.timestamp = datetime.utcnow().isoformat()
        
        # Inclure l'ID de requête dans les logs s'il est défini
        request_id = request_id_var.get()
        if request_id:
            self.request_id = request_id

class StructuredFormatter(logging.Formatter):
    """Formateur qui convertit les logs en format JSON structuré."""
    
    def format(self, record):
        log_obj = {
            "timestamp": getattr(record, "timestamp", datetime.utcnow().isoformat()),
            "level": record.levelname,
            "service": "search_service",
            "module": record.module,
            "message": record.getMessage(),
        }
        
        # Ajouter des méta-données contextuelles si présentes
        if hasattr(record, "request_id"):
            log_obj["request_id"] = record.request_id
            
        if hasattr(record, "user_id"):
            log_obj["user_id"] = record.user_id
            
        if hasattr(record, "query"):
            log_obj["query"] = record.query
            
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
    """Formateur de logs avec couleurs pour la console en développement."""
    
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

def setup_structured_logging(level: Optional[str] = None) -> logging.Logger:
    """
    Configure le logging structuré pour le service de recherche.
    
    Args:
        level: Niveau de log (DEBUG, INFO, WARNING, ERROR, CRITICAL)
              Par défaut selon LOG_LEVEL ou INFO
    
    Returns:
        Logger configuré
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
    environment = os.environ.get("ENVIRONMENT", "development").lower()
    if environment == "development":
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
    logger = logging.getLogger("search_service")
    logger.setLevel(numeric_level)
    
    # Si en mode DEBUG, configurer des niveaux spécifiques pour certains modules
    if numeric_level == logging.DEBUG:
        # Réduire le bruit des logs httpx en mode DEBUG
        logging.getLogger("httpx").setLevel(logging.INFO)
        logging.getLogger("httpcore").setLevel(logging.INFO)
        # Réduire les logs de SQLAlchemy
        logging.getLogger("sqlalchemy.engine").setLevel(logging.INFO)
    
    logger.info(f"Logging search_service initialized at {level} level")
    return logger

def set_request_context(request_id: Optional[str] = None) -> str:
    """
    Définit l'ID de requête pour le contexte de logging actuel.
    
    Args:
        request_id: ID de requête à utiliser, ou nouveau UUID si None
        
    Returns:
        L'ID de requête utilisé
    """
    if request_id is None:
        request_id = str(uuid.uuid4())
    
    request_id_var.set(request_id)
    return request_id

def get_request_id() -> Optional[str]:
    """
    Récupère l'ID de requête du contexte actuel.
    
    Returns:
        ID de requête ou None
    """
    return request_id_var.get()

class LoggerAdapter(logging.LoggerAdapter):
    """Adapter pour ajouter automatiquement des données contextuelles aux logs."""
    
    def __init__(self, logger, extra=None):
        """
        Initialise l'adapter avec des métadonnées contextuelles.
        
        Args:
            logger: Logger de base
            extra: Dictionnaire de métadonnées à injecter dans chaque log
        """
        super().__init__(logger, extra or {})
    
    def process(self, msg, kwargs):
        """Injecte les métadonnées dans chaque entrée de log."""
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
        **context: Métadonnées contextuelles (user_id, query, etc.)
    
    Returns:
        Un logger adapté avec contexte
    """
    logger = logging.getLogger(name)
    return LoggerAdapter(logger, context)