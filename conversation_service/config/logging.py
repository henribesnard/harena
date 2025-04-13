"""
Configuration du logging pour l'application.

Ce module configure le logging pour l'application, avec différentes options
pour les environnements de développement et de production.
"""

import logging
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from .settings import settings


class JsonFormatter(logging.Formatter):
    """
    Formatter qui formate les logs en JSON.
    
    Cette classe permet de formater les logs au format JSON,
    facilitant leur intégration avec des outils d'analyse de logs.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Formate un log en JSON.
        
        Args:
            record: L'enregistrement de log à formater
            
        Returns:
            str: Log formaté en JSON
        """
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Ajouter les informations d'exception si présentes
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": self.formatException(record.exc_info)
            }
        
        # Ajouter les attributs extras
        for key, value in record.__dict__.items():
            if key not in ["args", "asctime", "created", "exc_info", "exc_text", 
                          "filename", "funcName", "id", "levelname", "levelno", 
                          "lineno", "module", "msecs", "message", "msg", 
                          "name", "pathname", "process", "processName", 
                          "relativeCreated", "stack_info", "thread", "threadName"]:
                log_data[key] = value
                
        return json.dumps(log_data)


def setup_logging() -> None:
    """
    Configure le système de logging pour l'application.
    
    Cette fonction configure les handlers, formatters et niveaux de log
    en fonction des paramètres de l'application.
    """
    # Niveau de log
    log_level = getattr(logging, settings.LOG_LEVEL)
    
    # Formatter
    if settings.DEBUG:
        # Format standard pour développement
        formatter = logging.Formatter(settings.LOG_FORMAT)
    else:
        # Format JSON pour production
        formatter = JsonFormatter()
    
    # Handler de console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # Handler de fichier si nécessaire
    file_handler = None
    if settings.LOG_TO_FILE:
        log_dir = settings.BASE_PATH / "logs"
        log_dir.mkdir(exist_ok=True)
        
        file_handler = logging.FileHandler(
            log_dir / settings.LOG_FILE,
            encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
    
    # Configuration du root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Supprimer les handlers existants
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Ajouter les nouveaux handlers
    root_logger.addHandler(console_handler)
    if file_handler:
        root_logger.addHandler(file_handler)
    
    # Logger spécifique à l'application
    app_logger = logging.getLogger("conversation_service")
    app_logger.setLevel(log_level)
    
    # Désactiver la propagation si on utilise un logger spécifique
    app_logger.propagate = False
    
    # Supprimer les handlers existants
    for handler in app_logger.handlers[:]:
        app_logger.removeHandler(handler)
    
    # Ajouter les nouveaux handlers
    app_logger.addHandler(console_handler)
    if file_handler:
        app_logger.addHandler(file_handler)
    
    # Log d'initialisation
    app_logger.info(f"Logging initialized with level: {settings.LOG_LEVEL}")


def get_logger(name: str) -> logging.Logger:
    """
    Obtient un logger configuré avec le nom donné.
    
    Args:
        name: Nom du logger, généralement __name__
        
    Returns:
        logging.Logger: Logger configuré
    """
    return logging.getLogger(f"conversation_service.{name}")