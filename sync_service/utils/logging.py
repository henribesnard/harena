"""
Utilitaires de journalisation.

Ce module fournit des fonctions pour faciliter la journalisation contextualisée.
"""

import logging
from typing import Dict, Any, Optional

def get_contextual_logger(logger_name: str, **context) -> logging.Logger:
    """
    Crée un logger avec un contexte additionnel qui sera inclus dans tous les messages.
    
    Args:
        logger_name: Nom du logger
        **context: Paramètres de contexte additionnels
        
    Returns:
        logging.Logger: Logger contextualisé
    """
    logger = logging.getLogger(logger_name)
    
    context_str = " ".join([f"{key}={value}" for key, value in context.items() if value is not None])
    
    # Créer un adaptateur de logger avec un préfixe
    class ContextAdapter(logging.LoggerAdapter):
        def process(self, msg, kwargs):
            if context_str:
                return f"[{context_str}] {msg}", kwargs
            return msg, kwargs
    
    return ContextAdapter(logger, {})

def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """
    Configure la journalisation globale.
    
    Args:
        level: Niveau de journalisation (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Chemin vers le fichier de log (optionnel)
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Configuration de base
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Ajouter un gestionnaire de fichier si demandé
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        
        # Ajouter le gestionnaire au logger racine
        logging.getLogger('').addHandler(file_handler)
    
    # Réduire le niveau de verbosité des loggers tiers
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('asyncio').setLevel(logging.WARNING)
    logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)