"""
Configuration et utilitaires de logging pour le service d'enrichissement.

Ce module fournit une journalisation contextualisée et des fonctions
spécialisées pour tracer les opérations d'enrichissement.
"""

import logging
import sys
from typing import Dict, Any, Optional, Union
from datetime import datetime
from pathlib import Path

from enrichment_service.core.config import enrichment_settings

class EnrichmentLoggerAdapter(logging.LoggerAdapter):
    """
    Adaptateur de logger qui ajoute automatiquement le contexte d'enrichissement.
    
    Cet adaptateur enrichit chaque message de log avec des informations contextuelles
    comme l'ID utilisateur, l'ID de transaction, le type d'enrichissement, etc.
    """
    
    def __init__(self, logger: logging.Logger, extra: Dict[str, Any]):
        super().__init__(logger, extra)
        
    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple:
        """Traite le message et ajoute le contexte."""
        # Construire le préfixe contextuel
        context_parts = []
        
        # Ajouter les informations de base
        if self.extra.get("user_id"):
            context_parts.append(f"user={self.extra['user_id']}")
            
        if self.extra.get("transaction_id"):
            context_parts.append(f"tx={self.extra['transaction_id']}")
            
        if self.extra.get("enrichment_type"):
            context_parts.append(f"type={self.extra['enrichment_type']}")
            
        if self.extra.get("collection"):
            context_parts.append(f"coll={self.extra['collection']}")
            
        if self.extra.get("batch_id"):
            context_parts.append(f"batch={self.extra['batch_id']}")
            
        # Construire le message final
        if context_parts:
            context_str = "[" + " ".join(context_parts) + "]"
            formatted_msg = f"{context_str} {msg}"
        else:
            formatted_msg = msg
            
        return formatted_msg, kwargs

class EnrichmentFormatter(logging.Formatter):
    """Formateur personnalisé pour les logs du service d'enrichissement."""
    
    def __init__(self):
        # Format avec informations détaillées pour l'enrichissement
        fmt = (
            "%(asctime)s - %(name)s - %(levelname)s - "
            "%(module)s:%(funcName)s:%(lineno)d - %(message)s"
        )
        super().__init__(fmt, datefmt='%Y-%m-%d %H:%M:%S')
        
    def format(self, record: logging.LogRecord) -> str:
        """Formate l'enregistrement de log avec des informations supplémentaires."""
        # Ajouter des informations de performance si disponibles
        if hasattr(record, 'duration'):
            record.msg = f"{record.msg} (duration: {record.duration:.3f}s)"
            
        # Ajouter le nom du service pour distinction
        if not record.name.startswith('enrichment_service'):
            record.name = f"enrichment_service.{record.name}"
            
        return super().format(record)

def setup_enrichment_logging(
    level: Optional[str] = None,
    log_file: Optional[str] = None,
    enable_file_logging: Optional[bool] = None
) -> None:
    """
    Configure la journalisation pour le service d'enrichissement.
    
    Args:
        level: Niveau de journalisation (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Chemin vers le fichier de log (optionnel)
        enable_file_logging: Activer la journalisation vers fichier
    """
    # Utiliser les valeurs de configuration si non spécifiées
    level = level or enrichment_settings.log_level
    log_file = log_file or enrichment_settings.log_file
    enable_file_logging = enable_file_logging if enable_file_logging is not None else enrichment_settings.log_to_file
    
    # Convertir le niveau en objet logging
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Créer le formateur personnalisé
    formatter = EnrichmentFormatter()
    
    # Configuration du logger racine pour l'enrichissement
    enrichment_logger = logging.getLogger('enrichment_service')
    enrichment_logger.setLevel(log_level)
    
    # Supprimer les gestionnaires existants pour éviter les doublons
    for handler in enrichment_logger.handlers[:]:
        enrichment_logger.removeHandler(handler)
    
    # Gestionnaire pour la console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    enrichment_logger.addHandler(console_handler)
    
    # Gestionnaire pour fichier si demandé
    if enable_file_logging and log_file:
        try:
            # Créer le répertoire si nécessaire
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Créer le gestionnaire de fichier avec rotation si possible
            try:
                from logging.handlers import RotatingFileHandler
                file_handler = RotatingFileHandler(
                    log_file,
                    maxBytes=10*1024*1024,  # 10MB
                    backupCount=5
                )
            except ImportError:
                # Fallback vers FileHandler standard
                file_handler = logging.FileHandler(log_file)
            
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            enrichment_logger.addHandler(file_handler)
            
            enrichment_logger.info(f"Journalisation vers fichier activée: {log_file}")
            
        except Exception as e:
            enrichment_logger.error(f"Impossible de configurer la journalisation vers fichier: {e}")
    
    # Réduire le niveau de verbosité des loggers tiers
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)
    logging.getLogger('openai').setLevel(logging.WARNING)
    logging.getLogger('qdrant_client').setLevel(logging.WARNING)
    logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)
    logging.getLogger('asyncio').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    
    # Configuration spéciale pour les environnements de développement
    if enrichment_settings.is_development:
        logging.getLogger('enrichment_service.enrichers').setLevel(logging.DEBUG)
        logging.getLogger('enrichment_service.services').setLevel(logging.DEBUG)
        enrichment_logger.info("Mode développement activé - logs détaillés pour enrichers et services")
    
    enrichment_logger.info(f"Service d'enrichissement - journalisation configurée (niveau: {level})")

def get_contextual_logger(
    name: str,
    user_id: Optional[int] = None,
    transaction_id: Optional[Union[int, str]] = None,
    enrichment_type: Optional[str] = None,
    collection: Optional[str] = None,
    batch_id: Optional[str] = None,
    **extra_context
) -> EnrichmentLoggerAdapter:
    """
    Crée un logger contextuel pour les opérations d'enrichissement.
    
    Args:
        name: Nom du logger (généralement le nom du module)
        user_id: ID de l'utilisateur concerné
        transaction_id: ID de la transaction concernée
        enrichment_type: Type d'enrichissement (transaction, pattern, insight, etc.)
        collection: Nom de la collection Qdrant concernée
        batch_id: ID du lot de traitement
        **extra_context: Contexte supplémentaire
        
    Returns:
        EnrichmentLoggerAdapter: Logger avec contexte enrichi
    """
    logger = logging.getLogger(name)
    
    # Construire le contexte
    context = {
        "user_id": user_id,
        "transaction_id": transaction_id,
        "enrichment_type": enrichment_type,
        "collection": collection,
        "batch_id": batch_id,
        **extra_context
    }
    
    # Filtrer les valeurs None
    context = {k: v for k, v in context.items() if v is not None}
    
    return EnrichmentLoggerAdapter(logger, context)

def log_performance(func):
    """
    Décorateur pour mesurer et journaliser les performances des fonctions.
    
    Usage:
        @log_performance
        async def enrich_transaction(self, transaction):
            # Code de la fonction
    """
    import functools
    import time
    
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        logger = logging.getLogger(func.__module__)
        
        try:
            result = await func(*args, **kwargs)
            duration = time.time() - start_time
            logger.info(f"{func.__name__} completed successfully", extra={'duration': duration})
            return result
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"{func.__name__} failed after {duration:.3f}s: {str(e)}")
            raise
    
    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        logger = logging.getLogger(func.__module__)
        
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            logger.info(f"{func.__name__} completed successfully", extra={'duration': duration})
            return result
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"{func.__name__} failed after {duration:.3f}s: {str(e)}")
            raise
    
    # Retourner le wrapper approprié selon le type de fonction
    import inspect
    if inspect.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper

def log_enrichment_step(step_name: str, details: Optional[Dict[str, Any]] = None):
    """
    Décorateur pour journaliser les étapes d'enrichissement.
    
    Args:
        step_name: Nom de l'étape
        details: Détails supplémentaires à journaliser
    """
    def decorator(func):
        import functools
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            logger = logging.getLogger(func.__module__)
            logger.info(f"Début de l'étape: {step_name}")
            
            if details:
                for key, value in details.items():
                    logger.debug(f"  {key}: {value}")
            
            try:
                result = await func(*args, **kwargs)
                logger.info(f"Étape terminée avec succès: {step_name}")
                return result
            except Exception as e:
                logger.error(f"Échec de l'étape {step_name}: {str(e)}")
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            logger = logging.getLogger(func.__module__)
            logger.info(f"Début de l'étape: {step_name}")
            
            if details:
                for key, value in details.items():
                    logger.debug(f"  {key}: {value}")
            
            try:
                result = func(*args, **kwargs)
                logger.info(f"Étape terminée avec succès: {step_name}")
                return result
            except Exception as e:
                logger.error(f"Échec de l'étape {step_name}: {str(e)}")
                raise
        
        # Retourner le wrapper approprié
        import inspect
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

# Initialisation automatique si le module est importé
if enrichment_settings.log_level:
    setup_enrichment_logging()