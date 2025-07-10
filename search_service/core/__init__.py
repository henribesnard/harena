"""
Module Core du Search Service
Contient les composants principaux du moteur de recherche lexicale
"""

from .query_executor import QueryExecutor
from .result_processor import ResultProcessor
from .performance_optimizer import (
    PerformanceOptimizer, 
    OptimizationLevel, 
    QueryPerformanceMetrics, 
    OptimizationRule
)
from .lexical_engine import LexicalEngine, LexicalEngineFactory

# Version du module Core
__version__ = "1.0.0"

# Exports principaux
__all__ = [
    # Composants principaux
    "QueryExecutor",
    "ResultProcessor", 
    "PerformanceOptimizer",
    "LexicalEngine",
    
    # Factory
    "LexicalEngineFactory",
    
    # Enums et classes utilitaires
    "OptimizationLevel",
    "QueryPerformanceMetrics",
    "OptimizationRule",
    
    # Version
    "__version__"
]

# Configuration par défaut du module
DEFAULT_OPTIMIZATION_LEVEL = OptimizationLevel.STANDARD
DEFAULT_CACHE_SIZE = 500
DEFAULT_TIMEOUT = 30

# Logging pour le module
import logging
logger = logging.getLogger(__name__)
logger.info(f"✅ Module Core v{__version__} chargé")